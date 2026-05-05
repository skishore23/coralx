"""GEPA-style prompt evolution for GSM8K without adapter training."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

from core.domain.prompt_genome import PromptGenome, prompt_genome_from_dict
from core.domain.proof import (
    validate_proof_execution_policy,
    validate_proof_quality,
)
from core.domain.stable_hash import stable_digest
from core.ports.interfaces import DatasetProvider, FitnessFn, ModelRunner
from plugins.gsm8k_lora.plugin import _answers_match, _extract_final_answer

_FINAL_MARKER_RE = re.compile(
    r"(?:final(?:\s+answer)?|answer)\s*[:=]\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
PROMPT_EVAL_VERSION = "gsm8k_prompt_v5"

SYSTEM_PROMPT_FAMILIES = (
    "You are an exact-answer solver. Output only one boxed number.",
    "You are a concise exact-answer math solver. Put the boxed answer first.",
    "You are a careful math solver. Solve grade-school arithmetic exactly.",
    "You are a concise GSM8K tutor. Track quantities and units carefully.",
    "You solve math word problems by translating each sentence into equations.",
    "You are an exact-answer math assistant. Avoid guessing and arithmetic slips.",
)

REASONING_INSTRUCTIONS = (
    "Do the arithmetic internally. Do not show steps.",
    "Compute privately. Output the answer first, then at most three short check lines.",
    "Use compact arithmetic. Avoid numbered list labels unless they are calculations.",
    "Reason step by step. Keep each arithmetic step explicit.",
    "Identify the unknown, write the needed computations, then solve.",
    "Work through the problem carefully and check the arithmetic before answering.",
    "Use a short chain of thought with enough detail to make the calculation auditable.",
)

ANSWER_FORMAT_INSTRUCTIONS = (
    "Output exactly one line: \\boxed{number}. No explanation.",
    "The first line must be exactly \\boxed{number}.",
    "Start with \\boxed{number}; put any explanation after it.",
    "Put only the final numeric answer inside \\boxed{} at the end.",
    "End your response with Final: \\boxed{number}.",
    "After the reasoning, write the final answer as \\boxed{number}.",
)

VERIFICATION_INSTRUCTIONS = (
    "Check the answer before writing it, but do not show the check.",
    "If space is limited, skip explanation and write the boxed answer.",
    "After the boxed answer, give only the arithmetic check needed to justify it.",
    "Before the final answer, verify the calculation once.",
    "Check that the answer has the right units and magnitude.",
    "Do a quick reverse check before writing the boxed answer.",
    "If multiple operations are required, verify the intermediate totals.",
)


@dataclass(frozen=True)
class GSM8KPromptSettings:
    """Runtime settings for prompt evolution."""

    model_name: str
    max_seq_length: int
    reflection_samples: int
    dev_samples: int
    test_samples: int
    max_new_tokens: int
    seed: int
    output_dir: Path
    device: str
    temperature_range: tuple[float, float]
    top_p_range: tuple[float, float]
    max_new_tokens_range: tuple[int, int]
    self_consistency_values: tuple[int, ...]
    few_shot_count: int
    use_chat_template: bool
    resume: bool


@dataclass(frozen=True)
class PromptEvaluationMetrics:
    """Metrics for one prompt candidate on one split."""

    exact_accuracy: float
    token_efficiency: float
    latency_score: float
    formatting_success_rate: float
    novelty: float
    avg_generated_tokens: float
    latency_seconds: float
    predictions: tuple[dict[str, Any], ...]
    cache_key: str
    from_cache: bool = False

    def fitness(self) -> float:
        """Return the scalar used for within-suite ranking."""
        return (
            self.exact_accuracy
            + 0.04 * self.formatting_success_rate
            + 0.02 * self.token_efficiency
            + 0.01 * self.latency_score
            + 0.01 * self.novelty
        )

    def to_report_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics record."""
        return {
            "exact_accuracy": self.exact_accuracy,
            "token_efficiency": self.token_efficiency,
            "latency_score": self.latency_score,
            "formatting_success_rate": self.formatting_success_rate,
            "novelty": self.novelty,
            "avg_generated_tokens": self.avg_generated_tokens,
            "latency_seconds": self.latency_seconds,
            "cache_key": self.cache_key,
            "from_cache": self.from_cache,
            "predictions": list(self.predictions),
        }


def _optional_ml_imports() -> dict[str, Any]:
    """Import optional ML dependencies with a clear error message."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "The gsm8k_prompt_evolution target requires optional ML dependencies. "
            "Install with: python -m pip install -e '.[ml]'"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _settings(config: dict[str, Any]) -> GSM8KPromptSettings:
    experiment = config.get("experiment", {})
    evaluation = experiment.get("evaluation", {}) or {}
    execution = config.get("execution", {}) or {}
    cache = config.get("cache", {}) or {}
    model = experiment.get("model", {}) or {}
    cheap_knobs = config.get("cheap_knobs", {}) or {}

    output_dir = Path(
        str(execution.get("output_dir") or cache.get("artifacts_dir") or "./artifacts")
    )
    token_range = cheap_knobs.get("max_tokens_range") or [
        evaluation.get("max_new_tokens", 256),
        evaluation.get("max_new_tokens", 256),
    ]

    return GSM8KPromptSettings(
        model_name=str(model.get("name", "Qwen/Qwen2.5-Math-1.5B-Instruct")),
        max_seq_length=int(model.get("max_seq_length", 768)),
        reflection_samples=int(evaluation.get("reflection_samples", 128)),
        dev_samples=int(evaluation.get("dev_samples", 128)),
        test_samples=int(evaluation.get("test_samples", 256)),
        max_new_tokens=int(evaluation.get("max_new_tokens", 256)),
        seed=int(config.get("seed", 42)),
        output_dir=output_dir,
        device=str(evaluation.get("device", "auto")),
        temperature_range=tuple(
            float(v) for v in cheap_knobs.get("temperature_range", [0.0, 0.7])
        ),
        top_p_range=tuple(float(v) for v in cheap_knobs.get("top_p_range", [0.8, 1.0])),
        max_new_tokens_range=(int(token_range[0]), int(token_range[1])),
        self_consistency_values=tuple(
            int(v) for v in evaluation.get("self_consistency_values", [1])
        ),
        few_shot_count=int(evaluation.get("few_shot_count", 2)),
        use_chat_template=bool(evaluation.get("use_chat_template", True)),
        resume=bool(evaluation.get("resume", True)),
    )


def _row_id(prefix: str, index: int, row: dict[str, Any]) -> str:
    return f"{prefix}_{index:05d}_{stable_digest(row['question'], 8)}"


def _normalize_row(prefix: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row["answer"])
    return {
        "id": _row_id(prefix, index, row),
        "question": str(row["question"]),
        "answer": answer,
        "final_answer": _extract_final_answer(answer),
    }


def split_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    """Return a stable fingerprint for a dataset split."""
    return stable_digest(
        [
            {
                "id": row.get("id"),
                "question": row["question"],
                "final_answer": row.get("final_answer"),
            }
            for row in rows
        ],
        length=20,
    )


def assert_no_test_leakage(
    reflection_rows: Iterable[dict[str, Any]],
    dev_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
) -> None:
    """Raise if test questions appear in optimization splits."""
    optimization_questions = {
        row["question"] for row in list(reflection_rows) + list(dev_rows)
    }
    leaked = optimization_questions.intersection(row["question"] for row in test_rows)
    if leaked:
        sample = next(iter(leaked))
        raise ValueError(f"Test-set leakage detected for question: {sample[:120]}")


def extract_prompt_final_answer(text: str) -> str | None:
    """Extract only explicitly final GSM8K answers from prompt-evolution output."""
    boxed_answer = _extract_final_answer(text) if "\\boxed" in text else None
    if boxed_answer is not None:
        return boxed_answer
    if "####" in text:
        return _extract_final_answer(text.split("####", maxsplit=1)[1])
    matches = _FINAL_MARKER_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    return None


class GSM8KPromptDataset(DatasetProvider):
    """GSM8K provider with deterministic reflection/dev/test slices."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.settings = _settings(config)

    def problems(self) -> Iterable[dict[str, Any]]:
        deps = _optional_ml_imports()
        dataset = deps["load_dataset"]("openai/gsm8k", "main")
        train_split = dataset["train"].shuffle(seed=self.settings.seed)
        test_split = dataset["test"].shuffle(seed=self.settings.seed + 1)

        reflection_count = min(self.settings.reflection_samples, len(train_split))
        dev_count = min(
            self.settings.dev_samples, max(len(train_split) - reflection_count, 0)
        )
        test_count = min(self.settings.test_samples, len(test_split))

        reflection_rows = [
            _normalize_row("reflection", index, row)
            for index, row in enumerate(train_split.select(range(reflection_count)))
        ]
        dev_rows = [
            _normalize_row("dev", index, row)
            for index, row in enumerate(
                train_split.select(range(reflection_count, reflection_count + dev_count))
            )
        ]
        test_rows = [
            _normalize_row("test", index, row)
            for index, row in enumerate(test_split.select(range(test_count)))
        ]
        assert_no_test_leakage(reflection_rows, dev_rows, test_rows)

        yield {
            "name": "gsm8k_prompt_evolution",
            "dataset": "openai/gsm8k",
            "reflection": reflection_rows,
            "dev": dev_rows,
            "test": test_rows,
            "fingerprints": {
                "reflection": split_fingerprint(reflection_rows),
                "dev": split_fingerprint(dev_rows),
                "test": split_fingerprint(test_rows),
            },
        }


def create_initial_prompt_population(
    *,
    population_size: int,
    seed: int,
    reflection_rows: list[dict[str, Any]],
    settings: GSM8KPromptSettings,
    origin: str = "ca_neat",
) -> list[PromptGenome]:
    """Create deterministic CA-flavored prompt genomes."""
    genomes = []
    example_ids = [row["id"] for row in reflection_rows]
    for index in range(population_size):
        rng = Random(seed + index * 7919)
        if index == 0:
            few_shots = ()
        else:
            few_shots = tuple(
                rng.sample(example_ids, min(settings.few_shot_count, len(example_ids)))
            )
        order = list(range(len(few_shots)))
        rng.shuffle(order)
        genome = PromptGenome(
            id=f"gen0_prompt{index:04d}",
            system_prompt=SYSTEM_PROMPT_FAMILIES[index % len(SYSTEM_PROMPT_FAMILIES)],
            reasoning_instruction=REASONING_INSTRUCTIONS[0]
            if index == 0
            else REASONING_INSTRUCTIONS[rng.randrange(len(REASONING_INSTRUCTIONS))],
            answer_format_instruction=ANSWER_FORMAT_INSTRUCTIONS[0]
            if index == 0
            else ANSWER_FORMAT_INSTRUCTIONS[rng.randrange(len(ANSWER_FORMAT_INSTRUCTIONS))],
            verification_instruction=VERIFICATION_INSTRUCTIONS[0]
            if index == 0
            else VERIFICATION_INSTRUCTIONS[rng.randrange(len(VERIFICATION_INSTRUCTIONS))],
            few_shot_example_ids=few_shots,
            few_shot_order=tuple(order),
            temperature=_sample_float(rng, settings.temperature_range),
            top_p=_sample_float(rng, settings.top_p_range),
            max_new_tokens=rng.randint(
                settings.max_new_tokens_range[0], settings.max_new_tokens_range[1]
            ),
            self_consistency_n=rng.choice(settings.self_consistency_values),
            origin=origin,
            generation=0,
            seed=seed + index,
        )
        genomes.append(genome)
    return genomes


def base_prompt_genome(settings: GSM8KPromptSettings) -> PromptGenome:
    """Return the fixed base prompt baseline."""
    return PromptGenome(
        id="base_prompt",
        system_prompt="You are a helpful assistant.",
        reasoning_instruction="Solve the problem.",
        answer_format_instruction="Give the final answer.",
        verification_instruction="",
        few_shot_example_ids=(),
        few_shot_order=(),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=settings.max_new_tokens,
        self_consistency_n=1,
        origin="base",
        seed=settings.seed,
    )


def strong_cot_prompt_genome(settings: GSM8KPromptSettings) -> PromptGenome:
    """Return the handwritten strong CoT prompt baseline."""
    return PromptGenome(
        id="strong_cot_prompt",
        system_prompt="You are a careful math solver. Solve GSM8K problems exactly.",
        reasoning_instruction=(
            "Do the arithmetic internally. Do not show steps."
        ),
        answer_format_instruction=(
            "Output exactly one line: \\boxed{number}. No explanation."
        ),
        verification_instruction=(
            "Check the answer before writing it, but do not show the check."
        ),
        few_shot_example_ids=(),
        few_shot_order=(),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=settings.max_new_tokens,
        self_consistency_n=1,
        origin="strong_cot",
        seed=settings.seed,
    )


def random_prompt_genome(
    *,
    genome_id: str,
    seed: int,
    reflection_rows: list[dict[str, Any]],
    settings: GSM8KPromptSettings,
    generation: int = 0,
    origin: str = "random",
) -> PromptGenome:
    """Create one random prompt/decode candidate."""
    rng = Random(seed)
    example_ids = [row["id"] for row in reflection_rows]
    few_shot_count = rng.randint(0, min(settings.few_shot_count, len(example_ids)))
    few_shots = tuple(rng.sample(example_ids, few_shot_count))
    order = list(range(len(few_shots)))
    rng.shuffle(order)
    return PromptGenome(
        id=genome_id,
        system_prompt=rng.choice(SYSTEM_PROMPT_FAMILIES[:2]),
        reasoning_instruction=rng.choice(REASONING_INSTRUCTIONS[:2]),
        answer_format_instruction=rng.choice(ANSWER_FORMAT_INSTRUCTIONS[:2]),
        verification_instruction=rng.choice(VERIFICATION_INSTRUCTIONS[:2]),
        few_shot_example_ids=few_shots,
        few_shot_order=tuple(order),
        temperature=_sample_float(rng, settings.temperature_range),
        top_p=_sample_float(rng, settings.top_p_range),
        max_new_tokens=rng.randint(
            settings.max_new_tokens_range[0], settings.max_new_tokens_range[1]
        ),
        self_consistency_n=rng.choice(settings.self_consistency_values),
        origin=origin,
        generation=generation,
        seed=seed,
    )


class GSM8KPromptRunner(ModelRunner):
    """Run base-model GSM8K inference for prompt candidates."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.settings = _settings(config)
        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: Any | None = None

    def generate(
        self, prompt: str, max_tokens: int, cheap_knobs: Any | None = None
    ) -> str:
        self._ensure_model()
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_seq_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][encoded["input_ids"].shape[-1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True))

    def evaluate(
        self,
        genome: PromptGenome,
        rows: list[dict[str, Any]],
        reflection_rows: list[dict[str, Any]],
        split_name: str,
        novelty: float = 0.0,
    ) -> PromptEvaluationMetrics:
        """Evaluate a prompt genome on a split with disk caching."""
        cache_key = self.cache_key(genome, rows, reflection_rows, split_name)
        cached = self._read_disk_cache(cache_key)
        if cached is not None:
            return PromptEvaluationMetrics(
                exact_accuracy=cached.exact_accuracy,
                token_efficiency=cached.token_efficiency,
                latency_score=cached.latency_score,
                formatting_success_rate=cached.formatting_success_rate,
                novelty=novelty,
                avg_generated_tokens=cached.avg_generated_tokens,
                latency_seconds=cached.latency_seconds,
                predictions=cached.predictions,
                cache_key=cache_key,
                from_cache=True,
            )

        self._ensure_model()
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None

        reflection_by_id = {row["id"]: row for row in reflection_rows}
        predictions = []
        correct = 0
        formatted = 0
        generated_token_counts = []
        started = time.time()

        for index, row in enumerate(rows):
            prompt = _render_prompt(
                genome,
                row["question"],
                reflection_by_id,
                self._tokenizer if self.settings.use_chat_template else None,
            )
            completions = []
            parsed_answers = []
            for sample_index in range(genome.self_consistency_n):
                generated, token_count = self._generate_once(
                    prompt, genome, sample_index
                )
                completions.append(generated)
                generated_token_counts.append(token_count)
                parsed_answers.append(_extract_generated_answer(prompt, generated))

            prediction = _majority_answer(parsed_answers)
            reference = row["final_answer"]
            is_correct = _answers_match(prediction, reference)
            correct += int(is_correct)
            formatted += int(prediction is not None)
            predictions.append(
                {
                    "index": index,
                    "question": row["question"],
                    "prediction": prediction,
                    "reference": reference,
                    "correct": is_correct,
                    "parse_result": "parsed" if prediction is not None else "missing",
                    "generated": completions[0][:800] if completions else "",
                }
            )

        latency_seconds = time.time() - started
        total = max(len(rows), 1)
        avg_tokens = statistics.mean(generated_token_counts) if generated_token_counts else 0.0
        metrics = PromptEvaluationMetrics(
            exact_accuracy=correct / total,
            token_efficiency=1.0 / (1.0 + avg_tokens / 256.0),
            latency_score=1.0 / (1.0 + latency_seconds / total),
            formatting_success_rate=formatted / total,
            novelty=novelty,
            avg_generated_tokens=avg_tokens,
            latency_seconds=latency_seconds,
            predictions=tuple(predictions),
            cache_key=cache_key,
        )
        self._write_disk_cache(metrics)
        return metrics

    def cache_key(
        self,
        genome: PromptGenome,
        rows: list[dict[str, Any]],
        reflection_rows: list[dict[str, Any]],
        split_name: str,
    ) -> str:
        """Return the complete cache identity for a candidate evaluation."""
        fingerprint = stable_digest(
            {
                "split": split_name,
                "rows": split_fingerprint(rows),
                "reflection": split_fingerprint(reflection_rows),
                "render": {
                    "use_chat_template": self.settings.use_chat_template,
                    "max_seq_length": self.settings.max_seq_length,
                },
            },
            length=20,
        )
        return genome.cache_key(
            model=self.settings.model_name,
            split_fingerprint=fingerprint,
            seed=self.settings.seed,
            version=PROMPT_EVAL_VERSION,
        )

    def _generate_once(
        self, prompt: str, genome: PromptGenome, sample_index: int
    ) -> tuple[str, int]:
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_seq_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        if genome.temperature > 0:
            self._torch.manual_seed(genome.seed + sample_index)
        generation_kwargs = {
            "max_new_tokens": genome.max_new_tokens,
            "do_sample": genome.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if genome.temperature > 0:
            generation_kwargs["temperature"] = max(genome.temperature, 1e-5)
            generation_kwargs["top_p"] = genome.top_p
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                **generation_kwargs,
            )
        new_tokens = output_ids[0][encoded["input_ids"].shape[-1] :]
        return (
            str(self._tokenizer.decode(new_tokens, skip_special_tokens=True)),
            int(new_tokens.shape[-1]),
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        deps = _optional_ml_imports()
        torch = deps["torch"]
        tokenizer = deps["AutoTokenizer"].from_pretrained(self.settings.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = deps["AutoModelForCausalLM"].from_pretrained(self.settings.model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        device = self._resolve_device(torch)
        model.to(device)
        model.eval()
        torch.manual_seed(self.settings.seed)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _resolve_device(self, torch: Any) -> Any:
        requested = self.settings.device
        if requested != "auto":
            return torch.device(requested)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _cache_path(self, cache_key: str) -> Path:
        return self.settings.output_dir / "gsm8k_prompt_cache" / f"{cache_key}.json"

    def _read_disk_cache(self, cache_key: str) -> PromptEvaluationMetrics | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return PromptEvaluationMetrics(
            exact_accuracy=float(data["exact_accuracy"]),
            token_efficiency=float(data["token_efficiency"]),
            latency_score=float(data["latency_score"]),
            formatting_success_rate=float(data["formatting_success_rate"]),
            novelty=float(data.get("novelty", 0.0)),
            avg_generated_tokens=float(data["avg_generated_tokens"]),
            latency_seconds=float(data["latency_seconds"]),
            predictions=tuple(data.get("predictions", [])),
            cache_key=cache_key,
            from_cache=True,
        )

    def _write_disk_cache(self, metrics: PromptEvaluationMetrics) -> None:
        path = self._cache_path(metrics.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics.to_report_dict(), indent=2, sort_keys=True))


def _render_prompt(
    genome: PromptGenome,
    question: str,
    reflection_by_id: dict[str, dict[str, Any]],
    tokenizer: Any | None = None,
) -> str:
    instructions = "\n".join(
        part
        for part in (
            genome.reasoning_instruction,
            genome.verification_instruction,
            genome.answer_format_instruction,
        )
        if part
    )
    examples = []
    ordered_ids = [
        genome.few_shot_example_ids[index]
        for index in genome.few_shot_order
        if index < len(genome.few_shot_example_ids)
    ]
    for example_id in ordered_ids:
        row = reflection_by_id.get(example_id)
        if row is None:
            continue
        examples.append(
            "Question: "
            f"{row['question']}\nAnswer: {row['answer'].split('####')[0].strip()}\n"
            f"\\boxed{{{row['final_answer']}}}"
        )
    user_content = "\n\n".join(
        part
        for part in (
            instructions,
            "\n\n".join(examples),
            f"Question: {question}\nReturn only the number and close the brace.\nAnswer: \\boxed{{",
        )
        if part
    )
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return str(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": genome.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return f"{genome.system_prompt}\n\n{user_content}"


def _extract_generated_answer(prompt: str, generated: str) -> str | None:
    """Extract a final answer, honoring answer-prefix prompts."""
    if prompt.rstrip().endswith("\\boxed{"):
        return extract_prompt_final_answer("\\boxed{" + generated)
    return extract_prompt_final_answer(generated)


def _majority_answer(parsed_answers: list[str | None]) -> str | None:
    answers = [answer for answer in parsed_answers if answer is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def _sample_float(rng: Random, value_range: tuple[float, float]) -> float:
    if value_range[0] == value_range[1]:
        return float(value_range[0])
    return round(rng.uniform(value_range[0], value_range[1]), 4)


def _record_for_candidate(
    *,
    genome: PromptGenome,
    metrics: PromptEvaluationMetrics,
    suite: str,
    split: str,
) -> dict[str, Any]:
    return {
        "candidate_id": genome.id,
        "suite": suite,
        "split": split,
        "generation": genome.generation,
        "origin": genome.origin,
        "fitness": metrics.fitness(),
        "evaluation_version": PROMPT_EVAL_VERSION,
        "genome": genome.prompt_payload(),
        "metrics": metrics.to_report_dict(),
    }


def load_completed_candidate_records(
    path: Path,
) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    """Load completed records from a partial candidate JSONL file."""
    if not path.exists():
        return {}
    completed = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (
            str(record["suite"]),
            str(record["split"]),
            str(record["candidate_id"]),
            stable_digest(record.get("genome", {}), length=20),
            str(record.get("evaluation_version", "unknown")),
            _record_cache_key(record),
        )
        completed[key] = record
    return completed


def _record_cache_key(record: dict[str, Any]) -> str:
    metrics = record.get("metrics")
    if isinstance(metrics, dict) and metrics.get("cache_key"):
        return str(metrics["cache_key"])
    return str(record.get("evaluation_cache_key", "unknown"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


class PromptProofLogger:
    """Artifact writer with resume awareness."""

    def __init__(self, output_dir: Path, resume: bool):
        self.output_dir = output_dir
        self.candidate_jsonl = output_dir / "candidate_evaluations.jsonl"
        self.failure_jsonl = output_dir / "failure_traces.jsonl"
        self.completed = (
            load_completed_candidate_records(self.candidate_jsonl) if resume else {}
        )

    def evaluate_and_log(
        self,
        *,
        runner: GSM8KPromptRunner,
        genome: PromptGenome,
        rows: list[dict[str, Any]],
        reflection_rows: list[dict[str, Any]],
        suite: str,
        split: str,
        novelty: float,
    ) -> dict[str, Any]:
        split_name = f"{suite}:{split}"
        evaluation_cache_key = runner.cache_key(
            genome,
            rows,
            reflection_rows,
            split_name=split_name,
        )
        key = (
            suite,
            split,
            genome.id,
            genome.structural_key(),
            PROMPT_EVAL_VERSION,
            evaluation_cache_key,
        )
        if key in self.completed:
            return self.completed[key]

        metrics = runner.evaluate(
            genome,
            rows,
            reflection_rows,
            split_name=split_name,
            novelty=novelty,
        )
        record = _record_for_candidate(
            genome=genome, metrics=metrics, suite=suite, split=split
        )
        append_jsonl(self.candidate_jsonl, record)
        self.completed[key] = record
        self._log_failures(genome, metrics, suite, split)
        return record

    def _log_failures(
        self,
        genome: PromptGenome,
        metrics: PromptEvaluationMetrics,
        suite: str,
        split: str,
    ) -> None:
        for prediction in metrics.predictions:
            if prediction.get("correct"):
                continue
            append_jsonl(
                self.failure_jsonl,
                {
                    "candidate_id": genome.id,
                    "suite": suite,
                    "split": split,
                    "question": prediction.get("question"),
                    "reference_answer": prediction.get("reference"),
                    "candidate_generated_reasoning": prediction.get("generated"),
                    "candidate_final_answer": prediction.get("prediction"),
                    "parse_result": prediction.get("parse_result"),
                    "why_it_failed": _classify_failure(prediction),
                },
            )


def _classify_failure(prediction: dict[str, Any]) -> str:
    if prediction.get("prediction") is None:
        return "No parseable final numeric answer."
    return "Parsed final answer did not match the reference answer."


def _novelty(genome: PromptGenome, existing: list[PromptGenome]) -> float:
    if not existing:
        return 1.0
    key = genome.structural_key()
    distances = []
    for other in existing:
        other_key = other.structural_key()
        diff = sum(left != right for left, right in zip(key, other_key, strict=False))
        distances.append(diff / max(len(key), 1))
    return min(distances)


def _metric(record: dict[str, Any], name: str) -> float:
    return float(record["metrics"].get(name, 0.0))


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    objectives = (
        "exact_accuracy",
        "token_efficiency",
        "latency_score",
        "formatting_success_rate",
        "novelty",
    )
    at_least_one = False
    for objective in objectives:
        left_value = _metric(left, objective)
        right_value = _metric(right, objective)
        if left_value < right_value:
            return False
        if left_value > right_value:
            at_least_one = True
    return at_least_one


def pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated candidate records."""
    front = []
    for candidate in records:
        if any(_dominates(other, candidate) for other in records if other is not candidate):
            continue
        front.append(candidate)
    return sorted(
        front,
        key=lambda row: (
            _metric(row, "exact_accuracy"),
            row.get("fitness") or 0.0,
            _metric(row, "formatting_success_rate"),
        ),
        reverse=True,
    )


def _best_by_exact(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(
        records,
        key=lambda row: (
            _metric(row, "exact_accuracy"),
            row.get("fitness") or 0.0,
            _metric(row, "formatting_success_rate"),
        ),
    )


def _select_elite_records(
    records: list[dict[str, Any]], count: int, base_exact: float
) -> list[dict[str, Any]]:
    eligible = [row for row in pareto_front(records) if _metric(row, "exact_accuracy") >= base_exact]
    if not eligible:
        eligible = pareto_front(records)
    return eligible[:count]


def mutate_prompt_genome(
    parent: PromptGenome,
    *,
    genome_id: str,
    generation: int,
    seed: int,
    reflection_rows: list[dict[str, Any]],
    settings: GSM8KPromptSettings,
    origin: str = "mutation",
) -> PromptGenome:
    """Apply one prompt or decoding mutation."""
    rng = Random(seed)
    data = {
        "system_prompt": parent.system_prompt,
        "reasoning_instruction": parent.reasoning_instruction,
        "answer_format_instruction": parent.answer_format_instruction,
        "verification_instruction": parent.verification_instruction,
        "few_shot_example_ids": parent.few_shot_example_ids,
        "few_shot_order": parent.few_shot_order,
        "temperature": parent.temperature,
        "top_p": parent.top_p,
        "max_new_tokens": parent.max_new_tokens,
        "self_consistency_n": parent.self_consistency_n,
    }
    mutation = rng.choice(
        [
            "system",
            "reasoning",
            "answer_format",
            "verification",
            "few_shot",
            "temperature",
            "top_p",
            "max_new_tokens",
            "self_consistency",
        ]
    )
    if mutation == "system":
        data["system_prompt"] = rng.choice(SYSTEM_PROMPT_FAMILIES)
    elif mutation == "reasoning":
        data["reasoning_instruction"] = rng.choice(REASONING_INSTRUCTIONS)
    elif mutation == "answer_format":
        data["answer_format_instruction"] = rng.choice(ANSWER_FORMAT_INSTRUCTIONS)
    elif mutation == "verification":
        data["verification_instruction"] = rng.choice(VERIFICATION_INSTRUCTIONS)
    elif mutation == "few_shot":
        example_ids = [row["id"] for row in reflection_rows]
        count = min(settings.few_shot_count, len(example_ids))
        data["few_shot_example_ids"] = tuple(rng.sample(example_ids, count))
        order = list(range(count))
        rng.shuffle(order)
        data["few_shot_order"] = tuple(order)
    elif mutation == "temperature":
        data["temperature"] = _sample_float(rng, settings.temperature_range)
    elif mutation == "top_p":
        data["top_p"] = _sample_float(rng, settings.top_p_range)
    elif mutation == "max_new_tokens":
        data["max_new_tokens"] = rng.randint(
            settings.max_new_tokens_range[0], settings.max_new_tokens_range[1]
        )
    elif mutation == "self_consistency":
        data["self_consistency_n"] = rng.choice(settings.self_consistency_values)

    return PromptGenome(
        id=genome_id,
        generation=generation,
        origin=origin,
        seed=seed,
        **data,
    )


def crossover_prompt_genomes(
    left: PromptGenome,
    right: PromptGenome,
    *,
    genome_id: str,
    generation: int,
    seed: int,
) -> PromptGenome:
    """Merge prompt components from two parents."""
    rng = Random(seed)
    return PromptGenome(
        id=genome_id,
        system_prompt=rng.choice([left.system_prompt, right.system_prompt]),
        reasoning_instruction=rng.choice(
            [left.reasoning_instruction, right.reasoning_instruction]
        ),
        answer_format_instruction=rng.choice(
            [left.answer_format_instruction, right.answer_format_instruction]
        ),
        verification_instruction=rng.choice(
            [left.verification_instruction, right.verification_instruction]
        ),
        few_shot_example_ids=rng.choice(
            [left.few_shot_example_ids, right.few_shot_example_ids]
        ),
        few_shot_order=rng.choice([left.few_shot_order, right.few_shot_order]),
        temperature=rng.choice([left.temperature, right.temperature]),
        top_p=rng.choice([left.top_p, right.top_p]),
        max_new_tokens=rng.choice([left.max_new_tokens, right.max_new_tokens]),
        self_consistency_n=rng.choice(
            [left.self_consistency_n, right.self_consistency_n]
        ),
        origin="crossover",
        generation=generation,
        seed=seed,
    )


def reflective_mutation(
    parent: PromptGenome,
    failure_traces: list[dict[str, Any]],
    *,
    genome_id: str,
    generation: int,
    seed: int,
) -> PromptGenome:
    """GEPA-style reflective mutation from observed failures."""
    missing = sum(1 for trace in failure_traces if trace.get("parse_result") == "missing")
    arithmetic = len(failure_traces) - missing
    if missing > arithmetic:
        revised_format = (
            "Output exactly one line: \\boxed{number}. No explanation."
        )
        revised_reasoning = (
            "Do the arithmetic internally. Do not show steps."
        )
    else:
        revised_format = "Output exactly one line: \\boxed{number}. No explanation."
        revised_reasoning = (
            "Do the arithmetic internally. Do not show steps."
        )
    revised_verification = (
        "Check the answer before writing it, but do not show the check."
    )
    return PromptGenome(
        id=genome_id,
        system_prompt=parent.system_prompt,
        reasoning_instruction=revised_reasoning,
        answer_format_instruction=revised_format,
        verification_instruction=revised_verification,
        few_shot_example_ids=parent.few_shot_example_ids,
        few_shot_order=parent.few_shot_order,
        temperature=parent.temperature,
        top_p=parent.top_p,
        max_new_tokens=parent.max_new_tokens,
        self_consistency_n=parent.self_consistency_n,
        origin="gepa_reflection",
        generation=generation,
        seed=seed,
    )


def _failure_traces_from_record(record: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    traces = []
    for prediction in record["metrics"].get("predictions", []):
        if prediction.get("correct"):
            continue
        traces.append(
            {
                "question": prediction.get("question"),
                "reference_answer": prediction.get("reference"),
                "candidate_generated_reasoning": prediction.get("generated"),
                "candidate_final_answer": prediction.get("prediction"),
                "parse_result": prediction.get("parse_result"),
                "why_it_failed": _classify_failure(prediction),
            }
        )
        if len(traces) >= limit:
            break
    return traces


def _make_next_generation(
    *,
    mode: str,
    current: list[PromptGenome],
    records: list[dict[str, Any]],
    base_exact: float,
    generation: int,
    population_size: int,
    seed: int,
    reflection_rows: list[dict[str, Any]],
    settings: GSM8KPromptSettings,
) -> list[PromptGenome]:
    elite_count = max(1, math.ceil(population_size * 0.4))
    elites = _select_elite_records(records, elite_count, base_exact)
    by_id = {genome.id: genome for genome in current}
    parents = [by_id[row["candidate_id"]] for row in elites if row["candidate_id"] in by_id]
    if not parents:
        best = _best_by_exact(records)
        parents = [by_id[best["candidate_id"]]] if best and best["candidate_id"] in by_id else current[:1]

    next_population = [
        parent.with_generation(generation, "elite", f"gen{generation}_elite{i:04d}")
        for i, parent in enumerate(parents[:elite_count])
    ]
    rng = Random(seed + generation * 104729)
    record_by_id = {record["candidate_id"]: record for record in records}

    def add(child: PromptGenome) -> None:
        if len(next_population) < population_size:
            next_population.append(child)

    while len(next_population) < population_size:
        index = len(next_population)
        child_id = f"gen{generation}_{mode}_{index:04d}"
        if mode == "gepa_only":
            parent = rng.choice(parents)
            parent_record = record_by_id.get(parent.id, records[0])
            child = reflective_mutation(
                parent,
                _failure_traces_from_record(parent_record),
                genome_id=child_id,
                generation=generation,
                seed=seed + generation * 1000 + index,
            )
        elif mode == "ca_neat":
            if len(parents) >= 2 and rng.random() < 0.5:
                child = crossover_prompt_genomes(
                    rng.choice(parents),
                    rng.choice(parents),
                    genome_id=child_id,
                    generation=generation,
                    seed=seed + generation * 1000 + index,
                )
            else:
                child = mutate_prompt_genome(
                    rng.choice(parents),
                    genome_id=child_id,
                    generation=generation,
                    seed=seed + generation * 1000 + index,
                    reflection_rows=reflection_rows,
                    settings=settings,
                    origin="ca_neat_mutation",
                )
        else:
            draw = rng.random()
            if draw < 0.2:
                parent = rng.choice(parents)
                parent_record = record_by_id.get(parent.id, records[0])
                child = reflective_mutation(
                    parent,
                    _failure_traces_from_record(parent_record),
                    genome_id=child_id,
                    generation=generation,
                    seed=seed + generation * 1000 + index,
                )
            elif draw < 0.4 and len(parents) >= 2:
                child = crossover_prompt_genomes(
                    rng.choice(parents),
                    rng.choice(parents),
                    genome_id=child_id,
                    generation=generation,
                    seed=seed + generation * 1000 + index,
                )
            elif draw < 0.5:
                child = random_prompt_genome(
                    genome_id=child_id,
                    seed=seed + generation * 1000 + index,
                    reflection_rows=reflection_rows,
                    settings=settings,
                    generation=generation,
                    origin="novelty",
                )
            elif draw < 0.6:
                child = random_prompt_genome(
                    genome_id=child_id,
                    seed=seed + generation * 1000 + index,
                    reflection_rows=reflection_rows,
                    settings=settings,
                    generation=generation,
                    origin="random_immigrant",
                )
            else:
                child = mutate_prompt_genome(
                    rng.choice(parents),
                    genome_id=child_id,
                    generation=generation,
                    seed=seed + generation * 1000 + index,
                    reflection_rows=reflection_rows,
                    settings=settings,
                    origin="hybrid_mutation",
                )
        add(child)
    return next_population


def _run_suite(
    *,
    mode: str,
    runner: GSM8KPromptRunner,
    logger: PromptProofLogger,
    settings: GSM8KPromptSettings,
    reflection_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    population_size: int,
    generations: int,
    seed: int,
    base_exact: float,
) -> dict[str, Any]:
    if mode == "gepa_only":
        population = [
            strong_cot_prompt_genome(settings).with_generation(
                0, "gepa_seed", "gen0_gepa_seed0000"
            )
        ]
        while len(population) < population_size:
            population.append(
                random_prompt_genome(
                    genome_id=f"gen0_gepa_seed{len(population):04d}",
                    seed=seed + len(population),
                    reflection_rows=reflection_rows,
                    settings=settings,
                    origin="gepa_seed",
                )
            )
    else:
        population = create_initial_prompt_population(
            population_size=population_size,
            seed=seed,
            reflection_rows=reflection_rows,
            settings=settings,
            origin=mode,
        )

    all_records: list[dict[str, Any]] = []
    for generation in range(generations):
        generation_records = []
        existing: list[PromptGenome] = []
        for genome in population:
            novelty = _novelty(genome, existing)
            existing.append(genome)
            record = logger.evaluate_and_log(
                runner=runner,
                genome=genome,
                rows=dev_rows,
                reflection_rows=reflection_rows,
                suite=mode,
                split="dev",
                novelty=novelty,
            )
            generation_records.append(record)
            all_records.append(record)

        front = pareto_front(generation_records)
        write_json(settings.output_dir / "pareto_front.json", front)
        best = _best_by_exact(all_records)
        if best:
            write_json(settings.output_dir / "best_by_exact.json", best)
        write_json(
            settings.output_dir / "proof_report.partial.json",
            {
                "suite": mode,
                "generation": generation,
                "best_by_exact": best,
                "pareto_front": front,
            },
        )

        if generation < generations - 1:
            population = _make_next_generation(
                mode=mode,
                current=population,
                records=generation_records,
                base_exact=base_exact,
                generation=generation + 1,
                population_size=population_size,
                seed=seed,
                reflection_rows=reflection_rows,
                settings=settings,
            )

    return {
        "suite": mode,
        "records": all_records,
        "best": _best_by_exact(all_records),
        "pareto_front": pareto_front(all_records),
    }


def _evaluate_baseline(
    *,
    name: str,
    genome: PromptGenome,
    runner: GSM8KPromptRunner,
    logger: PromptProofLogger,
    reflection_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return logger.evaluate_and_log(
        runner=runner,
        genome=genome,
        rows=dev_rows,
        reflection_rows=reflection_rows,
        suite=name,
        split="dev",
        novelty=0.0,
    )


def _evaluate_on_test(
    *,
    records: list[dict[str, Any]],
    runner: GSM8KPromptRunner,
    logger: PromptProofLogger,
    reflection_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    test_records = []
    for record in records:
        decode = record["genome"].get("decode", {})
        genome_data = {
            "id": f"test_{record['suite']}_{record['candidate_id']}",
            "system_prompt": record["genome"]["system_prompt"],
            "reasoning_instruction": record["genome"]["reasoning_instruction"],
            "answer_format_instruction": record["genome"][
                "answer_format_instruction"
            ],
            "verification_instruction": record["genome"]["verification_instruction"],
            "few_shot_example_ids": record["genome"]["few_shot_example_ids"],
            "few_shot_order": record["genome"]["few_shot_order"],
            "temperature": decode["temperature"],
            "top_p": decode["top_p"],
            "max_new_tokens": decode["max_new_tokens"],
            "self_consistency_n": decode["self_consistency_n"],
            "origin": record["suite"],
            "generation": record.get("generation", 0),
            "seed": 0,
        }
        genome = prompt_genome_from_dict(genome_data)
        test_records.append(
            logger.evaluate_and_log(
                runner=runner,
                genome=genome,
                rows=test_rows,
                reflection_rows=reflection_rows,
                suite=record["suite"],
                split="test",
                novelty=_metric(record, "novelty"),
            )
        )
    return test_records


def _interpret_prompt_report(
    *,
    base_test: dict[str, Any] | None,
    random_test: dict[str, Any] | None,
    gepa_test: dict[str, Any] | None,
    ca_neat_test: dict[str, Any] | None = None,
    hybrid_test: dict[str, Any] | None,
) -> dict[str, Any]:
    base_exact = _metric(base_test, "exact_accuracy") if base_test else None
    random_exact = _metric(random_test, "exact_accuracy") if random_test else None
    gepa_exact = _metric(gepa_test, "exact_accuracy") if gepa_test else None
    ca_neat_exact = _metric(ca_neat_test, "exact_accuracy") if ca_neat_test else None
    hybrid_exact = _metric(hybrid_test, "exact_accuracy") if hybrid_test else None
    return {
        "exact_accuracy": {
            "base": base_exact,
            "random_best": random_exact,
            "gepa_only": gepa_exact,
            "ca_neat": ca_neat_exact,
            "hybrid": hybrid_exact,
        },
        "ca_neat_beats_base": ca_neat_exact is not None
        and base_exact is not None
        and ca_neat_exact > base_exact,
        "ca_neat_beats_random": ca_neat_exact is not None
        and random_exact is not None
        and ca_neat_exact > random_exact,
        "ca_neat_minimum_credible_win": ca_neat_exact is not None
        and base_exact is not None
        and random_exact is not None
        and ca_neat_exact >= base_exact + 0.03
        and ca_neat_exact >= random_exact + 0.02,
        "hybrid_beats_base": hybrid_exact is not None
        and base_exact is not None
        and hybrid_exact > base_exact,
        "hybrid_beats_random": hybrid_exact is not None
        and random_exact is not None
        and hybrid_exact > random_exact,
        "hybrid_beats_gepa_only": hybrid_exact is not None
        and gepa_exact is not None
        and hybrid_exact > gepa_exact,
        "minimum_credible_win": hybrid_exact is not None
        and base_exact is not None
        and random_exact is not None
        and gepa_exact is not None
        and hybrid_exact >= base_exact + 0.03
        and hybrid_exact >= random_exact + 0.02
        and hybrid_exact >= gepa_exact + 0.01,
    }


def _enabled_suites(config_dict: dict[str, Any]) -> set[str]:
    evaluation = config_dict.get("experiment", {}).get("evaluation", {}) or {}
    configured = evaluation.get("enabled_suites")
    if not configured:
        return {"gepa_only", "ca_neat", "hybrid"}
    return {str(item) for item in configured}


def _prompt_proof_quality_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return universal proof-quality verdict for prompt proof reports."""
    test_records = report["test"]["best_by_suite"]
    evolved = _best_by_exact(
        [
            record
            for key in ("gepa_only", "ca_neat", "hybrid")
            if (record := test_records.get(key)) is not None
        ]
    )
    if evolved is None:
        raise ValueError("FAIL-FAST: prompt proof requires an evolved held-out record")

    verdict = validate_proof_quality(
        {
            "base": _proof_record_from_prompt_exact(
                _required_prompt_record(test_records, "base_prompt")
            ),
            "fixed": _proof_record_from_prompt_exact(
                _required_prompt_record(test_records, "strong_cot_prompt")
            ),
            "random": _proof_record_from_prompt_exact(
                _required_prompt_record(test_records, "random_baseline")
            ),
            "evolved": _proof_record_from_prompt_exact(evolved),
            "held_out": _proof_record_from_prompt_exact(evolved),
            "seeds": report["seeds"],
        }
    )
    return asdict(verdict)


def _required_prompt_record(
    records: dict[str, dict[str, Any]], key: str
) -> dict[str, Any]:
    if key not in records:
        raise ValueError(f"FAIL-FAST: prompt proof missing held-out record '{key}'")
    return records[key]


def _proof_record_from_prompt_exact(record: dict[str, Any]) -> dict[str, float]:
    return {"fitness": _metric(record, "exact_accuracy")}


def _prompt_proof_seed_configs(config: Any) -> tuple[Any, ...]:
    """Return isolated prompt proof configs for each configured seed."""
    proof_seeds = validate_proof_execution_policy(config)
    configs = []
    base_output_dir = config.execution.output_dir
    base_artifacts_dir = config.cache.artifacts_dir
    base_run_id = config.cache.run_id or "prompt_proof"
    for seed in proof_seeds:
        payload = config.model_dump(mode="python")
        payload["seed"] = seed
        payload["execution"]["output_dir"] = base_output_dir / f"seed_{seed}"
        payload["cache"]["artifacts_dir"] = base_artifacts_dir / f"seed_{seed}"
        payload["cache"]["run_id"] = f"{base_run_id}_seed_{seed}"
        configs.append(config.__class__.model_validate(payload))
    return tuple(configs)


def _aggregate_prompt_seed_reports(
    config: Any,
    seed_reports: list[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    """Aggregate prompt proof reports across actual seed runs."""
    if not seed_reports:
        raise ValueError("FAIL-FAST: prompt proof aggregation requires seed reports")
    test_records = [
        report["test"]["best_by_suite"]
        for report in seed_reports
    ]
    base = _mean_prompt_record(
        [records["base_prompt"] for records in test_records], "base_prompt_mean"
    )
    fixed = _mean_prompt_record(
        [records["strong_cot_prompt"] for records in test_records],
        "strong_cot_prompt_mean",
    )
    random_best = _mean_prompt_record(
        [records["random_baseline"] for records in test_records],
        "random_baseline_mean",
    )
    gepa = _mean_optional_prompt_record(test_records, "gepa_only", "gepa_only_mean")
    ca_neat = _mean_optional_prompt_record(test_records, "ca_neat", "ca_neat_mean")
    hybrid = _mean_optional_prompt_record(test_records, "hybrid", "hybrid_mean")
    aggregate_test_records = {
        "base_prompt": base,
        "strong_cot_prompt": fixed,
        "random_baseline": random_best,
    }
    if gepa is not None:
        aggregate_test_records["gepa_only"] = gepa
    if ca_neat is not None:
        aggregate_test_records["ca_neat"] = ca_neat
    if hybrid is not None:
        aggregate_test_records["hybrid"] = hybrid
    report = {
        "target": config.experiment.target,
        "experiment": config.experiment.name,
        "model": _settings(config.model_dump(mode="json")).model_name,
        "dataset": "openai/gsm8k",
        "started_at": started_at,
        "total_seconds": time.time() - started_at,
        "seeds": [report["seeds"][0] for report in seed_reports],
        "seed_runs": seed_reports,
        "split_fingerprints_by_seed": {
            str(report["seeds"][0]): report["split_fingerprints"]
            for report in seed_reports
        },
        "budget": {
            "population_size": config.execution.population_size,
            "generations": config.execution.generations,
            "random_trials_per_seed": seed_reports[0]["budget"]["random_trials"],
            "enabled_suites": seed_reports[0]["budget"]["enabled_suites"],
        },
        "baselines": {
            "base_prompt": base,
            "strong_cot_prompt": fixed,
            "random": {"best": random_best},
        },
        "gepa_only": {"best": gepa},
        "ca_neat_only": {"best": ca_neat},
        "hybrid": {"best": hybrid},
        "test": {
            "records": list(aggregate_test_records.values()),
            "best_by_suite": aggregate_test_records,
        },
        "interpretation": _interpret_prompt_report(
            base_test=base,
            random_test=random_best,
            gepa_test=gepa,
            ca_neat_test=ca_neat,
            hybrid_test=hybrid,
        ),
        "artifacts": {
            "seed_reports": [
                report["artifacts"]["proof_report"] for report in seed_reports
            ]
        },
    }
    report["proof_quality"] = _prompt_proof_quality_summary(report)
    return report


def _mean_optional_prompt_record(
    records_by_seed: list[dict[str, dict[str, Any]]],
    key: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    records = [records[key] for records in records_by_seed if key in records]
    return _mean_prompt_record(records, candidate_id) if records else None


def _mean_prompt_record(records: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    """Return a mean metric record across proof seeds."""
    if not records:
        raise ValueError("FAIL-FAST: cannot average empty prompt records")
    metric_names = (
        "exact_accuracy",
        "token_efficiency",
        "latency_score",
        "formatting_success_rate",
        "novelty",
        "avg_generated_tokens",
        "latency_seconds",
    )
    metrics = {
        name: sum(_metric(record, name) for record in records) / len(records)
        for name in metric_names
    }
    return {
        "candidate_id": candidate_id,
        "suite": candidate_id.removesuffix("_mean"),
        "split": "test",
        "fitness": sum(float(record["fitness"]) for record in records) / len(records),
        "metrics": metrics,
        "seed_records": records,
    }


def _skipped_suite(name: str) -> dict[str, Any]:
    return {"suite": name, "records": [], "best": None, "pareto_front": []}


def run_gsm8k_prompt_evolution_proof(
    config: Any,
    random_trials: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run all configured proof seeds and write an aggregate prompt proof."""
    started_at = time.time()
    seed_reports = [
        _run_gsm8k_prompt_evolution_single_seed_proof(seed_config, random_trials)
        for seed_config in _prompt_proof_seed_configs(config)
    ]
    report = _aggregate_prompt_seed_reports(config, seed_reports, started_at)
    destination = output_path or _settings(config.model_dump(mode="json")).output_dir / "proof_report.json"
    report["artifacts"]["proof_report"] = str(destination)
    write_json(destination, report)
    return report


def _run_gsm8k_prompt_evolution_single_seed_proof(
    config: Any,
    random_trials: int | None = None,
) -> dict[str, Any]:
    """Run base, random, GEPA-only, CA/NEAT-only, and hybrid prompt proof for one seed."""
    if config.experiment.target != "gsm8k_prompt_evolution":
        raise ValueError(
            "Prompt proof requires experiment.target='gsm8k_prompt_evolution'"
        )
    config_dict = config.model_dump(mode="json")
    settings = _settings(config_dict)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    dataset = next(iter(GSM8KPromptDataset(config_dict).problems()))
    reflection_rows = list(dataset["reflection"])
    dev_rows = list(dataset["dev"])
    test_rows = list(dataset["test"])
    assert_no_test_leakage(reflection_rows, dev_rows, test_rows)

    runner = GSM8KPromptRunner(config_dict)
    logger = PromptProofLogger(settings.output_dir, resume=settings.resume)
    budget = random_trials or config.execution.population_size * config.execution.generations
    enabled_suites = _enabled_suites(config_dict)

    base_record = _evaluate_baseline(
        name="base_prompt",
        genome=base_prompt_genome(settings),
        runner=runner,
        logger=logger,
        reflection_rows=reflection_rows,
        dev_rows=dev_rows,
    )
    strong_record = _evaluate_baseline(
        name="strong_cot_prompt",
        genome=strong_cot_prompt_genome(settings),
        runner=runner,
        logger=logger,
        reflection_rows=reflection_rows,
        dev_rows=dev_rows,
    )
    base_exact = _metric(base_record, "exact_accuracy")

    random_records = []
    for index in range(budget):
        genome = random_prompt_genome(
            genome_id=f"random_{index:04d}",
            seed=settings.seed + 200_000 + index,
            reflection_rows=reflection_rows,
            settings=settings,
            origin="random_baseline",
        )
        random_records.append(
            logger.evaluate_and_log(
                runner=runner,
                genome=genome,
                rows=dev_rows,
                reflection_rows=reflection_rows,
                suite="random_baseline",
                split="dev",
                novelty=0.0,
            )
        )

    gepa = (
        _run_suite(
            mode="gepa_only",
            runner=runner,
            logger=logger,
            settings=settings,
            reflection_rows=reflection_rows,
            dev_rows=dev_rows,
            population_size=config.execution.population_size,
            generations=config.execution.generations,
            seed=settings.seed + 10_000,
            base_exact=base_exact,
        )
        if "gepa_only" in enabled_suites
        else _skipped_suite("gepa_only")
    )
    ca_neat = (
        _run_suite(
            mode="ca_neat",
            runner=runner,
            logger=logger,
            settings=settings,
            reflection_rows=reflection_rows,
            dev_rows=dev_rows,
            population_size=config.execution.population_size,
            generations=config.execution.generations,
            seed=settings.seed + 20_000,
            base_exact=base_exact,
        )
        if "ca_neat" in enabled_suites
        else _skipped_suite("ca_neat")
    )
    hybrid = (
        _run_suite(
            mode="hybrid",
            runner=runner,
            logger=logger,
            settings=settings,
            reflection_rows=reflection_rows,
            dev_rows=dev_rows,
            population_size=config.execution.population_size,
            generations=config.execution.generations,
            seed=settings.seed + 30_000,
            base_exact=base_exact,
        )
        if "hybrid" in enabled_suites
        else _skipped_suite("hybrid")
    )

    selected_for_test = [
        base_record,
        strong_record,
        _best_by_exact(random_records),
        gepa["best"],
        ca_neat["best"],
        hybrid["best"],
    ]
    selected_for_test = [record for record in selected_for_test if record is not None]
    test_records = _evaluate_on_test(
        records=selected_for_test,
        runner=runner,
        logger=logger,
        reflection_rows=reflection_rows,
        test_rows=test_rows,
    )
    test_by_suite = {record["suite"]: record for record in test_records}

    report = {
        "target": config.experiment.target,
        "experiment": config.experiment.name,
        "model": settings.model_name,
        "dataset": "openai/gsm8k",
        "started_at": started_at,
        "total_seconds": time.time() - started_at,
        "split_fingerprints": dataset["fingerprints"],
        "budget": {
            "population_size": config.execution.population_size,
            "generations": config.execution.generations,
            "random_trials": len(random_records),
            "enabled_suites": sorted(enabled_suites),
        },
        "seeds": [config.seed],
        "baselines": {
            "base_prompt": base_record,
            "strong_cot_prompt": strong_record,
            "random": {
                "best": _best_by_exact(random_records),
                "trials": random_records,
            },
        },
        "gepa_only": gepa,
        "ca_neat_only": ca_neat,
        "hybrid": hybrid,
        "test": {
            "records": test_records,
            "best_by_suite": test_by_suite,
        },
        "interpretation": _interpret_prompt_report(
            base_test=test_by_suite.get("base_prompt"),
            random_test=test_by_suite.get("random_baseline"),
            gepa_test=test_by_suite.get("gepa_only"),
            ca_neat_test=test_by_suite.get("ca_neat"),
            hybrid_test=test_by_suite.get("hybrid"),
        ),
        "artifacts": {
            "candidate_jsonl": str(logger.candidate_jsonl),
            "failure_traces_jsonl": str(logger.failure_jsonl),
            "pareto_front": str(settings.output_dir / "pareto_front.json"),
            "best_by_exact": str(settings.output_dir / "best_by_exact.json"),
        },
    }
    destination = settings.output_dir / "proof_report.json"
    report["artifacts"]["proof_report"] = str(destination)
    write_json(destination, report)
    return report


class GSM8KPromptEvolutionFitness(FitnessFn):
    """Placeholder protocol implementation for registry compatibility."""

    def __call__(
        self,
        genome: Any,
        model: ModelRunner,
        problems: Iterable[dict[str, Any]],
    ) -> float:
        raise RuntimeError("Use `core.cli.main prove` for gsm8k_prompt_evolution.")

    def evaluate_multi_objective(
        self,
        genome: Any,
        model: ModelRunner,
        problems: Iterable[dict[str, Any]],
        ca_features: Any | None = None,
    ) -> Any:
        raise RuntimeError("Use `core.cli.main prove` for gsm8k_prompt_evolution.")


class GSM8KPromptEvolutionPlugin:
    """Prompt-evolution plugin target."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        settings = _settings(config)
        print(
            "GSM8K prompt-evolution plugin initialized "
            f"(model={settings.model_name}, "
            f"reflection={settings.reflection_samples}, "
            f"dev={settings.dev_samples}, test={settings.test_samples})"
        )

    def dataset(self) -> DatasetProvider:
        return GSM8KPromptDataset(self.config)

    def model_factory(self) -> Callable[[Any, Any | None], ModelRunner]:
        def create_model(_lora_cfg: Any = None, _genome: Any | None = None) -> ModelRunner:
            return GSM8KPromptRunner(self.config)

        return create_model

    def fitness_fn(self) -> FitnessFn:
        return GSM8KPromptEvolutionFitness()
