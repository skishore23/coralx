"""Experimental GSM8K LoRA benchmark plugin.

This target downloads a small language model and GSM8K, trains real PEFT LoRA
adapters, and reports exact-answer accuracy plus held-out answer-token loss.
"""

from __future__ import annotations

import gc
import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from core.domain.cheap_knobs import CheapKnobs
from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.mapping import LoRAConfig
from core.domain.stable_hash import stable_digest
from core.ports.interfaces import DatasetProvider, FitnessFn, ModelRunner

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class GSM8KLoRASettings:
    """Runtime settings for the GSM8K LoRA micro benchmark."""

    model_name: str
    max_seq_length: int
    train_samples: int
    eval_samples: int
    held_out_samples: int
    max_train_steps: int
    max_new_tokens: int
    loss_weight: float
    answer_format: str
    system_prompt: str | None
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    output_dir: Path
    device: str


@dataclass(frozen=True)
class GSM8KLoRAMetrics:
    """Evaluation metrics returned by a single LoRA training run."""

    exact_accuracy: float
    formatted_answer_rate: float
    train_loss: float
    eval_loss: float
    loss_score: float
    train_seconds: float
    eval_seconds: float
    predictions: tuple[dict[str, Any], ...]
    cache_key: str
    from_cache: bool = False

    def to_report_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metrics for candidate reports."""
        return {
            "exact_accuracy": self.exact_accuracy,
            "formatted_answer_rate": self.formatted_answer_rate,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "loss_score": self.loss_score,
            "train_seconds": self.train_seconds,
            "eval_seconds": self.eval_seconds,
            "cache_key": self.cache_key,
            "from_cache": self.from_cache,
            "predictions": list(self.predictions),
        }


def score_gsm8k_metrics(metrics: GSM8KLoRAMetrics, loss_weight: float) -> float:
    """Combine exact accuracy and loss score for laptop-scale GSM8K fitness."""
    clamped_weight = max(0.0, min(1.0, loss_weight))
    return (
        metrics.exact_accuracy * (1.0 - clamped_weight)
        + metrics.loss_score * clamped_weight
    )


def _optional_ml_imports() -> dict[str, Any]:
    """Import optional ML dependencies with a clear error message."""
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - exercised only with missing extras
        raise RuntimeError(
            "The gsm8k_lora target requires optional ML dependencies. "
            "Install with: python -m pip install -e '.[ml]'"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _settings(config: dict[str, Any]) -> GSM8KLoRASettings:
    experiment = config.get("experiment", {})
    evaluation = experiment.get("evaluation", {}) or {}
    training = config.get("training", {}) or {}
    execution = config.get("execution", {}) or {}
    cache = config.get("cache", {}) or {}
    model = experiment.get("model", {}) or {}

    output_dir = Path(
        str(execution.get("output_dir") or cache.get("artifacts_dir") or "./artifacts")
    )

    return GSM8KLoRASettings(
        model_name=str(model.get("name", "Qwen/Qwen2.5-0.5B-Instruct")),
        max_seq_length=int(model.get("max_seq_length", 384)),
        train_samples=int(evaluation.get("train_samples", 32)),
        eval_samples=int(evaluation.get("eval_samples", 16)),
        held_out_samples=int(evaluation.get("held_out_samples", 16)),
        max_train_steps=int(evaluation.get("max_train_steps", 20)),
        max_new_tokens=int(evaluation.get("max_new_tokens", 64)),
        loss_weight=float(evaluation.get("loss_weight", 1.0)),
        answer_format=str(evaluation.get("answer_format", "gsm8k")),
        system_prompt=evaluation.get("system_prompt"),
        seed=int(config.get("seed", 42)),
        batch_size=int(training.get("batch_size", 1)),
        epochs=int(training.get("epochs", 1)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        output_dir=output_dir,
        device=str(evaluation.get("device", "auto")),
    )


def _format_prompt(
    question: str,
    tokenizer: Any | None = None,
    settings: GSM8KLoRASettings | None = None,
) -> str:
    if settings and settings.system_prompt:
        instruction = settings.system_prompt
    elif settings and settings.answer_format == "boxed":
        instruction = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
    else:
        instruction = (
            "Solve the grade-school math problem. "
            "End with the final answer in the form #### number."
        )
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{instruction}\n\nQuestion: {question}\nAnswer:"


def _format_answer(
    answer: str,
    final_answer: str | None = None,
    settings: GSM8KLoRASettings | None = None,
) -> str:
    if settings and settings.answer_format == "boxed":
        reasoning = answer.split("####", maxsplit=1)[0].strip()
        final = final_answer or _extract_final_answer(answer) or ""
        return f" {reasoning}\n\\boxed{{{final}}}"
    return f" {answer.strip()}"


def _extract_final_answer(text: str) -> str | None:
    """Extract a comparable final numeric answer from GSM8K text."""
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        text = boxed_matches[-1]

    if "####" in text:
        text = text.split("####")[-1]

    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None

    return matches[-1].replace(",", "")


def _normalize_number(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def _answers_match(prediction: str | None, reference: str | None) -> bool:
    predicted = _normalize_number(prediction)
    expected = _normalize_number(reference)
    return predicted is not None and expected is not None and predicted == expected


class GSM8KLoRADataset(DatasetProvider):
    """GSM8K provider with deterministic small train/eval slices."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.settings = _settings(config)

    def problems(self) -> Iterable[dict[str, Any]]:
        deps = _optional_ml_imports()
        load_dataset = deps["load_dataset"]

        dataset = load_dataset("openai/gsm8k", "main")
        train_split = dataset["train"].shuffle(seed=self.settings.seed)
        eval_split = dataset["test"].shuffle(seed=self.settings.seed + 1)

        train_count = min(self.settings.train_samples, len(train_split))
        eval_count = min(self.settings.eval_samples, len(eval_split))
        held_out_count = min(
            self.settings.held_out_samples, max(0, len(eval_split) - eval_count)
        )

        train_rows = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "final_answer": _extract_final_answer(row["answer"]),
            }
            for row in train_split.select(range(train_count))
        ]
        eval_rows = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "final_answer": _extract_final_answer(row["answer"]),
            }
            for row in eval_split.select(range(eval_count))
        ]
        held_out_rows = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "final_answer": _extract_final_answer(row["answer"]),
            }
            for row in eval_split.select(range(eval_count, eval_count + held_out_count))
        ]

        yield {
            "name": "gsm8k_lora",
            "dataset": "openai/gsm8k",
            "train": train_rows,
            "eval": eval_rows,
            "held_out": held_out_rows,
        }


class GSM8KLoRARunner(ModelRunner):
    """Train and evaluate one real LoRA adapter for GSM8K."""

    def __init__(
        self,
        lora_cfg: LoRAConfig,
        config: dict[str, Any],
        genome: Genome | None = None,
        result_cache: dict[str, GSM8KLoRAMetrics] | None = None,
    ):
        self.lora_cfg = lora_cfg
        self.config = config
        self.genome = genome
        self.settings = _settings(config)
        self.result_cache = result_cache if result_cache is not None else {}
        self.last_metrics: GSM8KLoRAMetrics | None = None

    def generate(
        self, prompt: str, max_tokens: int = 512, cheap_knobs: CheapKnobs | None = None
    ) -> str:
        raise RuntimeError(
            "gsm8k_lora evaluates through train_and_evaluate(), not ad hoc generate()."
        )

    def train_and_evaluate(self, problem: dict[str, Any]) -> GSM8KLoRAMetrics:
        cache_key = self._cache_key(problem)
        if cache_key in self.result_cache:
            cached = self.result_cache[cache_key]
            self.last_metrics = GSM8KLoRAMetrics(
                exact_accuracy=cached.exact_accuracy,
                formatted_answer_rate=cached.formatted_answer_rate,
                train_loss=cached.train_loss,
                eval_loss=cached.eval_loss,
                loss_score=cached.loss_score,
                train_seconds=cached.train_seconds,
                eval_seconds=cached.eval_seconds,
                predictions=cached.predictions,
                cache_key=cached.cache_key,
                from_cache=True,
            )
            return self.last_metrics

        disk_cached = self._read_disk_cache(cache_key)
        if disk_cached is not None:
            self.result_cache[cache_key] = disk_cached
            self.last_metrics = disk_cached
            return disk_cached

        metrics = self._train_and_evaluate_uncached(problem, cache_key)
        self.result_cache[cache_key] = metrics
        self._write_disk_cache(metrics)
        self.last_metrics = metrics
        return metrics

    def _cache_key(self, problem: dict[str, Any]) -> str:
        train_rows = problem["train"]
        eval_rows = problem["eval"]
        train_fingerprint = stable_digest(
            [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "final_answer": row.get("final_answer"),
                }
                for row in train_rows
            ],
            length=16,
        )
        eval_fingerprint = stable_digest(
            [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "final_answer": row.get("final_answer"),
                }
                for row in eval_rows
            ],
            length=16,
        )
        payload = {
            "target": "gsm8k_lora",
            "model": self.settings.model_name,
            "max_seq_length": self.settings.max_seq_length,
            "train_samples": len(train_rows),
            "eval_samples": len(eval_rows),
            "train_fingerprint": train_fingerprint,
            "eval_fingerprint": eval_fingerprint,
            "max_train_steps": self.settings.max_train_steps,
            "max_new_tokens": self.settings.max_new_tokens,
            "loss_weight": self.settings.loss_weight,
            "answer_format": self.settings.answer_format,
            "system_prompt": self.settings.system_prompt,
            "batch_size": self.settings.batch_size,
            "epochs": self.settings.epochs,
            "learning_rate": self.settings.learning_rate,
            "gradient_accumulation_steps": self.settings.gradient_accumulation_steps,
            "max_grad_norm": self.settings.max_grad_norm,
            "seed": self.settings.seed,
            "fitness_version": "eval_loss_v1",
            "prompt_version": "gsm8k_chat_or_plain_v2",
            "lora": {
                "r": self.lora_cfg.r,
                "alpha": self.lora_cfg.alpha,
                "dropout": self.lora_cfg.dropout,
                "target_modules": list(self.lora_cfg.target_modules),
                "adapter_type": self.lora_cfg.adapter_type,
            },
            "run_id": self.genome.run_id if self.genome else None,
        }
        return stable_digest(payload, length=16)

    def _cache_path(self, cache_key: str) -> Path:
        return self.settings.output_dir / "gsm8k_lora_cache" / f"{cache_key}.json"

    def _read_disk_cache(self, cache_key: str) -> GSM8KLoRAMetrics | None:
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None
        data = json.loads(cache_path.read_text())
        return GSM8KLoRAMetrics(
            exact_accuracy=float(data["exact_accuracy"]),
            formatted_answer_rate=float(data["formatted_answer_rate"]),
            train_loss=float(data["train_loss"]),
            eval_loss=float(data["eval_loss"]),
            loss_score=float(data["loss_score"]),
            train_seconds=float(data["train_seconds"]),
            eval_seconds=float(data["eval_seconds"]),
            predictions=tuple(data.get("predictions", [])),
            cache_key=cache_key,
            from_cache=True,
        )

    def _write_disk_cache(self, metrics: GSM8KLoRAMetrics) -> None:
        cache_path = self._cache_path(metrics.cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "exact_accuracy": metrics.exact_accuracy,
                    "formatted_answer_rate": metrics.formatted_answer_rate,
                    "train_loss": metrics.train_loss,
                    "eval_loss": metrics.eval_loss,
                    "loss_score": metrics.loss_score,
                    "train_seconds": metrics.train_seconds,
                    "eval_seconds": metrics.eval_seconds,
                    "predictions": list(metrics.predictions),
                    "cache_key": metrics.cache_key,
                },
                indent=2,
                sort_keys=True,
            )
        )

    def _train_and_evaluate_uncached(
        self, problem: dict[str, Any], cache_key: str
    ) -> GSM8KLoRAMetrics:
        deps = _optional_ml_imports()
        torch = deps["torch"]
        AutoModelForCausalLM = deps["AutoModelForCausalLM"]
        AutoTokenizer = deps["AutoTokenizer"]
        LoraConfig = deps["LoraConfig"]
        TaskType = deps["TaskType"]
        get_peft_model = deps["get_peft_model"]

        if self.lora_cfg.adapter_type not in {"lora", "none"}:
            raise RuntimeError(
                "gsm8k_lora supports adapter_type='lora' for trained adapters "
                "and adapter_type='none' for base-model baselines. "
                f"Received adapter_type={self.lora_cfg.adapter_type!r}."
            )

        self._set_torch_seed(torch)
        device = self._resolve_device(torch)

        tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.settings.model_name)
        model.config.pad_token_id = tokenizer.pad_token_id

        if self.lora_cfg.adapter_type == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(self.lora_cfg.r),
                lora_alpha=float(self.lora_cfg.alpha),
                lora_dropout=float(self.lora_cfg.dropout),
                target_modules=list(self.lora_cfg.target_modules),
                bias="none",
            )
            model = get_peft_model(model, peft_config)
        model.to(device)

        train_started = time.time()
        if self.lora_cfg.adapter_type == "lora":
            train_loss = self._train_model(
                torch, model, tokenizer, problem["train"], device
            )
        else:
            train_loss = 0.0
        train_seconds = time.time() - train_started

        eval_started = time.time()
        eval_loss = self._evaluate_loss(
            torch, model, tokenizer, problem["eval"], device
        )
        exact_accuracy, formatted_rate, predictions = self._evaluate_model(
            torch, model, tokenizer, problem["eval"], device
        )
        eval_seconds = time.time() - eval_started
        loss_score = math.exp(-eval_loss)

        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        return GSM8KLoRAMetrics(
            exact_accuracy=exact_accuracy,
            formatted_answer_rate=formatted_rate,
            train_loss=train_loss,
            eval_loss=eval_loss,
            loss_score=loss_score,
            train_seconds=train_seconds,
            eval_seconds=eval_seconds,
            predictions=tuple(predictions),
            cache_key=cache_key,
        )

    def _set_torch_seed(self, torch: Any) -> None:
        torch.manual_seed(self.settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.settings.seed)

    def _resolve_device(self, torch: Any) -> Any:
        requested = self.settings.device
        if requested != "auto":
            return torch.device(requested)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _encode_supervised(self, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
        prompt = _format_prompt(row["question"], tokenizer, self.settings)
        completion = _format_answer(
            row["answer"], row.get("final_answer"), self.settings
        )
        eos = tokenizer.eos_token or ""

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full = tokenizer(
            prompt + completion + eos,
            add_special_tokens=False,
            truncation=True,
            max_length=self.settings.max_seq_length,
        )

        input_ids = list(full.input_ids)
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def _collate(self, torch: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> dict:
        pad_id = tokenizer.pad_token_id
        max_len = max(len(row["input_ids"]) for row in rows)

        input_ids = []
        attention_mask = []
        labels = []
        for row in rows:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(row["attention_mask"] + [0] * pad_len)
            labels.append(row["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _train_model(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        train_rows: list[dict[str, Any]],
        device: Any,
    ) -> float:
        from torch.utils.data import DataLoader

        encoded = [self._encode_supervised(tokenizer, row) for row in train_rows]
        loader = DataLoader(
            encoded,
            batch_size=self.settings.batch_size,
            shuffle=True,
            collate_fn=lambda rows: self._collate(torch, tokenizer, list(rows)),
        )
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=self.settings.learning_rate,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        loss_total = 0.0
        loss_count = 0

        for _epoch in range(self.settings.epochs):
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                loss = output.loss / self.settings.gradient_accumulation_steps
                loss.backward()

                loss_total += float(output.loss.detach().cpu())
                loss_count += 1

                should_step = (
                    loss_count % self.settings.gradient_accumulation_steps == 0
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.settings.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if global_step >= self.settings.max_train_steps:
                    return loss_total / max(loss_count, 1)

        return loss_total / max(loss_count, 1)

    def _evaluate_loss(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        eval_rows: list[dict[str, Any]],
        device: Any,
    ) -> float:
        from torch.utils.data import DataLoader

        encoded = [self._encode_supervised(tokenizer, row) for row in eval_rows]
        loader = DataLoader(
            encoded,
            batch_size=self.settings.batch_size,
            shuffle=False,
            collate_fn=lambda rows: self._collate(torch, tokenizer, list(rows)),
        )

        model.eval()
        loss_total = 0.0
        loss_count = 0
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                loss_total += float(output.loss.detach().cpu())
                loss_count += 1

        return loss_total / max(loss_count, 1)

    def _evaluate_model(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        eval_rows: list[dict[str, Any]],
        device: Any,
    ) -> tuple[float, float, list[dict[str, Any]]]:
        model.eval()
        correct = 0
        formatted = 0
        predictions = []

        with torch.no_grad():
            for index, row in enumerate(eval_rows):
                prompt = _format_prompt(row["question"], tokenizer, self.settings)
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=self.settings.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_tokens = output_ids[0][encoded["input_ids"].shape[-1] :]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
                prediction = _extract_final_answer(generated)
                reference = row["final_answer"]
                is_correct = _answers_match(prediction, reference)

                correct += int(is_correct)
                formatted += int(prediction is not None)
                predictions.append(
                    {
                        "index": index,
                        "prediction": prediction,
                        "reference": reference,
                        "correct": is_correct,
                        "generated": generated[:300],
                    }
                )

        total = max(len(eval_rows), 1)
        return correct / total, formatted / total, predictions


class GSM8KLoRAFitness(FitnessFn):
    """GSM8K fitness for real LoRA adapters."""

    def __call__(
        self,
        genome: Genome,
        model: ModelRunner,
        problems: Iterable[dict[str, Any]],
    ) -> float:
        return self.evaluate_multi_objective(genome, model, problems).overall_fitness()

    def evaluate_multi_objective(
        self,
        genome: Genome,
        model: ModelRunner,
        problems: Iterable[dict[str, Any]],
        ca_features: Any | None = None,
    ) -> MultiObjectiveScores:
        if not isinstance(model, GSM8KLoRARunner):
            raise TypeError("gsm8k_lora requires GSM8KLoRARunner")

        problem = next(iter(problems))
        metrics = model.train_and_evaluate(problem)
        score = score_gsm8k_metrics(metrics, model.settings.loss_weight)

        print(
            "GSM8K LoRA evaluation: "
            f"genome={genome.id}, "
            f"r={genome.lora_cfg.r}, "
            f"alpha={genome.lora_cfg.alpha}, "
            f"dropout={genome.lora_cfg.dropout}, "
            f"accuracy={metrics.exact_accuracy:.4f}, "
            f"formatted={metrics.formatted_answer_rate:.4f}, "
            f"train_loss={metrics.train_loss:.4f}, "
            f"eval_loss={metrics.eval_loss:.4f}, "
            f"loss_score={metrics.loss_score:.4f}, "
            f"fitness={score:.4f}, "
            f"cache={metrics.from_cache}"
        )

        return MultiObjectiveScores(
            task_score=metrics.exact_accuracy,
            quality_score=metrics.formatted_answer_rate,
            risk_score=1.0,
            efficiency_score=metrics.loss_score,
            validity_score=metrics.formatted_answer_rate,
        )


class GSM8KLoRAPlugin:
    """Experimental real-LoRA plugin for the GSM8K public benchmark."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._result_cache: dict[str, GSM8KLoRAMetrics] = {}
        settings = _settings(config)
        print(
            "GSM8K LoRA plugin initialized "
            f"(model={settings.model_name}, "
            f"train_samples={settings.train_samples}, "
            f"eval_samples={settings.eval_samples})"
        )

    def dataset(self) -> DatasetProvider:
        return GSM8KLoRADataset(self.config)

    def model_factory(self) -> Callable[[LoRAConfig, Genome | None], ModelRunner]:
        def create_model(
            lora_cfg: LoRAConfig, genome: Genome | None = None
        ) -> ModelRunner:
            return GSM8KLoRARunner(
                lora_cfg,
                self.config,
                genome=genome,
                result_cache=self._result_cache,
            )

        return create_model

    def fitness_fn(self) -> FitnessFn:
        return GSM8KLoRAFitness()
