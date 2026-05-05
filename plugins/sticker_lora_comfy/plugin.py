"""Formal plugin boundary for the Comfy sticker evolution target."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.mapping import AdapterConfig
from core.domain.stable_hash import stable_int
from core.ports.interfaces import DatasetProvider, FitnessFn, ModelRunner
from scripts import sticker_lora_evolve as sticker_evolve


@dataclass(frozen=True)
class StickerLoRAComfySettings:
    """Runtime-independent sticker benchmark settings."""

    benchmark: str
    split: str
    subjects: int
    api_url: str
    output_dir: Any
    prompt_timeout: float


def _settings(config: dict[str, Any]) -> StickerLoRAComfySettings:
    execution = config.get("execution", {}) or {}
    evaluation = config.get("experiment", {}).get("evaluation", {}) or {}
    dataset = config.get("experiment", {}).get("dataset", {}) or {}
    datasets = tuple(dataset.get("datasets") or ("coco",))
    benchmark = str(datasets[0])
    if benchmark not in sticker_evolve.BENCHMARK_SPLITS:
        raise ValueError(
            f"FAIL-FAST: unknown sticker benchmark '{benchmark}'. "
            f"Supported: {', '.join(sorted(sticker_evolve.BENCHMARK_SPLITS))}"
        )
    split = str(evaluation.get("split", "dev"))
    if split not in sticker_evolve.BENCHMARK_SPLITS[benchmark]:
        raise ValueError(
            f"FAIL-FAST: unknown sticker split '{split}' for benchmark '{benchmark}'"
        )
    return StickerLoRAComfySettings(
        benchmark=benchmark,
        split=split,
        subjects=int(evaluation.get("subjects", 4)),
        api_url=str(evaluation.get("api_url", sticker_evolve.API_URL)),
        output_dir=sticker_evolve.Path(
            str(execution.get("output_dir", sticker_evolve.ARTIFACT_ROOT / "evolution"))
        ),
        prompt_timeout=float(evaluation.get("prompt_timeout", 600.0)),
    )


class StickerLoRAComfyDataset(DatasetProvider):
    """Expose fixed sticker benchmark splits through the plugin protocol."""

    def __init__(self, config: dict[str, Any]):
        self.settings = _settings(config)

    def problems(self) -> Iterable[dict[str, Any]]:
        splits = sticker_evolve.BENCHMARK_SPLITS[self.settings.benchmark]
        yield {
            "name": "sticker_lora_comfy",
            "benchmark": self.settings.benchmark,
            "train_subjects": tuple(splits["train"]),
            "dev_subjects": tuple(splits["dev"]),
            "test_subjects": tuple(splits["test"]),
        }


class StickerLoRAComfyRunner(ModelRunner):
    """Runner marker for the external Comfy sticker workflow."""

    def __init__(self, adapter_config: AdapterConfig, genome: Genome | None = None):
        self.adapter_config = adapter_config
        self.genome = genome

    def generate(self, prompt: str, max_tokens: int, cheap_knobs=None) -> str:
        raise RuntimeError(
            "FAIL-FAST: sticker_lora_comfy uses Comfy image generation, not text "
            "generation. Run scripts/sticker_lora_evolve.py until the image "
            "candidate genome is wired into core evolution."
        )


class StickerLoRAComfyFitness(FitnessFn):
    """Script-backed sticker candidate fitness."""

    def __init__(self, config: dict[str, Any]):
        self.settings = _settings(config)

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
        problem = _single_problem(problems)
        subjects = tuple(problem[f"{self.settings.split}_subjects"])[
            : self.settings.subjects
        ]
        if not subjects:
            raise ValueError("FAIL-FAST: sticker evaluation requires at least one subject")
        result = sticker_evolve.evaluate_candidate(
            _candidate_from_genome(genome),
            subjects,
            self.settings.api_url,
            self.settings.output_dir,
            self.settings.prompt_timeout,
        )
        return _scores_from_result(result)


class StickerLoRAComfyPlugin:
    """Plugin adapter for the sticker LoRA Comfy benchmark."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def dataset(self) -> DatasetProvider:
        return StickerLoRAComfyDataset(self.config)

    def model_factory(self) -> Callable[[AdapterConfig, Genome | None], ModelRunner]:
        def create_runner(
            adapter_config: AdapterConfig, genome: Genome | None = None
        ) -> ModelRunner:
            return StickerLoRAComfyRunner(adapter_config, genome)

        return create_runner

    def fitness_fn(self) -> FitnessFn:
        return StickerLoRAComfyFitness(self.config)


def _single_problem(problems: Iterable[dict[str, Any]]) -> dict[str, Any]:
    problem_list = list(problems)
    if len(problem_list) != 1:
        raise ValueError("FAIL-FAST: sticker evaluation expects exactly one problem")
    return problem_list[0]


def _candidate_from_genome(genome: Genome) -> sticker_evolve.StickerCandidate:
    """Map a core genome into a deterministic sticker inference candidate."""
    adapter = genome.lora_cfg
    lora_name = (
        "cxsticker_v2.safetensors" if adapter.r >= 8 else "cxsticker_v1.safetensors"
    )
    strength_model = sticker_evolve.clamp(0.25 + float(adapter.alpha) / 64.0, 0.15, 0.85)
    strength_clip = sticker_evolve.clamp(0.20 + float(adapter.dropout), 0.10, 0.70)
    cfg = round(sticker_evolve.clamp(6.0 + adapter.r / 8.0, 4.5, 10.5), 1)
    steps = (20, 24, 28, 32, 36)[adapter.r % 5]
    seed_offset = stable_int(
        {"genome_id": genome.id, "run_id": genome.run_id or ""}, 900_000
    ) + 10_000
    return sticker_evolve.StickerCandidate(
        candidate_id=genome.id,
        lora_name=lora_name,
        strength_model=round(strength_model, 2),
        strength_clip=round(strength_clip, 2),
        cfg=cfg,
        steps=steps,
        prompt_template_id=adapter.r % len(sticker_evolve.PROMPT_TEMPLATES),
        negative_prompt_id=adapter.r % len(sticker_evolve.NEGATIVE_PROMPTS),
        sampler_name="euler",
        scheduler="karras" if adapter.dropout > 0.05 else "normal",
        seed_offset=seed_offset,
    )


def _scores_from_result(result: dict[str, Any]) -> MultiObjectiveScores:
    subject_metrics = result.get("subjects") or []
    if not subject_metrics:
        raise ValueError("FAIL-FAST: sticker evaluation returned no subject metrics")
    task_score = float(result["score"])
    background = _mean(subject_metrics, "background")
    outline = _mean(subject_metrics, "outline")
    edge = _mean(subject_metrics, "edge")
    center = _mean(subject_metrics, "center")
    area = _mean(subject_metrics, "area")
    clutter = _mean(subject_metrics, "clutter_penalty")
    return MultiObjectiveScores(
        task_score=task_score,
        quality_score=(outline + edge + area) / 3.0,
        risk_score=background,
        efficiency_score=1.0 - clutter,
        validity_score=center,
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)
