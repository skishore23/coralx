"""Formal plugin boundary for the Comfy sticker evolution target."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    output_dir: Path
    prompt_timeout: float
    workflow_path: Path
    comfy_output_dir: Path
    require_api: bool


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
        output_dir=Path(
            str(execution.get("output_dir", sticker_evolve.ARTIFACT_ROOT / "evolution"))
        ),
        prompt_timeout=float(evaluation.get("prompt_timeout", 600.0)),
        workflow_path=Path(
            str(evaluation.get("workflow_path", sticker_evolve.WORKFLOW_API))
        ),
        comfy_output_dir=Path(
            str(evaluation.get("comfy_output_dir", sticker_evolve.COMFY_OUTPUT))
        ),
        require_api=bool(evaluation.get("require_api", True)),
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


@dataclass(frozen=True)
class StickerLoRAComfyMetrics:
    """Serializable result from one Comfy sticker candidate evaluation."""

    candidate: dict[str, Any]
    candidate_key: str
    score: float
    subjects: tuple[dict[str, Any], ...]
    sheet: str
    workflow_path: str
    api_url: str

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "candidate_key": self.candidate_key,
            "score": self.score,
            "subjects": list(self.subjects),
            "sheet": self.sheet,
            "workflow_path": self.workflow_path,
            "api_url": self.api_url,
        }


class StickerLoRAComfyRunner(ModelRunner):
    """ComfyUI image runner for already-trained sticker LoRAs."""

    def __init__(
        self,
        adapter_config: AdapterConfig,
        settings: StickerLoRAComfySettings,
        genome: Genome | None = None,
    ):
        self.adapter_config = adapter_config
        self.settings = settings
        self.genome = genome
        self.last_metrics: StickerLoRAComfyMetrics | None = None

    def generate(self, prompt: str, max_tokens: int, cheap_knobs=None) -> str:
        raise RuntimeError(
            "FAIL-FAST: sticker_lora_comfy uses Comfy image generation, not text "
            "generation. Use evaluate_candidate() through the sticker fitness "
            "function."
        )

    def evaluate_candidate(
        self,
        candidate: sticker_evolve.StickerCandidate,
        subjects: tuple[str, ...],
    ) -> dict[str, Any]:
        """Render and score a candidate through the configured Comfy workflow."""
        self.validate_runtime()
        base_workflow = json.loads(
            self.settings.workflow_path.read_text(encoding="utf-8")
        )
        subject_scores = []
        image_paths = {}
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        for index, subject in enumerate(subjects):
            prefix = f"evo_{candidate.candidate_id}_{subject.replace(' ', '_')}"
            workflow = sticker_evolve.render_workflow(
                base_workflow, candidate, subject, index, prefix
            )
            prompt_id = sticker_evolve.post_prompt(workflow, self.settings.api_url)
            image_path = _wait_for_output(
                prompt_id,
                self.settings.api_url,
                self.settings.prompt_timeout,
                self.settings.comfy_output_dir,
            )
            metrics = sticker_evolve.image_proxy_metrics(image_path)
            subject_scores.append(
                {
                    "subject": subject,
                    "prompt_id": prompt_id,
                    "image": str(image_path),
                    **metrics,
                }
            )
            image_paths[subject] = image_path

        avg_score = sum(item["score"] for item in subject_scores) / len(subject_scores)
        sheet = sticker_evolve.write_sheet(
            self.settings.output_dir / f"{candidate.candidate_id}_sheet.png",
            image_paths,
        )
        result = {
            "candidate": asdict(candidate),
            "candidate_key": candidate.key(),
            "score": avg_score,
            "subjects": subject_scores,
            "sheet": str(sheet),
        }
        self.last_metrics = StickerLoRAComfyMetrics(
            candidate=result["candidate"],
            candidate_key=result["candidate_key"],
            score=float(result["score"]),
            subjects=tuple(subject_scores),
            sheet=str(sheet),
            workflow_path=str(self.settings.workflow_path),
            api_url=self.settings.api_url,
        )
        return result

    def validate_runtime(self) -> None:
        """Fail clearly when the local Comfy runtime is not ready."""
        if not self.settings.workflow_path.exists():
            raise RuntimeError(
                "FAIL-FAST: sticker_lora_comfy workflow is missing: "
                f"{self.settings.workflow_path}. Run "
                "`python scripts/sticker_lora_comfy.py setup --overwrite` first."
            )
        workflow = json.loads(self.settings.workflow_path.read_text(encoding="utf-8"))
        required_nodes = {
            "2": "CLIPTextEncode",
            "3": "CLIPTextEncode",
            "5": "KSampler",
            "7": "SaveImage",
            "8": "LoraLoader",
        }
        missing_nodes = [
            node_id for node_id in required_nodes if node_id not in workflow
        ]
        if missing_nodes:
            raise RuntimeError(
                "FAIL-FAST: sticker_lora_comfy workflow is missing required nodes: "
                + ", ".join(missing_nodes)
            )
        wrong_nodes = [
            f"{node_id}:{workflow[node_id].get('class_type')}"
            for node_id, expected_type in required_nodes.items()
            if workflow[node_id].get("class_type") != expected_type
        ]
        if wrong_nodes:
            raise RuntimeError(
                "FAIL-FAST: sticker_lora_comfy workflow has incompatible node types: "
                + ", ".join(wrong_nodes)
            )
        if self.settings.require_api and not _comfy_api_alive(self.settings.api_url):
            raise RuntimeError(
                "FAIL-FAST: ComfyUI API is not reachable at "
                f"{self.settings.api_url}. Start ComfyUI before running this target."
            )


class StickerLoRAComfyFitness(FitnessFn):
    """Comfy-backed sticker candidate fitness."""

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
        if not isinstance(model, StickerLoRAComfyRunner):
            raise TypeError("sticker_lora_comfy requires StickerLoRAComfyRunner")
        problem = _single_problem(problems)
        subjects = tuple(problem[f"{self.settings.split}_subjects"])[
            : self.settings.subjects
        ]
        if not subjects:
            raise ValueError(
                "FAIL-FAST: sticker evaluation requires at least one subject"
            )
        result = model.evaluate_candidate(_candidate_from_genome(genome), subjects)
        return _scores_from_result(result)


class StickerLoRAComfyPlugin:
    """Plugin adapter for the sticker LoRA Comfy benchmark."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def dataset(self) -> DatasetProvider:
        return StickerLoRAComfyDataset(self.config)

    def model_factory(self) -> Callable[[AdapterConfig, Genome | None], ModelRunner]:
        settings = _settings(self.config)

        def create_runner(
            adapter_config: AdapterConfig, genome: Genome | None = None
        ) -> ModelRunner:
            return StickerLoRAComfyRunner(adapter_config, settings, genome)

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
    strength_model = sticker_evolve.clamp(
        0.25 + float(adapter.alpha) / 64.0, 0.15, 0.85
    )
    strength_clip = sticker_evolve.clamp(0.20 + float(adapter.dropout), 0.10, 0.70)
    cfg = round(sticker_evolve.clamp(6.0 + adapter.r / 8.0, 4.5, 10.5), 1)
    steps = (20, 24, 28, 32, 36)[adapter.r % 5]
    seed_offset = (
        stable_int({"genome_id": genome.id, "run_id": genome.run_id or ""}, 900_000)
        + 10_000
    )
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


def _comfy_api_alive(api_url: str) -> bool:
    try:
        base_url = _http_api_base_url(api_url)
        with urllib.request.urlopen(  # nosec B310 - URL scheme validated above.
            f"{base_url}/system_stats", timeout=5
        ):
            return True
    except Exception:
        return False


def _wait_for_output(
    prompt_id: str,
    api_url: str,
    timeout: float,
    comfy_output_dir: Path,
) -> Path:
    base_url = _http_api_base_url(api_url)
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(  # nosec B310 - URL scheme validated above.
            f"{base_url}/history/{prompt_id}", timeout=20
        ) as response:
            history = json.loads(response.read().decode("utf-8"))
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if (
                not status.get("status_str") == "success"
                and status.get("completed") is False
            ):
                messages = status.get("messages") or []
                raise RuntimeError(f"Comfy prompt {prompt_id} failed: {messages}")
            if item.get("outputs") and "7" not in item["outputs"]:
                raise RuntimeError(
                    f"Comfy prompt {prompt_id} completed without SaveImage output"
                )
        if item and item.get("status", {}).get("completed"):
            image = item["outputs"]["7"]["images"][0]
            return comfy_output_dir / image["filename"]
        time.sleep(1.5)
    raise TimeoutError(f"Timed out waiting for Comfy prompt {prompt_id}")


def _http_api_base_url(api_url: str) -> str:
    parsed = urlparse(api_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "FAIL-FAST: ComfyUI api_url must be an absolute http(s) URL, got "
            f"{api_url!r}"
        )
    return api_url.rstrip("/")
