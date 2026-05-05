"""Pydantic configuration models for CORAL-X."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExecutorType(str, Enum):
    LOCAL = "local"


class SelectionMode(str, Enum):
    PARETO = "pareto"
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"


class ThresholdSchedule(str, Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"


class DiversityMode(str, Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class PathsConfig(BaseModel):
    """Path configuration for different execution environments."""

    cache_root: Path = Field(default=Path("./cache"))
    dataset: Path = Field(default=Path("./data"))
    models: Path = Field(default=Path("./models"))
    results: Path = Field(default=Path("./results"))
    adapters: Path = Field(default=Path("./cache/adapters"))
    exports: Path = Field(default=Path("./exports"))
    progress: Path = Field(default=Path("./cache/evolution_progress.json"))
    emergent_behavior: Path = Field(default=Path("./cache/emergent_behavior"))
    emergent_alerts: Path = Field(default=Path("./cache/emergent_behavior/alerts.json"))
    realtime_benchmarks: Path = Field(default=Path("./cache/realtime_benchmarks"))
    coralx_root: Path | None = None


class CAConfig(BaseModel):
    """Cellular Automata configuration."""

    grid_size: list[int] = Field(default=[8, 8], min_length=2, max_length=2)
    rule_range: list[int] = Field(default=[30, 255], min_length=2, max_length=2)
    steps_range: list[int] = Field(default=[5, 20], min_length=2, max_length=2)
    initial_density: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("grid_size")
    @classmethod
    def validate_grid_size(cls, v):
        if any(size <= 0 for size in v):
            raise ValueError("Grid dimensions must be positive")
        return v

    @field_validator("rule_range", "steps_range")
    @classmethod
    def validate_ranges(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Range minimum must be less than maximum")
        return v


class EvolutionConfig(BaseModel):
    """Evolution algorithm configuration."""

    ca: CAConfig | None = Field(default_factory=CAConfig)
    rank_candidates: list[int] = Field(default=[4, 8, 16, 32], min_length=1)
    alpha_candidates: list[int] = Field(default=[8, 16, 32], min_length=1)
    dropout_candidates: list[float] = Field(default=[0.05, 0.1, 0.15], min_length=1)
    target_modules: list[str] = Field(default=["q_proj", "v_proj"], min_length=1)

    @field_validator("rank_candidates", "alpha_candidates")
    @classmethod
    def validate_positive_integers(cls, v):
        if any(val <= 0 for val in v):
            raise ValueError("All candidates must be positive integers")
        return v

    @field_validator("dropout_candidates")
    @classmethod
    def validate_dropout_range(cls, v):
        if any(val < 0.0 or val > 1.0 for val in v):
            raise ValueError("Dropout values must be between 0.0 and 1.0")
        return v


class ExecutionConfig(BaseModel):
    """Execution parameters for evolution."""

    generations: int = Field(gt=0, le=1000)
    population_size: int = Field(gt=1, le=1000)
    output_dir: Path = Field(default=Path("./results"))
    selection_mode: SelectionMode = Field(default=SelectionMode.PARETO)
    survival_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    run_held_out_benchmark: bool = Field(default=False)
    proof_seeds: list[int] = Field(default_factory=list)
    current_generation: int = Field(default=0, ge=0)

    @field_validator("crossover_rate")
    @classmethod
    def validate_crossover_rate(cls, v):
        # Ensure crossover rate is valid (mutation rate validation would be in parent config)
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    path: Path
    dataset_path: Path | None = None
    max_samples: int | None = Field(None, gt=0)
    datasets: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def set_dataset_path(cls, values):
        if isinstance(values, dict) and values.get("dataset_path") is None:
            values["dataset_path"] = values.get("path")
        return values


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str
    model_name: str | None = None
    max_seq_length: int = Field(default=512, gt=0, le=4096)

    @model_validator(mode="before")
    @classmethod
    def set_model_name(cls, values):
        if isinstance(values, dict) and values.get("model_name") is None:
            values["model_name"] = values.get("name")
        return values


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    target: str
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: dict[str, Any] | None = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    epochs: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    warmup_steps: int = Field(default=100, ge=0)
    logging_steps: int = Field(default=50, gt=0)
    save_steps: int = Field(default=500, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    adam_epsilon: float = Field(default=1e-8, gt=0.0)
    save_strategy: str = Field(default="steps")


class FitnessWeights(BaseModel):
    """Target-neutral fitness weights."""

    task_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    efficiency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    validity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    bugfix: float | None = Field(default=None, ge=0.0, le=1.0)
    style: float | None = Field(default=None, ge=0.0, le=1.0)
    security: float | None = Field(default=None, ge=0.0, le=1.0)
    runtime: float | None = Field(default=None, ge=0.0, le=1.0)
    syntax: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum(self):
        total = sum(self.to_dict().values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Fitness weights must sum to 1.0, got {total}")
        return self

    def to_dict(self) -> dict[str, float]:
        return _neutral_objective_dict(self, "fitness weight")


class BaselineTestConfig(BaseModel):
    """Baseline performance testing configuration."""

    enabled: bool = Field(default=True)
    test_samples: int = Field(default=30, gt=0)
    multiple_attempts: int = Field(default=3, gt=0)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    improvement_threshold: float = Field(
        default=0.05, ge=0.0
    )  # 5% improvement threshold
    prompt_styles: list[str] = Field(
        default=[
            "Classify this news as fake or real:\n\n{text}\n\nClassification:",
            "Is this news article real or fake?\n\n{text}\n\nAnswer:",
            "Determine if this is genuine news or misinformation:\n\n{text}\n\nResult:",
        ]
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    test_samples: int = Field(gt=0)
    fitness_weights: FitnessWeights
    adaptive_testing: dict[str, Any] | None = None
    baseline_testing: BaselineTestConfig = Field(default_factory=BaselineTestConfig)


class InfrastructureConfig(BaseModel):
    """Infrastructure configuration."""

    executor: ExecutorType


class PlottingConfig(BaseModel):
    """Plotting and visualization configuration."""

    enabled: bool = Field(default=True)
    output_dir: Path = Field(default=Path("./artifacts/plots"))
    pareto_fronts: bool = Field(default=True)
    hypervolume_progress: bool = Field(default=True)
    generation_summaries: bool = Field(default=True)


class ExportConfig(BaseModel):
    """Export configuration for results."""

    enabled: bool = Field(default=True)
    output_dir: Path = Field(default=Path("./artifacts/exports"))
    csv_per_generation: bool = Field(default=True)
    csv_final_population: bool = Field(default=True)
    jsonl_detailed: bool = Field(default=True)


class CacheConfig(BaseModel):
    """Cache configuration."""

    artifacts_dir: Path
    base_checkpoint: str
    metadata: bool = Field(default=True)
    cleanup_threshold: int = Field(default=100, gt=0)
    run_id: str | None = None


class CheapKnobsConfig(BaseModel):
    """Generation parameter ranges."""

    temperature_range: list[float] = Field(min_length=2, max_length=2)
    top_p_range: list[float] = Field(min_length=2, max_length=2)
    top_k_range: list[int] = Field(min_length=2, max_length=2)
    repetition_penalty_range: list[float] = Field(min_length=2, max_length=2)
    max_tokens_range: list[int] = Field(min_length=2, max_length=2)

    @field_validator("temperature_range", "top_p_range", "repetition_penalty_range")
    @classmethod
    def validate_float_ranges(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Range minimum must be less than maximum")
        return v

    @field_validator("top_k_range", "max_tokens_range")
    @classmethod
    def validate_int_ranges(cls, v):
        if v[0] >= v[1] or v[0] <= 0:
            raise ValueError("Range minimum must be less than maximum and positive")
        return v


def _neutral_objective_dict(source: Any, context: str) -> dict[str, float]:
    values = {
        "task_score": _coalesce(source, "task_score", "bugfix"),
        "quality_score": _coalesce(source, "quality_score", "style"),
        "risk_score": _coalesce(source, "risk_score", "security"),
        "efficiency_score": _coalesce(source, "efficiency_score", "runtime"),
        "validity_score": _coalesce(source, "validity_score", "syntax"),
    }
    missing = tuple(key for key, value in values.items() if value is None)
    if missing:
        raise ValueError(
            f"Missing {context} objective fields: " + ", ".join(missing)
        )
    return {key: float(value) for key, value in values.items()}


def _coalesce(source: Any, neutral_key: str, legacy_key: str) -> float | None:
    neutral = getattr(source, neutral_key)
    legacy = getattr(source, legacy_key)
    if neutral is not None and legacy is not None:
        raise ValueError(
            f"Provide either '{neutral_key}' or legacy alias '{legacy_key}', not both"
        )
    return neutral if neutral is not None else legacy


class ObjectiveThresholds(BaseModel):
    """Target-neutral threshold values for objectives."""

    task_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    efficiency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    validity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    bugfix: float | None = Field(default=None, ge=0.0, le=1.0)
    runtime: float | None = Field(default=None, ge=0.0, le=1.0)
    security: float | None = Field(default=None, ge=0.0, le=1.0)
    style: float | None = Field(default=None, ge=0.0, le=1.0)
    syntax: float | None = Field(default=None, ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for compatibility with threshold gate functions."""
        return _neutral_objective_dict(self, "threshold")


class ThresholdConfig(BaseModel):
    """Threshold configuration for objectives."""

    base_thresholds: ObjectiveThresholds
    max_thresholds: ObjectiveThresholds
    schedule: ThresholdSchedule = Field(default=ThresholdSchedule.SIGMOID)


class DiversityConfig(BaseModel):
    """Diversity configuration."""

    mode: DiversityMode = Field(default=DiversityMode.ADAPTIVE)
    base_strength: float = Field(default=1.0, ge=0.0)


class OrganizationConfig(BaseModel):
    """Project organization and cleanup configuration."""

    use_run_ids: bool = Field(default=True)
    run_id_format: str = Field(default="run_%Y%m%d_%H%M%S")
    preserve_runs: int = Field(default=10, ge=1)
    auto_cleanup: bool = Field(default=True)


class ProjectPathsConfig(BaseModel):
    """Project-wide path configuration."""

    logs: str = Field(default="./logs")
    cache: str = Field(default="./cache")
    runs: str = Field(default="./runs")
    datasets: str = Field(default="./datasets")


class CoralConfig(BaseModel):
    """Complete CORAL-X configuration."""

    execution: ExecutionConfig
    evo: EvolutionConfig
    experiment: ExperimentConfig
    training: TrainingConfig | None = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig
    infra: InfrastructureConfig
    cache: CacheConfig
    cheap_knobs: CheapKnobsConfig | None = None
    threshold: ThresholdConfig
    diversity: DiversityConfig | None = Field(default_factory=DiversityConfig)
    organization: OrganizationConfig | None = Field(default_factory=OrganizationConfig)
    paths: ProjectPathsConfig | None = Field(default_factory=ProjectPathsConfig)
    plotting: PlottingConfig | None = Field(default_factory=PlottingConfig)
    export: ExportConfig | None = Field(default_factory=ExportConfig)
    seed: int = Field(default=42)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Disallow extra fields not defined in the schema
    )
