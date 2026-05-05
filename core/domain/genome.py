"""Genome data structures for target-independent evolutionary candidates."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .ca import CASeed
from .mapping import LoRAConfig
from .objectives import ObjectiveVector

if TYPE_CHECKING:
    from .feature_extraction import CAFeatures


LEGACY_OBJECTIVE_ALIASES = {
    "bugfix": "task_score",
    "style": "quality_score",
    "security": "risk_score",
    "runtime": "efficiency_score",
    "syntax": "validity_score",
}

DEFAULT_OBJECTIVE_WEIGHTS = {
    "task_score": 0.3,
    "quality_score": 0.15,
    "risk_score": 0.25,
    "efficiency_score": 0.1,
    "validity_score": 0.2,
}


@dataclass(frozen=True, init=False)
class MultiObjectiveScores:
    """Multi-objective evaluation scores for CORAL-X."""

    task_score: float
    quality_score: float
    risk_score: float
    efficiency_score: float
    validity_score: float

    def __init__(
        self,
        task_score: float | None = None,
        quality_score: float | None = None,
        risk_score: float | None = None,
        efficiency_score: float | None = None,
        validity_score: float | None = None,
        **legacy_scores: float,
    ):
        """Create neutral scores, accepting legacy plugin aliases at the boundary."""
        values = {
            "task_score": task_score,
            "quality_score": quality_score,
            "risk_score": risk_score,
            "efficiency_score": efficiency_score,
            "validity_score": validity_score,
        }
        for legacy_key, neutral_key in LEGACY_OBJECTIVE_ALIASES.items():
            if legacy_key not in legacy_scores:
                continue
            if values[neutral_key] is not None:
                raise ValueError(
                    f"FAIL-FAST: provide either '{neutral_key}' or legacy alias "
                    f"'{legacy_key}', not both"
                )
            values[neutral_key] = legacy_scores.pop(legacy_key)
        if legacy_scores:
            unknown = ", ".join(sorted(legacy_scores))
            raise ValueError(f"FAIL-FAST: unknown objective score fields: {unknown}")
        missing = tuple(key for key, value in values.items() if value is None)
        if missing:
            raise ValueError(
                "FAIL-FAST: missing objective score fields: " + ", ".join(missing)
            )
        for key, value in values.items():
            score = float(value)
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    f"FAIL-FAST: objective '{key}' must be between 0.0 and 1.0"
                )
            object.__setattr__(self, key, score)

    def overall_fitness(self, weights: dict[str, float] | None = None) -> float:
        """Calculate overall fitness as weighted average."""
        if weights is None:
            weights = DEFAULT_OBJECTIVE_WEIGHTS

        return self.as_objective_vector(weights=weights).weighted_fitness()

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for easier iteration."""
        return {
            "task_score": self.task_score,
            "quality_score": self.quality_score,
            "risk_score": self.risk_score,
            "efficiency_score": self.efficiency_score,
            "validity_score": self.validity_score,
        }

    def as_objective_vector(
        self,
        weights: dict[str, float] | None = None,
        labels: dict[str, str] | None = None,
    ) -> ObjectiveVector:
        """Expose legacy score fields through the target-neutral objective API."""
        if weights is None:
            weights = DEFAULT_OBJECTIVE_WEIGHTS
        weights = _normalize_objective_mapping(weights)
        labels = _normalize_objective_mapping(labels or {})
        return ObjectiveVector.from_mapping(
            values=self.to_dict(),
            weights=weights,
            labels=labels,
        )

    @property
    def bugfix(self) -> float:
        """Legacy alias for task success."""
        return self.task_score

    @property
    def style(self) -> float:
        """Legacy alias for output quality."""
        return self.quality_score

    @property
    def security(self) -> float:
        """Legacy alias for risk-control score."""
        return self.risk_score

    @property
    def runtime(self) -> float:
        """Legacy alias for efficiency score."""
        return self.efficiency_score

    @property
    def syntax(self) -> float:
        """Legacy alias for validity score."""
        return self.validity_score


def _normalize_objective_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Translate legacy objective keys to neutral keys."""
    normalized = {}
    for key, value in mapping.items():
        normalized[LEGACY_OBJECTIVE_ALIASES.get(key, key)] = value
    return normalized


@dataclass(frozen=True)
class Genome:
    """CA seed + mapped candidate parameters + scores + CA features."""

    seed: CASeed
    lora_cfg: LoRAConfig
    id: str  # Unique genome identifier
    ca_features: Optional["CAFeatures"] = None
    fitness: float | None = None
    multi_scores: MultiObjectiveScores | None = None
    metadata: dict[str, Any] | None = None
    run_id: str | None = None  # Experiment-specific identifier

    def with_fitness(self, fitness: float) -> "Genome":
        """Return new genome with updated fitness score."""
        return Genome(
            seed=self.seed,
            lora_cfg=self.lora_cfg,
            id=self.id,
            ca_features=self.ca_features,
            fitness=fitness,
            multi_scores=self.multi_scores,
            metadata=self.metadata,
            run_id=self.run_id,
        )

    def with_multi_scores(
        self, scores: MultiObjectiveScores, weights: dict[str, float] | None = None
    ) -> "Genome":
        """Return new genome with multi-objective scores."""
        # Update overall fitness based on multi-objective scores
        overall_fitness = scores.overall_fitness(weights=weights)
        return Genome(
            seed=self.seed,
            lora_cfg=self.lora_cfg,
            id=self.id,
            ca_features=self.ca_features,
            fitness=overall_fitness,
            multi_scores=scores,
            metadata=self.metadata,
            run_id=self.run_id,
        )

    def with_metadata(self, metadata: dict[str, Any]) -> "Genome":
        """Return new genome with updated metadata."""
        return Genome(
            seed=self.seed,
            lora_cfg=self.lora_cfg,
            id=self.id,
            ca_features=self.ca_features,
            fitness=self.fitness,
            multi_scores=self.multi_scores,
            metadata=metadata,
            run_id=self.run_id,
        )

    def with_ca_features(self, ca_features: "CAFeatures") -> "Genome":
        """Return new genome with CA features for consistency."""
        return Genome(
            seed=self.seed,
            lora_cfg=self.lora_cfg,
            id=self.id,
            ca_features=ca_features,
            fitness=self.fitness,
            multi_scores=self.multi_scores,
            metadata=self.metadata,
            run_id=self.run_id,
        )

    def is_evaluated(self) -> bool:
        """Check if genome has been evaluated."""
        return self.fitness is not None

    def has_multi_scores(self) -> bool:
        """Check if genome has multi-objective scores."""
        return self.multi_scores is not None

    def get_heavy_genes_key(self) -> tuple:
        """Extract heavy genes that require adapter training."""
        return (
            self.lora_cfg.r,
            self.lora_cfg.alpha,
            self.lora_cfg.dropout,
            self.lora_cfg.target_modules,
            self.lora_cfg.adapter_type,
            self.run_id,
        )

    def __lt__(self, other: "Genome") -> bool:
        """Support sorting by fitness (higher is better)."""
        if self.fitness is None and other.fitness is None:
            return False
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness
