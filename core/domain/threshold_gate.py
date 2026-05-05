"""
Dynamic threshold filtering for multi-objective optimization.

This module implements threshold gates with σ-wave progression for
evolutionary algorithms, enabling adaptive selection pressure that
increases over generations according to CORAL-X architecture.
"""

import math
from dataclasses import dataclass

from .genome import Genome, MultiObjectiveScores


@dataclass(frozen=True)
class ObjectiveThresholds:
    """Target-neutral threshold configuration."""

    task_score: float | None = None
    quality_score: float | None = None
    risk_score: float | None = None
    efficiency_score: float | None = None
    validity_score: float | None = None
    bugfix: float | None = None
    style: float | None = None
    security: float | None = None
    runtime: float | None = None
    syntax: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for easier iteration."""
        return _neutral_threshold_dict(self)


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for dynamic threshold evolution - NO DEFAULTS."""

    base_thresholds: ObjectiveThresholds
    max_thresholds: ObjectiveThresholds
    schedule: str  # linear | sqrt | sigmoid


def calculate_sigma(gen: int, max_gen: int, mode: str = "sigmoid") -> float:
    """
    Calculate σ-wave progression factor [0,1] - CORAL-X Architecture.

    From architecture: sigma(gen) = 1/(1+exp(-12*(x-0.5))) where x = gen/max_gen
    This creates the dynamic threshold progression: loose early → strict at gen 40
    """
    if max_gen <= 0:
        return 1.0

    x = gen / max_gen

    if mode == "sigmoid":
        # CORAL-X specified formula: σ-wave with 12 coefficient
        return 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
    elif mode == "sqrt":
        return math.sqrt(x)
    elif mode == "linear":
        return x
    else:
        raise ValueError(f"  Unknown threshold schedule mode: {mode}")


def get_sla_targets() -> dict[str, float]:
    """Get default target-neutral objective targets."""
    return {
        "task_score": 0.90,
        "quality_score": 0.97,
        "risk_score": 1.0,
        "efficiency_score": 0.90,
    }


def calculate_dynamic_thresholds(
    gen: int, max_gen: int, config: ThresholdConfig
) -> ObjectiveThresholds:
    """Calculate current thresholds based on generation and σ-wave."""
    # Convert enum to string value
    schedule_str = (
        config.schedule.value
        if hasattr(config.schedule, "value")
        else str(config.schedule)
    )
    sigma = calculate_sigma(gen, max_gen, schedule_str)

    base = config.base_thresholds.to_dict()
    max_vals = config.max_thresholds.to_dict()

    current = {}
    for key in base:
        current[key] = base[key] + sigma * (max_vals[key] - base[key])

    return ObjectiveThresholds(**current)


def apply_threshold_gate(
    scores: MultiObjectiveScores, thresholds: ObjectiveThresholds
) -> bool:
    """Apply threshold gate - returns True if genome passes all thresholds."""
    score_dict = scores.to_dict()
    threshold_dict = thresholds.to_dict()

    for objective, score in score_dict.items():
        if score < threshold_dict[objective]:
            return False

    return True


def filter_population_by_thresholds(
    genomes: list[Genome], score_extractor, thresholds: ObjectiveThresholds
) -> list[Genome]:
    """Filter population by threshold gate."""
    survivors = []

    for genome in genomes:
        if not genome.is_evaluated():
            continue

        # Extract multi-objective scores from genome
        scores = score_extractor(genome)

        # Apply threshold gate
        if apply_threshold_gate(scores, thresholds):
            survivors.append(genome)

    return survivors


def _neutral_threshold_dict(thresholds: ObjectiveThresholds) -> dict[str, float]:
    values = {
        "task_score": _coalesce_threshold(
            thresholds.task_score, thresholds.bugfix
        ),
        "quality_score": _coalesce_threshold(
            thresholds.quality_score, thresholds.style
        ),
        "risk_score": _coalesce_threshold(
            thresholds.risk_score, thresholds.security
        ),
        "efficiency_score": (
            thresholds.efficiency_score
            if thresholds.efficiency_score is not None
            else thresholds.runtime
        ),
        "validity_score": (
            thresholds.validity_score
            if thresholds.validity_score is not None
            else thresholds.syntax
        ),
    }
    missing = tuple(key for key, value in values.items() if value is None)
    if missing:
        raise ValueError(
            "FAIL-FAST: missing threshold objective fields: " + ", ".join(missing)
        )
    return {key: float(value) for key, value in values.items()}


def _coalesce_threshold(neutral: float | None, legacy: float | None) -> float | None:
    return neutral if neutral is not None else legacy
