"""Default fitness function for local CORAL-X wiring tests."""

from collections.abc import Iterable
from typing import Any

from ..ports.interfaces import FitnessFn
from .genome import Genome, MultiObjectiveScores


class DefaultFitnessFunction(FitnessFn):
    """Deterministic neutral fitness function for wiring tests."""

    def __init__(self, config):
        self.config = config

    def __call__(
        self,
        genome: Genome,
        model,
        problems: Iterable[dict[str, Any]],
        ca_features=None,
    ) -> float:
        """Single-objective evaluation."""
        multi_scores = self.evaluate_multi_objective(
            genome, model, problems, ca_features
        )
        return multi_scores.overall_fitness()

    def evaluate_multi_objective(
        self,
        genome: Genome,
        model,
        problems: Iterable[dict[str, Any]],
        ca_features=None,
    ) -> MultiObjectiveScores:
        """Multi-objective evaluation with neutral constant scores."""

        return MultiObjectiveScores(
            task_score=0.5,
            quality_score=0.5,
            risk_score=0.5,
            efficiency_score=0.5,
            validity_score=0.5,
        )
