"""
Default fitness function for CORAL-X.
Fallback fitness function when no specific plugin is available.
"""
from typing import Iterable, Dict, Any
from ..ports.interfaces import FitnessFn
from .genome import Genome, MultiObjectiveScores


class DefaultFitnessFunction(FitnessFn):
    """Default fitness function for M1 testing."""

    def __init__(self, config):
        self.config = config

    def __call__(self,
                 genome: Genome,
                 model,
                 problems: Iterable[Dict[str, Any]],
                 ca_features = None) -> float:
        """Single-objective evaluation."""
        multi_scores = self.evaluate_multi_objective(genome, model, problems, ca_features)
        return multi_scores.overall_fitness()

    def evaluate_multi_objective(self,
                                genome: Genome,
                                model,
                                problems: Iterable[Dict[str, Any]],
                                ca_features = None) -> MultiObjectiveScores:
        """Multi-objective evaluation with mock scores."""

        # For M1 testing, return mock scores
        return MultiObjectiveScores(
            bugfix=0.5,
            style=0.5,
            security=0.5,
            runtime=0.5,
            syntax=0.5
        )
