"""Controlled OneMax benchmark over the CA grid genome."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from core.domain.cheap_knobs import CheapKnobs
from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.mapping import LoRAConfig
from core.ports.interfaces import DatasetProvider, FitnessFn, ModelRunner


class CAOneMaxDataset(DatasetProvider):
    """Single synthetic optimization task."""

    def problems(self) -> Iterable[dict[str, Any]]:
        yield {
            "name": "ca_onemax",
            "description": "Maximize active cells in the CA seed grid",
        }


class CAOneMaxRunner(ModelRunner):
    """No-op runner; fitness is computed directly from the genome."""

    def generate(
        self, prompt: str, max_tokens: int, cheap_knobs: CheapKnobs | None = None
    ) -> str:
        return prompt


class CAOneMaxFitness(FitnessFn):
    """Fitness score for the controlled CA OneMax task."""

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
        active = float(genome.seed.grid.sum())
        total = float(genome.seed.grid.size)
        score = active / total if total else 0.0
        print(
            f"CA OneMax evaluation: genome={genome.id}, "
            f"active={int(active)}/{int(total)}, fitness={score:.4f}"
        )
        return MultiObjectiveScores(
            bugfix=score,
            style=score,
            security=score,
            runtime=score,
            syntax=score,
        )


class CAOneMaxPlugin:
    """Controlled benchmark plugin for validating evolutionary search pressure."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        print("CA OneMax plugin initialized")

    def dataset(self) -> DatasetProvider:
        return CAOneMaxDataset()

    def model_factory(self) -> Callable[[LoRAConfig, Genome | None], ModelRunner]:
        def create_model(
            lora_cfg: LoRAConfig, genome: Genome | None = None
        ) -> ModelRunner:
            return CAOneMaxRunner()

        return create_model

    def fitness_fn(self) -> FitnessFn:
        return CAOneMaxFitness()
