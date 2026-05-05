"""Regression tests for core review findings."""

from pathlib import Path

import numpy as np
import pytest

from core.application.evolution_orchestrator import (
    EvolutionOrchestrator,
    EvolutionServices,
)
from core.application.services import create_evolution_services
from core.common.config import (
    CacheConfig,
    CoralConfig,
    DatasetConfig,
    EvaluationConfig,
    EvolutionConfig,
    ExecutionConfig,
    ExperimentConfig,
    FitnessWeights,
    InfrastructureConfig,
    ModelConfig,
    ObjectiveThresholds,
    ThresholdConfig,
)
from core.domain.ca import CASeed
from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.mapping import LoRAConfig
from core.domain.neat import Population, _mutate_ca_seed
from core.ports.interfaces import ExecutionResult, ExecutionStatus
from core.services.genetic_operations import GeneticOperationsService
from core.services.population_manager import PopulationManager
from core.services.progress_tracker import ProgressTracker


def _config(tmp_path: Path, weights: FitnessWeights | None = None) -> CoralConfig:
    """Create a minimal core config for orchestration tests."""
    return CoralConfig(
        execution=ExecutionConfig(
            generations=1,
            population_size=3,
            output_dir=tmp_path / "output",
            selection_mode="tournament",
            survival_rate=0.5,
            crossover_rate=0.0,
        ),
        evo=EvolutionConfig(
            rank_candidates=[4],
            alpha_candidates=[8],
            dropout_candidates=[0.0],
            target_modules=["q_proj"],
        ),
        experiment=ExperimentConfig(
            target="test_target",
            name="review_regression",
            dataset=DatasetConfig(path=tmp_path / "data", datasets=["test"]),
            model=ModelConfig(name="test_model"),
        ),
        evaluation=EvaluationConfig(
            test_samples=1,
            fitness_weights=weights
            or FitnessWeights(
                task_score=0.2,
                quality_score=0.2,
                risk_score=0.2,
                efficiency_score=0.2,
                validity_score=0.2,
            ),
        ),
        infra=InfrastructureConfig(executor="local"),
        cache=CacheConfig(
            artifacts_dir=tmp_path / "cache",
            base_checkpoint="test_model",
        ),
        threshold=ThresholdConfig(
            base_thresholds=ObjectiveThresholds(
                task_score=0.0,
                quality_score=0.0,
                risk_score=0.0,
                efficiency_score=0.0,
                validity_score=0.0,
            ),
            max_thresholds=ObjectiveThresholds(
                task_score=0.0,
                quality_score=0.0,
                risk_score=0.0,
                efficiency_score=0.0,
                validity_score=0.0,
            ),
        ),
        seed=123,
    )


def _genome(genome_id: str) -> Genome:
    return Genome(
        seed=CASeed(grid=np.zeros((2, 2), dtype=int), rule=30, steps=1),
        lora_cfg=LoRAConfig(
            r=4,
            alpha=8,
            dropout=0.0,
            target_modules=("q_proj",),
            adapter_type="lora",
        ),
        id=genome_id,
    )


class ConstantFitness:
    """Fitness function with intentionally asymmetric objective scores."""

    def __call__(self, genome, model, problems):
        return self.evaluate_multi_objective(genome, model, problems).overall_fitness()

    def evaluate_multi_objective(self, genome, model, problems, ca_features=None):
        return MultiObjectiveScores(
            task_score=1.0,
            quality_score=0.0,
            risk_score=0.0,
            efficiency_score=0.0,
            validity_score=0.0,
        )


class StaticDataset:
    def problems(self):
        return [{"id": "problem-1"}]


class StaticPlugin:
    def __init__(self):
        self.dataset_provider = StaticDataset()
        self.fitness = ConstantFitness()

    def dataset(self):
        return self.dataset_provider

    def model_factory(self):
        return lambda lora_cfg, genome=None: object()

    def fitness_fn(self):
        return self.fitness


class RecordingExecutor:
    """Executor that records submitted genome IDs and runs work immediately."""

    def __init__(self):
        self.submitted_genomes: list[str] = []

    def submit(self, fn, *args, **kwargs):
        genome = args[0]
        self.submitted_genomes.append(genome.id)
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            result=fn(*args, **kwargs),
        )


def _services(config: CoralConfig, executor: RecordingExecutor) -> EvolutionServices:
    return EvolutionServices(
        population_manager=PopulationManager(config, config.seed),
        genetic_operations=GeneticOperationsService(config, config.seed),
        progress_tracker=ProgressTracker(config, "review-test"),
        fitness_fn=ConstantFitness(),
        executor=executor,
        config=config,
        dataset_provider=StaticDataset(),
        model_factory=lambda lora_cfg, genome=None: object(),
    )


@pytest.mark.asyncio
async def test_orchestrator_uses_configured_fitness_weights(tmp_path):
    """Scalar fitness should come from experiment weights, not hardcoded defaults."""
    config = _config(
        tmp_path,
        weights=FitnessWeights(
            task_score=1.0,
            quality_score=0.0,
            risk_score=0.0,
            efficiency_score=0.0,
            validity_score=0.0,
        ),
    )
    orchestrator = EvolutionOrchestrator(_services(config, RecordingExecutor()))

    population = await orchestrator._evaluate_population(
        Population((_genome("weighted-genome"),)), generation=0
    )

    assert population.genomes[0].fitness == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_orchestrator_evaluates_genomes_through_executor(tmp_path):
    """Configured executor should own candidate evaluation execution."""
    config = _config(tmp_path)
    executor = RecordingExecutor()
    orchestrator = EvolutionOrchestrator(_services(config, executor))

    await orchestrator._evaluate_population(
        Population((_genome("g0"), _genome("g1"))), generation=0
    )

    assert executor.submitted_genomes == ["g0", "g1"]


def test_ca_seed_grid_mutation_can_activate_empty_binary_grid():
    """A forced grid mutation should be able to flip an all-zero CA grid."""

    class ForcedGridMutationRng:
        def __init__(self):
            self.random_values = iter([0.0, *([1.0] * 100)])

        def random(self):
            return next(self.random_values)

        def randint(self, low, high):
            return low

    seed = CASeed(grid=np.zeros((4, 4), dtype=int), rule=30, steps=5)
    mutated = _mutate_ca_seed(seed, ForcedGridMutationRng())

    assert int(mutated.grid.sum()) > 0


def test_create_evolution_services_accepts_plugin_dependency(tmp_path):
    """Core service wiring should receive plugins as dependencies."""
    services = create_evolution_services(_config(tmp_path), plugin=StaticPlugin())

    assert isinstance(services.dataset_provider, StaticDataset)
    assert isinstance(services.fitness_fn, ConstantFitness)


def test_core_application_services_does_not_import_plugin_registry():
    """Core application services should not resolve concrete plugin modules."""
    source = Path("core/application/services.py").read_text()

    assert "plugins.registry" not in source
