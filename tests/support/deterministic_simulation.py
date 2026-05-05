"""Deterministic simulation harness for core CORAL-X evolution tests.

The harness intentionally checks replayability and invariants, not whether a
tiny stochastic run improves. Improvement is a benchmark/runtime claim; these
tests prove the local CA, scoring, selection, and reproduction contracts remain
stable for a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.common.config import (
    CacheConfig,
    CAConfig,
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
    SelectionMode,
    ThresholdConfig,
    TrainingConfig,
)
from core.domain.experiment import create_experiment_config, create_initial_population
from core.domain.genome import Genome
from core.domain.neat import Population
from core.domain.stable_hash import stable_digest
from core.services.genetic_operations import GeneticOperationsService
from core.services.population_manager import PopulationManager
from plugins.ca_onemax.plugin import CAOneMaxFitness, CAOneMaxRunner


@dataclass(frozen=True)
class GenomeTrace:
    """Path-independent snapshot of one genome in a simulation generation."""

    genome_id: str
    grid_hash: str
    active_cells: int
    total_cells: int
    rule: int
    steps: int
    lora: tuple[int, float, float, tuple[str, ...], str]
    fitness: float | None
    scores: tuple[float, ...] | None


@dataclass(frozen=True)
class GenerationTrace:
    """Deterministic snapshot of one simulated generation."""

    generation: int
    evaluated_population: tuple[GenomeTrace, ...]
    survivor_ids: tuple[str, ...]
    next_population_ids: tuple[str, ...]
    best_fitness: float
    mean_fitness: float


@dataclass(frozen=True)
class SimulationTrace:
    """Complete deterministic trace for a no-model CORAL-X simulation."""

    selection_mode: str
    seed: int
    generations: tuple[GenerationTrace, ...]


def build_simulation_config(
    root: Path,
    *,
    seed: int = 42,
    selection_mode: SelectionMode = SelectionMode.TOURNAMENT,
    generations: int = 3,
    population_size: int = 6,
    crossover_rate: float = 0.4,
) -> CoralConfig:
    """Create a small CA OneMax config for deterministic core simulations."""

    return CoralConfig(
        execution=ExecutionConfig(
            generations=generations,
            population_size=population_size,
            max_workers=1,
            output_dir=root / "artifacts",
            selection_mode=selection_mode,
            survival_rate=0.5,
            crossover_rate=crossover_rate,
            run_held_out_benchmark=False,
        ),
        evo=EvolutionConfig(
            ca=CAConfig(
                grid_size=[5, 5],
                rule_range=[30, 255],
                steps_range=[1, 5],
                initial_density=0.35,
            ),
            rank_candidates=[4, 8, 16],
            alpha_candidates=[8, 16, 32],
            dropout_candidates=[0.05, 0.1, 0.2],
            target_modules=["q_proj", "v_proj"],
        ),
        experiment=ExperimentConfig(
            target="ca_onemax",
            name="deterministic_simulation",
            dataset=DatasetConfig(path=root / "datasets", datasets=["ca_onemax"]),
            model=ModelConfig(name="no_model_required", max_seq_length=128),
        ),
        training=TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            epochs=1,
            learning_rate=2e-4,
            warmup_steps=0,
            logging_steps=1,
            save_steps=10,
            max_grad_norm=1.0,
            weight_decay=0.01,
            adam_epsilon=1e-8,
        ),
        evaluation=EvaluationConfig(
            test_samples=1,
            fitness_weights=FitnessWeights(
                task_score=0.2,
                quality_score=0.2,
                risk_score=0.2,
                efficiency_score=0.2,
                validity_score=0.2,
            ),
        ),
        infra=InfrastructureConfig(executor="local"),
        cache=CacheConfig(
            artifacts_dir=root / "cache",
            base_checkpoint="no_model_required",
            run_id="deterministic-sim",
        ),
        threshold=ThresholdConfig(
            base_thresholds=_thresholds(0.0),
            max_thresholds=_thresholds(0.0),
        ),
        seed=seed,
    )


def run_deterministic_simulation(config: CoralConfig) -> SimulationTrace:
    """Run a deterministic, no-model simulation through core service paths."""

    population_manager = PopulationManager(config, config.seed)
    genetic_operations = GeneticOperationsService(config, config.seed)
    raw_config = config.model_dump()
    experiment_config = create_experiment_config(raw_config)
    population = create_initial_population(
        experiment_config,
        diversity_strength=1.0,
        raw_config=raw_config,
        run_id=config.cache.run_id,
    )

    generations: list[GenerationTrace] = []
    for generation in range(config.execution.generations):
        evaluated = _evaluate_population(config, population)
        filtered = population_manager.apply_threshold_gate(evaluated, generation)
        population_manager.record_generation_stats(filtered)

        survivor_ids: tuple[str, ...] = ()
        next_population_ids: tuple[str, ...]
        if generation < config.execution.generations - 1:
            genetic_operations.adjust_genetic_parameters(filtered, generation)
            survivors = population_manager.select_survivors(filtered)
            survivor_ids = tuple(genome.id for genome in survivors.genomes)
            population = genetic_operations.reproduce_population(
                survivors,
                config.execution.population_size,
                generation,
            )
            next_population_ids = tuple(genome.id for genome in population.genomes)
        else:
            population = filtered
            next_population_ids = tuple(genome.id for genome in filtered.genomes)

        fitness_values = [genome.fitness for genome in filtered.genomes]
        generations.append(
            GenerationTrace(
                generation=generation,
                evaluated_population=tuple(
                    genome_trace(genome) for genome in filtered.genomes
                ),
                survivor_ids=survivor_ids,
                next_population_ids=next_population_ids,
                best_fitness=round(max(fitness_values), 12),
                mean_fitness=round(sum(fitness_values) / len(fitness_values), 12),
            )
        )

    return SimulationTrace(
        selection_mode=config.execution.selection_mode.value,
        seed=config.seed,
        generations=tuple(generations),
    )


def genome_trace(genome: Genome) -> GenomeTrace:
    """Return a compact, stable genome signature suitable for trace equality."""

    scores = None
    if genome.multi_scores is not None:
        scores = tuple(
            round(value, 12) for value in genome.multi_scores.to_dict().values()
        )

    return GenomeTrace(
        genome_id=genome.id,
        grid_hash=stable_digest(genome.seed.grid, length=16),
        active_cells=int(genome.seed.grid.sum()),
        total_cells=int(genome.seed.grid.size),
        rule=int(genome.seed.rule),
        steps=int(genome.seed.steps),
        lora=(
            int(genome.lora_cfg.r),
            float(genome.lora_cfg.alpha),
            float(genome.lora_cfg.dropout),
            tuple(genome.lora_cfg.target_modules),
            genome.lora_cfg.adapter_type,
        ),
        fitness=round(genome.fitness, 12) if genome.fitness is not None else None,
        scores=scores,
    )


def assert_core_simulation_invariants(
    trace: SimulationTrace, config: CoralConfig
) -> None:
    """Assert correctness invariants that should hold independent of improvement."""

    assert len(trace.generations) == config.execution.generations

    for generation in trace.generations:
        evaluated_ids = [genome.genome_id for genome in generation.evaluated_population]
        assert len(evaluated_ids) == len(set(evaluated_ids))
        assert len(generation.evaluated_population) == config.execution.population_size

        for genome in generation.evaluated_population:
            expected_score = genome.active_cells / genome.total_cells
            assert genome.fitness == pytest.approx(expected_score)
            assert genome.scores is not None
            assert all(
                score == pytest.approx(expected_score) for score in genome.scores
            )

        if generation.generation < config.execution.generations - 1:
            assert len(generation.survivor_ids) >= 1
            assert set(generation.survivor_ids).issubset(evaluated_ids)
            assert (
                len(generation.next_population_ids) == config.execution.population_size
            )
            assert (
                generation.next_population_ids[: len(generation.survivor_ids)]
                == generation.survivor_ids
            )


def _evaluate_population(config: CoralConfig, population: Population) -> Population:
    fitness_fn = CAOneMaxFitness()
    model = CAOneMaxRunner()
    problems = [{"name": "ca_onemax"}]
    weights = config.evaluation.fitness_weights.to_dict()
    evaluated = []
    for genome in population.genomes:
        if genome.is_evaluated():
            evaluated.append(genome)
            continue
        scores = fitness_fn.evaluate_multi_objective(genome, model, problems)
        fitness = scores.overall_fitness(weights=weights)
        evaluated.append(
            genome.with_fitness(fitness).with_multi_scores(scores, weights)
        )
    return Population(tuple(evaluated))


def _thresholds(value: float) -> ObjectiveThresholds:
    return ObjectiveThresholds(
        task_score=value,
        quality_score=value,
        risk_score=value,
        efficiency_score=value,
        validity_score=value,
    )
