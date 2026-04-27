"""Evolution orchestrator for CORAL-X using service composition."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..common.config import CoralConfig
from ..common.exceptions import EvolutionError
from ..common.logging import LoggingMixin
from ..domain.genome import Genome
from ..domain.mapping import LoRAConfig
from ..domain.neat import Population
from ..domain.stable_hash import stable_digest
from ..ports.interfaces import DatasetProvider, Executor, FitnessFn, ModelRunner
from ..services.genetic_operations import GeneticOperationsService
from ..services.population_manager import PopulationManager
from ..services.progress_tracker import ProgressTracker


@dataclass
class EvolutionResult:
    """Result of evolution process."""

    final_population: Population
    best_genome: Genome | None
    generations_completed: int
    total_time: float
    status: str


@dataclass
class EvolutionServices:
    """Dependency injection container for evolution services."""

    population_manager: PopulationManager
    genetic_operations: GeneticOperationsService
    progress_tracker: ProgressTracker
    fitness_fn: FitnessFn
    executor: Executor
    config: CoralConfig
    dataset_provider: DatasetProvider | None = None
    model_factory: Callable[[LoRAConfig, Genome | None], ModelRunner] | None = None


class EvolutionOrchestrator(LoggingMixin):
    """Orchestrates evolution process with clear separation of concerns."""

    def __init__(self, services: EvolutionServices):
        super().__init__()
        self.services = services
        self.config = services.config

        # Evolution state
        self.start_time = None
        self.current_generation = 0
        self.max_generations = self.config.execution.generations

        self.logger.info(
            f"Evolution orchestrator initialized: {self.config.experiment.name}, max_generations={self.max_generations}, population_size={self.config.execution.population_size}, executor={self.config.infra.executor}"
        )

    async def run_evolution(self) -> EvolutionResult:
        """Main evolution loop - focused only on orchestration.

        Returns:
            EvolutionResult with final state
        """
        self.start_time = time.time()

        try:
            self._validate_preconditions()

            self.logger.info("Evolution started")
            self.services.progress_tracker.update_status(
                "starting", "Evolution starting..."
            )

            # Initialize population
            population = await self._initialize_population()

            # Main evolution loop
            for generation in range(self.max_generations):
                self.current_generation = generation

                self.logger.info(
                    f"Generation {generation + 1}/{self.max_generations} started"
                )

                population = await self._run_generation(generation, population)

                # Check early stopping conditions
                if self._should_stop_early(population, generation):
                    self.logger.info(
                        f"Early stopping triggered at generation {generation}"
                    )
                    break

            # Evolution completed
            result = self._create_result(population, "completed")

            self.logger.info(
                f"Evolution completed: {result.generations_completed} generations in {result.total_time:.2f}s, best fitness: {result.best_genome.fitness if result.best_genome else 0.0:.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
            self.services.progress_tracker.update_status(
                "failed", f"Evolution failed: {e}"
            )

            # Create failure result
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            return EvolutionResult(
                final_population=Population(()),
                best_genome=None,
                generations_completed=self.current_generation,
                total_time=elapsed_time,
                status="failed",
            )

    def _validate_preconditions(self) -> None:
        """Validate that evolution can run successfully.

        Raises:
            EvolutionError: If preconditions are not met
        """
        if self.config.execution.generations <= 0:
            raise EvolutionError("Generations must be positive")

        if self.config.execution.population_size <= 0:
            raise EvolutionError("Population size must be positive")

        if not self.services.fitness_fn:
            raise EvolutionError("Fitness function is required")

        if not self.services.executor:
            raise EvolutionError("Executor is required")

        if not self.services.dataset_provider:
            raise EvolutionError("Dataset provider is required")

        if not self.services.model_factory:
            raise EvolutionError("Model factory is required")

        self.logger.info("Preconditions validated")

    async def _initialize_population(self) -> Population:
        """Initialize the starting population.

        Returns:
            Initial population
        """
        self.logger.info(
            f"Initializing population of size {self.config.execution.population_size}"
        )

        # Create initial population using domain function
        # Convert full config to experiment config format
        from core.domain.experiment import (
            create_experiment_config,
            create_initial_population,
        )

        experiment_config = create_experiment_config(self.config.model_dump())
        population = create_initial_population(
            config=experiment_config,
            diversity_strength=1.0,
            raw_config=self.config.model_dump(),
            run_id=self.config.cache.run_id,
        )

        # Validate population
        self.services.population_manager.validate_population(population)

        self.logger.info(f"Population initialized with {population.size()} genomes")

        return population

    async def _run_generation(
        self, generation: int, population: Population
    ) -> Population:
        """Run a single generation of evolution.

        Args:
            generation: Current generation number (0-based)
            population: Current population

        Returns:
            Population for next generation
        """
        self.logger.info(f"Processing generation {generation}")

        # Update progress
        self.services.progress_tracker.update_status(
            "evolving",
            f"Generation {generation + 1}/{self.max_generations} - Evaluating population",
        )

        # Phase 1: Evaluate population fitness
        evaluated_population = await self._evaluate_population(population, generation)

        # Phase 2: Apply threshold gates
        filtered_population = self.services.population_manager.apply_threshold_gate(
            evaluated_population, generation
        )

        # Update progress tracking
        self.services.progress_tracker.update_generation_progress(
            generation, filtered_population
        )

        # Record generation statistics
        self.services.population_manager.record_generation_stats(filtered_population)

        # Phase 3: Selection and reproduction (if not last generation)
        if generation < self.max_generations - 1:
            next_population = await self._reproduce_population(
                filtered_population, generation
            )
        else:
            next_population = filtered_population

        # Process genetic tracking
        self.services.genetic_operations.process_genetic_tracking(generation)

        self.logger.info(
            f"Generation processing completed: generation={generation}, population_size={next_population.size()}"
        )

        return next_population

    async def _evaluate_population(
        self, population: Population, generation: int
    ) -> Population:
        """Evaluate population fitness using the fitness function and executor.

        Args:
            population: Population to evaluate

        Returns:
            Population with evaluated genomes
        """
        self.logger.info(
            f"Starting population evaluation for {population.size()} genomes"
        )

        # Find genomes that need evaluation
        unevaluated = [g for g in population.genomes if not g.is_evaluated()]
        evaluated = [g for g in population.genomes if g.is_evaluated()]

        if not unevaluated:
            self.logger.info("population_already_evaluated")
            return population

        self.logger.info(
            f"Evaluating genomes: {len(unevaluated)} unevaluated, {len(evaluated)} already evaluated"
        )

        # Use executor and fitness function to evaluate genomes
        # This is a simplified interface - the actual implementation would
        # handle the complexities of training and evaluation
        newly_evaluated = []

        for genome in unevaluated:
            try:
                # This would typically involve:
                # 1. Training the adapter if not cached
                # 2. Running inference/evaluation
                # 3. Calculating fitness scores

                # Get multi-objective evaluation
                (
                    fitness_scores,
                    multi_scores,
                    evaluation_metadata,
                ) = await self._evaluate_single_genome_with_scores(genome)

                # Create evaluated genome with both fitness and multi-objective scores
                evaluated_genome = genome.with_fitness(
                    fitness_scores
                ).with_multi_scores(multi_scores)
                if evaluation_metadata:
                    evaluated_genome = evaluated_genome.with_metadata(
                        {
                            **(evaluated_genome.metadata or {}),
                            "evaluation": evaluation_metadata,
                        }
                    )
                self._write_candidate_evaluation(generation, evaluated_genome)
                newly_evaluated.append(evaluated_genome)

            except Exception as e:
                self.logger.error(f"  Genome evaluation failed for {genome.id}: {e}")
                raise RuntimeError(
                    f"  Cannot continue evolution with failed genome evaluation. "
                    f"Genome: {genome.id}, Error: {e}"
                ) from e

        # Combine all genomes
        all_genomes = evaluated + newly_evaluated
        result_population = Population(tuple(all_genomes))

        self.logger.info(
            f"Population evaluation completed: {len(all_genomes)} genomes evaluated"
        )

        return result_population

    async def _evaluate_single_genome(self, genome: Genome) -> float:
        """Evaluate a single genome's fitness.

        Args:
            genome: Genome to evaluate

        Returns:
            Fitness score
        """
        (
            fitness,
            _multi_scores,
            _metadata,
        ) = await self._evaluate_single_genome_with_scores(genome)
        return fitness

    async def _evaluate_single_genome_with_scores(self, genome: Genome):
        """Evaluate a single genome and return both fitness and multi-objective scores.

        Args:
            genome: Genome to evaluate

        Returns:
            Tuple of (fitness_score, multi_objective_scores)
        """
        try:
            if not self.services.model_factory or not self.services.dataset_provider:
                raise EvolutionError("Dataset provider and model factory are required")

            model_runner: ModelRunner = self.services.model_factory(
                genome.lora_cfg, genome
            )
            problems = list(self.services.dataset_provider.problems())
            multi_scores = self.services.fitness_fn.evaluate_multi_objective(
                genome, model_runner, problems, genome.ca_features
            )
            evaluation_metadata = self._extract_evaluation_metadata(model_runner)

            # Get overall fitness from multi-objective scores
            fitness = multi_scores.overall_fitness()

            self.logger.debug(
                f"Genome evaluated with multi-scores: {genome.id}, fitness: {fitness:.4f}"
            )

            return fitness, multi_scores, evaluation_metadata

        except Exception as e:
            self.logger.error(
                f"  Single genome evaluation with scores failed for {genome.id}: {e}"
            )
            raise RuntimeError(
                f"  Cannot assign default scores when evaluation fails. "
                f"Genome: {genome.id}, Error: {e}"
            ) from e

    def _extract_evaluation_metadata(self, model_runner: ModelRunner) -> dict | None:
        """Extract optional plugin-specific metrics from a model runner."""
        metrics = getattr(model_runner, "last_metrics", None)
        if metrics is None:
            return None
        if hasattr(metrics, "to_report_dict"):
            return metrics.to_report_dict()
        if hasattr(metrics, "__dict__"):
            return dict(metrics.__dict__)
        return {"value": str(metrics)}

    def _write_candidate_evaluation(self, generation: int, genome: Genome) -> None:
        """Append a machine-readable candidate evaluation row."""
        try:
            output_path = (
                self.config.execution.output_dir / "candidate_evaluations.jsonl"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "generation": generation,
                "genome_id": genome.id,
                "run_id": genome.run_id,
                "fitness": genome.fitness,
                "multi_scores": (
                    genome.multi_scores.to_dict() if genome.multi_scores else None
                ),
                "lora": {
                    "r": genome.lora_cfg.r,
                    "alpha": genome.lora_cfg.alpha,
                    "dropout": genome.lora_cfg.dropout,
                    "target_modules": list(genome.lora_cfg.target_modules),
                    "adapter_type": genome.lora_cfg.adapter_type,
                },
                "ca": {
                    "rule": genome.seed.rule,
                    "steps": genome.seed.steps,
                    "grid_shape": list(genome.seed.grid.shape),
                    "grid_hash": stable_digest(genome.seed.grid, length=12),
                },
                "metadata": genome.metadata or {},
            }
            with output_path.open("a") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            self.logger.warning(
                f"Candidate evaluation logging failed for {genome.id}: {exc}"
            )

    async def _reproduce_population(
        self, population: Population, generation: int
    ) -> Population:
        """Reproduce population for next generation.

        Args:
            population: Current population
            generation: Current generation number

        Returns:
            Next generation population
        """
        self.logger.info(f"Starting reproduction for generation {generation}")

        # Update progress
        self.services.progress_tracker.update_status(
            "evolving",
            f"Generation {generation + 1}/{self.max_generations} - Reproducing population",
        )

        # Adjust genetic parameters based on population diversity
        self.services.genetic_operations.adjust_genetic_parameters(
            population, generation
        )

        # Select survivors
        survivors = self.services.population_manager.select_survivors(population)

        # Reproduce to target population size
        next_population = self.services.genetic_operations.reproduce_population(
            survivors, self.config.execution.population_size, generation
        )

        # Validate the new population
        self.services.population_manager.validate_population(next_population)

        self.logger.info(
            f"Reproduction completed: {next_population.size()} genomes for generation {generation + 1}"
        )

        return next_population

    def _should_stop_early(self, population: Population, generation: int) -> bool:
        """Determine if evolution should stop early.

        Args:
            population: Current population
            generation: Current generation number

        Returns:
            True if evolution should stop early
        """
        return self.services.population_manager.should_stop_early(
            population, generation
        )

    def _create_result(self, population: Population, status: str) -> EvolutionResult:
        """Create evolution result.

        Args:
            population: Final population
            status: Evolution status

        Returns:
            EvolutionResult with final state
        """
        # Find best genome
        best_genome = None
        try:
            if population.size() > 0:
                best_genome = population.best()
        except Exception:
            pass

        # Calculate total time
        total_time = time.time() - self.start_time if self.start_time else 0

        # Update final progress
        if best_genome:
            self.services.progress_tracker.update_status(
                status,
                f"Evolution {status} - Best fitness: {best_genome.fitness:.3f}",
                best_genome,
            )
        else:
            self.services.progress_tracker.update_status(status, f"Evolution {status}")

        return EvolutionResult(
            final_population=population,
            best_genome=best_genome,
            generations_completed=self.current_generation + 1,
            total_time=total_time,
            status=status,
        )
