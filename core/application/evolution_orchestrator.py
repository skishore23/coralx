"""Evolution orchestrator for CORAL-X using service composition."""

from typing import Optional
from dataclasses import dataclass
import time

from ..domain.neat import Population
from ..domain.genome import Genome
from ..services.population_manager import PopulationManager
from ..services.genetic_operations import GeneticOperationsService
from ..services.progress_tracker import ProgressTracker
from ..common.config import CoralConfig
from ..common.logging import LoggingMixin
from ..common.exceptions import EvolutionError
from ..ports.interfaces import FitnessFn, Executor


@dataclass
class EvolutionResult:
    """Result of evolution process."""
    final_population: Population
    best_genome: Optional[Genome]
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

        self.logger.info(f"Evolution orchestrator initialized: {self.config.experiment.name}, max_generations={self.max_generations}, population_size={self.config.execution.population_size}, executor={self.config.infra.executor}")

    async def run_evolution(self) -> EvolutionResult:
        """Main evolution loop - focused only on orchestration.
        
        Returns:
            EvolutionResult with final state
        """
        self.start_time = time.time()

        try:
            self._validate_preconditions()

            self.logger.info("Evolution started")
            self.services.progress_tracker.update_status('starting', 'Evolution starting...')

            # Initialize population
            population = await self._initialize_population()

            # Main evolution loop
            for generation in range(self.max_generations):
                self.current_generation = generation

                self.logger.info(f"Generation {generation + 1}/{self.max_generations} started")

                population = await self._run_generation(generation, population)

                # Check early stopping conditions
                if self._should_stop_early(population, generation):
                    self.logger.info(f"Early stopping triggered at generation {generation}")
                    break

            # Evolution completed
            result = self._create_result(population, 'completed')

            self.logger.info(f"Evolution completed: {result.generations_completed} generations in {result.total_time:.2f}s, best fitness: {result.best_genome.fitness if result.best_genome else 0.0:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
            self.services.progress_tracker.update_status('failed', f'Evolution failed: {e}')

            # Create failure result
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            return EvolutionResult(
                final_population=Population(tuple()),
                best_genome=None,
                generations_completed=self.current_generation,
                total_time=elapsed_time,
                status='failed'
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

        self.logger.info("Preconditions validated")

    async def _initialize_population(self) -> Population:
        """Initialize the starting population.
        
        Returns:
            Initial population
        """
        self.logger.info(f"Initializing population of size {self.config.execution.population_size}")

        # Create initial population using domain function
        # Convert full config to experiment config format
        from core.domain.experiment import create_experiment_config, create_initial_population

        experiment_config = create_experiment_config(self.config.model_dump())
        population = create_initial_population(
            config=experiment_config,
            diversity_strength=1.0,
            raw_config=self.config.model_dump(),
            run_id=getattr(self.config, 'run_id', None)
        )

        # Validate population
        self.services.population_manager.validate_population(population)

        self.logger.info(f"Population initialized with {population.size()} genomes")

        return population

    async def _run_generation(self, generation: int, population: Population) -> Population:
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
            'evolving',
            f'Generation {generation + 1}/{self.max_generations} - Evaluating population'
        )

        # Phase 1: Evaluate population fitness
        evaluated_population = await self._evaluate_population(population)

        # Phase 2: Apply threshold gates
        filtered_population = self.services.population_manager.apply_threshold_gate(
            evaluated_population, generation
        )

        # Update progress tracking
        self.services.progress_tracker.update_generation_progress(generation, filtered_population)

        # Record generation statistics
        self.services.population_manager.record_generation_stats(filtered_population)

        # Phase 3: Selection and reproduction (if not last generation)
        if generation < self.max_generations - 1:
            next_population = await self._reproduce_population(filtered_population, generation)
        else:
            next_population = filtered_population

        # Process genetic tracking
        self.services.genetic_operations.process_genetic_tracking(generation)

        self.logger.info(f"Generation processing completed: generation={generation}, population_size={next_population.size()}")

        return next_population

    async def _evaluate_population(self, population: Population) -> Population:
        """Evaluate population fitness using the fitness function and executor.
        
        Args:
            population: Population to evaluate
            
        Returns:
            Population with evaluated genomes
        """
        self.logger.info(f"Starting population evaluation for {population.size()} genomes")

        # Find genomes that need evaluation
        unevaluated = [g for g in population.genomes if not g.is_evaluated()]
        evaluated = [g for g in population.genomes if g.is_evaluated()]

        if not unevaluated:
            self.logger.info("population_already_evaluated")
            return population

        self.logger.info(f"Evaluating genomes: {len(unevaluated)} unevaluated, {len(evaluated)} already evaluated")

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
                fitness_scores, multi_scores = await self._evaluate_single_genome_with_scores(genome)

                # Create evaluated genome with both fitness and multi-objective scores
                evaluated_genome = genome.with_fitness(fitness_scores).with_multi_scores(multi_scores)
                newly_evaluated.append(evaluated_genome)

            except Exception as e:
                self.logger.error(f"  Genome evaluation failed for {genome.id}: {e}")
                raise RuntimeError(
                    f"  Cannot continue evolution with failed genome evaluation. "
                    f"Genome: {genome.id}, Error: {e}"
                )

        # Combine all genomes
        all_genomes = evaluated + newly_evaluated
        result_population = Population(tuple(all_genomes))

        self.logger.info(f"Population evaluation completed: {len(all_genomes)} genomes evaluated")

        return result_population

    async def _evaluate_single_genome(self, genome: Genome) -> float:
        """Evaluate a single genome's fitness.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Fitness score
        """
        # This is a placeholder - actual implementation would use
        # the fitness function and executor services
        try:
            # Create model runner for this genome
            # For now, we'll use a placeholder since we need dataset and model integration
            from plugins.fakenews_tinyllama.plugin import MultiModalAISafetyPlugin

            # Get plugin configuration from config
            plugin_config = {
                'dataset': {
                    'dataset_path': './datasets',
                    'max_samples': 20,  # Reasonable for testing
                    'datasets': ['fake_news']  # Will fall back to synthetic if no Kaggle credentials
                },
                'model': {
                    'model_name': self.config.experiment.model.name,
                    'max_seq_length': 2048,
                    'simulation_mode': True  # Enable simulation mode for local testing
                },
                'evaluation': {
                    'test_samples': 5  # Smaller for testing
                },
                'training': {
                    # No simulation - real training only
                }
            }

            # Create plugin instance
            plugin = MultiModalAISafetyPlugin(plugin_config)

            # Create model runner and dataset
            model_factory = plugin.model_factory()
            dataset_provider = plugin.dataset()

            # Create model with genome's LoRA config
            model_runner = model_factory(genome.lora_cfg, genome)

            # Get problems from dataset
            problems = list(dataset_provider.problems())

            # Get the fitness function and call it directly to access multi-objective scores
            fitness_fn = self.services.fitness_fn
            multi_scores = fitness_fn.evaluate_multi_objective(genome, model_runner, problems)

            # Get overall fitness from multi-objective scores
            fitness = multi_scores.overall_fitness()

            self.logger.debug(f"Genome evaluated: {genome.id}, fitness: {fitness:.4f}")

            return fitness

        except Exception as e:
            self.logger.error(f"  Single genome evaluation failed for {genome.id}: {e}")
            raise RuntimeError(
                f"  Evaluation failure prevents evolution continuation. "
                f"Genome: {genome.id}, Error: {e}"
            )

    async def _evaluate_single_genome_with_scores(self, genome: Genome):
        """Evaluate a single genome and return both fitness and multi-objective scores.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Tuple of (fitness_score, multi_objective_scores)
        """
        try:
            # Create model runner for this genome
            from plugins.fakenews_tinyllama.plugin import MultiModalAISafetyPlugin

            # Get plugin configuration from config
            plugin_config = {
                'dataset': {
                    'dataset_path': './datasets',
                    'max_samples': 20,  # Reasonable for testing
                    'datasets': ['fake_news']  # Will fall back to synthetic if no Kaggle credentials
                },
                'model': {
                    'model_name': self.config.experiment.model.name,
                    'max_seq_length': 2048,
                    'simulation_mode': True  # Enable simulation mode for local testing
                },
                'evaluation': {
                    'test_samples': 5  # Smaller for testing
                },
                'training': {
                    # No simulation - real training only
                }
            }

            # Create plugin instance
            plugin = MultiModalAISafetyPlugin(plugin_config)

            # Create model runner and dataset
            model_factory = plugin.model_factory()
            dataset_provider = plugin.dataset()

            # Create model with genome's LoRA config
            model_runner = model_factory(genome.lora_cfg, genome)

            # Get problems from dataset
            problems = list(dataset_provider.problems())

            # Get the fitness function and call evaluate_multi_objective directly
            fitness_fn = self.services.fitness_fn
            multi_scores = fitness_fn.evaluate_multi_objective(genome, model_runner, problems)

            # Get overall fitness from multi-objective scores
            fitness = multi_scores.overall_fitness()

            self.logger.debug(f"Genome evaluated with multi-scores: {genome.id}, fitness: {fitness:.4f}")

            return fitness, multi_scores

        except Exception as e:
            self.logger.error(f"  Single genome evaluation with scores failed for {genome.id}: {e}")
            raise RuntimeError(
                f"  Cannot assign default scores when evaluation fails. "
                f"Genome: {genome.id}, Error: {e}"
            )

    async def _reproduce_population(self, population: Population,
                                  generation: int) -> Population:
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
            'evolving',
            f'Generation {generation + 1}/{self.max_generations} - Reproducing population'
        )

        # Adjust genetic parameters based on population diversity
        self.services.genetic_operations.adjust_genetic_parameters(population, generation)

        # Select survivors
        survivors = self.services.population_manager.select_survivors(population)

        # Reproduce to target population size
        next_population = self.services.genetic_operations.reproduce_population(
            survivors,
            self.config.execution.population_size,
            generation
        )

        # Validate the new population
        self.services.population_manager.validate_population(next_population)

        self.logger.info(f"Reproduction completed: {next_population.size()} genomes for generation {generation + 1}")

        return next_population

    def _should_stop_early(self, population: Population, generation: int) -> bool:
        """Determine if evolution should stop early.
        
        Args:
            population: Current population
            generation: Current generation number
            
        Returns:
            True if evolution should stop early
        """
        return self.services.population_manager.should_stop_early(population, generation)

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
        except:
            pass

        # Calculate total time
        total_time = time.time() - self.start_time if self.start_time else 0

        # Update final progress
        if best_genome:
            self.services.progress_tracker.update_status(
                status,
                f'Evolution {status} - Best fitness: {best_genome.fitness:.3f}',
                best_genome
            )
        else:
            self.services.progress_tracker.update_status(
                status,
                f'Evolution {status}'
            )

        return EvolutionResult(
            final_population=population,
            best_genome=best_genome,
            generations_completed=self.current_generation + 1,
            total_time=total_time,
            status=status
        )
