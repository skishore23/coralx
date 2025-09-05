"""Genetic operations service for CORAL-X evolution."""

from typing import Dict, Any, Optional
from random import Random
import time

from ..domain.neat import Population, mutate, crossover
from ..domain.genome import Genome
from .genetic_operations_tracker import GeneticOperationsTracker
from ..common.config import CoralConfig
from ..common.logging import LoggingMixin
from ..common.exceptions import GeneticOperationError


class GeneticOperationsService(LoggingMixin):
    """Handles crossover, mutation, and genetic diversity operations."""

    def __init__(self, config: CoralConfig, random_seed: Optional[int] = None):
        super().__init__()
        self.config = config
        self.execution_config = config.execution

        # Set up random generator
        self.random = Random(random_seed or config.seed)

        # Genetic parameters
        self.crossover_rate = self.execution_config.crossover_rate
        self.population_size = self.execution_config.population_size

        # Initialize genetic operations tracker
        genetic_tracking_dir = self.config.execution.output_dir / "genetic_tracking"
        self.genetic_tracker = GeneticOperationsTracker(output_dir=str(genetic_tracking_dir))

        # Statistics tracking
        self.generation_stats = {
            'crossovers_performed': [],
            'mutations_performed': [],
            'diversity_strengths': [],
            'operation_times': []
        }

        self.logger.info(f"Genetic operations service initialized: crossover_rate={self.crossover_rate}, population_size={self.population_size}")

    def reproduce_population(self, survivors: Population, target_size: int,
                           generation: int) -> Population:
        """Reproduce population to target size using genetic operations.
        
        Args:
            survivors: Population of survivor genomes
            target_size: Target population size
            generation: Current generation number
            
        Returns:
            New population with target size
        """
        start_time = time.time()

        if survivors.size() == 0:
            raise GeneticOperationError("Cannot reproduce from empty survivor population")

        self.logger.info(f"Reproduction started: generation={generation}, survivors_count={survivors.size()}, target_size={target_size}, crossover_rate={self.crossover_rate}")

        new_genomes = list(survivors.genomes)
        crossovers_performed = 0
        mutations_performed = 0

        # Calculate how many offspring to generate
        offspring_needed = target_size - survivors.size()

        if offspring_needed <= 0:
            self.logger.info(f"No offspring needed: target_size={target_size}, survivors={survivors.size()}")
            return survivors

        # Generate offspring through crossover and mutation
        while len(new_genomes) < target_size:
            if self.random.random() < self.crossover_rate and survivors.size() >= 2:
                # Perform crossover
                offspring = self._perform_crossover(survivors, generation)
                if offspring:
                    new_genomes.append(offspring)
                    crossovers_performed += 1
                    self.logger.debug(f"Crossover performed: generation={generation}, parent_count=2, offspring_id={offspring.id}")
            else:
                # Perform mutation
                offspring = self._perform_mutation(survivors, generation)
                if offspring:
                    new_genomes.append(offspring)
                    mutations_performed += 1
                    self.logger.debug(f"Mutation performed: generation={generation}, offspring_id={offspring.id}")

        # Trim to exact target size if we overshot
        if len(new_genomes) > target_size:
            new_genomes = new_genomes[:target_size]

        # Record statistics
        operation_time = time.time() - start_time
        self._record_generation_stats(generation, crossovers_performed,
                                    mutations_performed, operation_time)

        result_population = Population(tuple(new_genomes))

        self.logger.info(f"Reproduction completed: generation={generation}, final_size={result_population.size()}, crossovers={crossovers_performed}, mutations={mutations_performed}, operation_time={operation_time}")

        return result_population

    def _perform_crossover(self, population: Population, generation: int) -> Optional[Genome]:
        """Perform crossover between two randomly selected parents.
        
        Args:
            population: Population to select parents from
            generation: Current generation number
            
        Returns:
            Offspring genome or None if crossover failed
        """
        try:
            # Select two different parents
            parents = self.random.sample(list(population.genomes), 2)
            parent1, parent2 = parents[0], parents[1]

            # Calculate diversity strength for the population
            diversity_strength = self.calculate_diversity_strength(population)

            # Get evolution config and other required parameters
            evo_cfg = self.config.evo
            config_dict = self.config.model_dump()
            run_id = getattr(self.config, 'run_id', None)

            # Perform crossover using domain function with all required parameters
            offspring = crossover(
                parent1, parent2,
                evo_cfg=evo_cfg,
                rng=self.random,
                generation=generation,
                diversity_strength=diversity_strength,
                config_dict=config_dict,
                run_id=run_id
            )

            # Track the crossover operation
            self.genetic_tracker.track_crossover(
                child=offspring,
                parent1=parent1,
                parent2=parent2,
                generation=generation,
                diversity_strength=diversity_strength
            )

            return offspring

        except Exception as e:
            self.logger.error(f"Crossover failed: generation={generation}, error={str(e)}")
            return None

    def _perform_mutation(self, population: Population, generation: int) -> Optional[Genome]:
        """Perform mutation on a randomly selected parent.
        
        Args:
            population: Population to select parent from
            generation: Current generation number
            
        Returns:
            Mutated genome or None if mutation failed
        """
        try:
            # Select random parent
            parent = self.random.choice(list(population.genomes))

            # Calculate diversity strength for the population
            diversity_strength = self.calculate_diversity_strength(population)

            # Get evolution config and other required parameters
            evo_cfg = self.config.evo
            config_dict = self.config.model_dump()
            run_id = getattr(self.config, 'run_id', None)

            # Perform mutation using domain function with all required parameters
            offspring = mutate(
                parent,
                evo_cfg=evo_cfg,
                rng=self.random,
                generation=generation,
                diversity_strength=diversity_strength,
                config_dict=config_dict,
                run_id=run_id
            )

            # Determine mutation type based on whether CA or LoRA was mutated
            mutation_type = "ca_mutation"  # Default assumption - could be enhanced to detect actual type

            # Track the mutation operation
            self.genetic_tracker.track_mutation(
                child=offspring,
                parent=parent,
                generation=generation,
                mutation_type=mutation_type,
                diversity_strength=diversity_strength
            )

            return offspring

        except Exception as e:
            self.logger.error(f"Mutation failed: generation={generation}, error={str(e)}")
            return None

    def calculate_diversity_strength(self, population: Population) -> float:
        """Calculate diversity strength for adaptive genetic operations.
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity strength value (0.0 to 2.0)
        """
        if population.size() <= 1:
            return 1.0  # Default diversity strength

        genomes = population.genomes

        # Calculate fitness diversity
        fitness_scores = [g.fitness for g in genomes if g.is_evaluated()]
        if len(fitness_scores) <= 1:
            fitness_diversity = 0.0
        else:
            mean_fitness = sum(fitness_scores) / len(fitness_scores)
            variance = sum((f - mean_fitness) ** 2 for f in fitness_scores) / len(fitness_scores)
            fitness_diversity = variance ** 0.5

        # Calculate genetic diversity (LoRA parameter diversity)
        unique_ranks = set()
        unique_alphas = set()
        unique_dropouts = set()

        for genome in genomes:
            if hasattr(genome, 'lora_cfg') and genome.lora_cfg:
                unique_ranks.add(genome.lora_cfg.r)
                unique_alphas.add(genome.lora_cfg.alpha)
                unique_dropouts.add(genome.lora_cfg.dropout)

        genetic_diversity = (
            len(unique_ranks) / max(1, len(genomes)) +
            len(unique_alphas) / max(1, len(genomes)) +
            len(unique_dropouts) / max(1, len(genomes))
        ) / 3.0

        # Combine diversity measures
        diversity_strength = (fitness_diversity * 0.6 + genetic_diversity * 0.4) * 2.0
        diversity_strength = max(0.5, min(2.0, diversity_strength))  # Clamp to reasonable range

        self.logger.debug(f"Diversity strength calculated: fitness_diversity={fitness_diversity}, genetic_diversity={genetic_diversity}, diversity_strength={diversity_strength}")

        return diversity_strength

    def adjust_genetic_parameters(self, population: Population, generation: int) -> None:
        """Adjust genetic parameters based on population diversity.
        
        Args:
            population: Current population
            generation: Current generation number
        """
        diversity_strength = self.calculate_diversity_strength(population)

        # Adjust crossover rate based on diversity
        # Low diversity -> increase crossover (more exploration)
        # High diversity -> maintain or decrease crossover
        base_crossover = self.execution_config.crossover_rate
        if diversity_strength < 0.8:
            adjusted_crossover = min(0.9, base_crossover * 1.2)
            self.logger.info(f"Increasing crossover rate due to low diversity: generation={generation}, old_rate={base_crossover}, new_rate={adjusted_crossover}")
        elif diversity_strength > 1.5:
            adjusted_crossover = max(0.3, base_crossover * 0.8)
            self.logger.info(f"Decreasing crossover rate due to high diversity: generation={generation}, old_rate={base_crossover}, new_rate={adjusted_crossover}")
        else:
            adjusted_crossover = base_crossover

        self.crossover_rate = adjusted_crossover
        self.generation_stats['diversity_strengths'].append(diversity_strength)

    def _record_generation_stats(self, generation: int, crossovers: int,
                               mutations: int, operation_time: float) -> None:
        """Record statistics for the generation.
        
        Args:
            generation: Generation number
            crossovers: Number of crossovers performed
            mutations: Number of mutations performed
            operation_time: Time taken for operations
        """
        self.generation_stats['crossovers_performed'].append(crossovers)
        self.generation_stats['mutations_performed'].append(mutations)
        self.generation_stats['operation_times'].append(operation_time)

        self.logger.info(f"Generation genetic stats: generation={generation}, crossovers={crossovers}, mutations={mutations}, operation_time={operation_time}, crossover_rate={self.crossover_rate}")

    def process_genetic_tracking(self, generation: int) -> None:
        """Process and save genetic tracking data for the generation.
        
        Args:
            generation: Generation number to process
        """
        try:
            self.genetic_tracker.save_tracking_data(generation)
            self.genetic_tracker.detect_genetic_patterns(generation)

            self.logger.info(f"Genetic tracking processed for generation {generation}")

        except Exception as e:
            self.logger.error(f"Genetic tracking failed: generation={generation}, error={str(e)}")

    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of genetic operations across all generations.
        
        Returns:
            Dictionary with genetic operations summary
        """
        if not self.generation_stats['crossovers_performed']:
            return {'message': 'No genetic operations performed yet'}

        total_crossovers = sum(self.generation_stats['crossovers_performed'])
        total_mutations = sum(self.generation_stats['mutations_performed'])
        avg_operation_time = sum(self.generation_stats['operation_times']) / len(self.generation_stats['operation_times'])
        avg_diversity_strength = sum(self.generation_stats['diversity_strengths']) / len(self.generation_stats['diversity_strengths']) if self.generation_stats['diversity_strengths'] else 1.0

        return {
            'total_crossovers': total_crossovers,
            'total_mutations': total_mutations,
            'avg_operation_time': avg_operation_time,
            'avg_diversity_strength': avg_diversity_strength,
            'generations_processed': len(self.generation_stats['crossovers_performed'])
        }
