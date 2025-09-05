"""Population management service for CORAL-X evolution."""

from typing import Dict, Optional
from random import Random

from ..domain.neat import Population, select, tournament_select
from .pareto.selection import nsga2_select
from ..domain.threshold_gate import calculate_dynamic_thresholds, filter_population_by_thresholds
from ..common.config import CoralConfig
from ..common.logging import LoggingMixin
from ..common.exceptions import EvolutionError


class PopulationManager(LoggingMixin):
    """Manages population lifecycle and diversity."""

    def __init__(self, config: CoralConfig, random_seed: Optional[int] = None):
        super().__init__()
        self.config = config
        self.execution_config = config.execution
        self.threshold_config = config.threshold

        # Set up random generator
        self.random = Random(random_seed or config.seed)

        # Population state
        self.current_generation = 0
        self.generation_history = {
            'best_fitness': [],
            'diversity_scores': [],
            'population_sizes': []
        }

        self.logger.info(f"Population manager initialized: size={self.execution_config.population_size}, mode={self.execution_config.selection_mode}, survival_rate={self.execution_config.survival_rate}")

    def validate_population(self, population: Population) -> None:
        """Validate population meets requirements.
        
        Args:
            population: Population to validate
            
        Raises:
            EvolutionError: If population is invalid
        """
        if population.size() == 0:
            raise EvolutionError("Population is empty")

        if population.size() > self.execution_config.population_size * 2:
            self.logger.warning(f"Population size exceeded: current_size={population.size()}, max_size={self.execution_config.population_size * 2}")

        # Check for duplicate genomes
        genome_ids = [g.id for g in population.genomes]
        if len(genome_ids) != len(set(genome_ids)):
            raise EvolutionError("Population contains duplicate genome IDs")

    def apply_threshold_gate(self, population: Population, generation: int) -> Population:
        """Apply threshold gate filtering to population.
        
        Args:
            population: Population to filter
            generation: Current generation number
            
        Returns:
            Filtered population that meets thresholds
        """
        self.logger.info(f"Applying threshold gate: generation={generation}, population_size={population.size()}")

        # Calculate dynamic thresholds for current generation
        dynamic_thresholds = calculate_dynamic_thresholds(
            generation,
            self.execution_config.generations,
            self.threshold_config
        )

        # Apply filtering
        def score_extractor(genome):
            """Extract multi-objective scores from genome - NO FALLBACKS."""
            if genome.has_multi_scores():
                return genome.multi_scores
            else:
                raise RuntimeError(
                    f"  Genome {genome.id} lacks multi-objective scores. "
                    f"Cannot create default scores from potentially invalid fitness. "
                    f"All genomes must have proper multi-objective evaluation."
                )

        filtered_genomes = filter_population_by_thresholds(
            list(population.genomes),
            score_extractor,
            dynamic_thresholds
        )

        from ..domain.neat import Population
        filtered_population = Population(tuple(filtered_genomes))

        passed_count = filtered_population.size()
        filtered_count = population.size() - passed_count

        self.logger.info(f"Threshold gate applied: generation={generation}, original_size={population.size()}, passed_count={passed_count}, filtered_count={filtered_count}, pass_rate={passed_count / population.size() if population.size() > 0 else 0}")

        # Ensure minimum population size
        if passed_count < 2:
            self.logger.warning(f"Population too small after filtering: remaining_genomes={passed_count}, generation={generation}")

            # Keep top genomes if threshold gate is too strict
            if population.size() >= 2:
                self.logger.info("Keeping top genomes due to strict thresholds")
                sorted_genomes = sorted(population.genomes, key=lambda g: g.fitness, reverse=True)
                return Population(tuple(sorted_genomes[:max(2, passed_count)]))

        return filtered_population

    def select_survivors(self, population: Population) -> Population:
        """Select survivors for next generation.
        
        Args:
            population: Population to select from
            
        Returns:
            Population containing selected survivors
        """
        if population.size() == 0:
            raise EvolutionError("Cannot select survivors from empty population")

        target_size = self.execution_config.population_size
        survival_rate = self.execution_config.survival_rate
        num_survivors = max(1, int(target_size * survival_rate))

        self.logger.info(f"Selecting survivors: population_size={population.size()}, survival_rate={survival_rate}, num_survivors={num_survivors}, selection_mode={self.execution_config.selection_mode}")

        # Select based on configured method
        if self.execution_config.selection_mode.value == "pareto":
            # Use NSGA-II for multi-objective selection
            survivors = nsga2_select(population, num_survivors)
        elif self.execution_config.selection_mode.value == "tournament":
            # Use tournament selection for M1
            survivors = tournament_select(population, num_survivors, tournament_size=3, rng=self.random)
        else:
            # Use fitness-based selection
            survivors = select(population, num_survivors)

        self.validate_population(survivors)

        self.logger.info(f"Survivors selected: selected_count={survivors.size()}, selection_mode={self.execution_config.selection_mode}")

        return survivors

    def calculate_diversity_metrics(self, population: Population) -> Dict[str, float]:
        """Calculate diversity metrics for the population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Dictionary containing diversity metrics
        """
        if population.size() <= 1:
            return {
                'fitness_diversity': 0.0,
                'genetic_diversity': 0.0,
                'phenotype_diversity': 0.0
            }

        genomes = population.genomes

        # Fitness diversity (standard deviation of fitness scores)
        fitness_scores = [g.fitness for g in genomes if g.is_evaluated()]
        if len(fitness_scores) > 1:
            mean_fitness = sum(fitness_scores) / len(fitness_scores)
            fitness_variance = sum((f - mean_fitness) ** 2 for f in fitness_scores) / len(fitness_scores)
            fitness_diversity = fitness_variance ** 0.5
        else:
            fitness_diversity = 0.0

        # Genetic diversity (based on LoRA configurations)
        unique_configs = set()
        for genome in genomes:
            if hasattr(genome, 'lora_cfg') and genome.lora_cfg:
                config_signature = (
                    genome.lora_cfg.r,
                    genome.lora_cfg.alpha,
                    tuple(genome.lora_cfg.target_modules),
                    genome.lora_cfg.dropout
                )
                unique_configs.add(config_signature)

        genetic_diversity = len(unique_configs) / len(genomes) if genomes else 0.0

        # Phenotype diversity (based on multi-objective scores)
        phenotype_vectors = []
        for genome in genomes:
            if genome.has_multi_scores():
                scores = genome.multi_scores
                vector = (scores.bugfix, scores.style, scores.security, scores.runtime, scores.syntax)
                phenotype_vectors.append(vector)

        if len(phenotype_vectors) > 1:
            # Calculate average pairwise distance
            total_distance = 0
            pair_count = 0
            for i in range(len(phenotype_vectors)):
                for j in range(i + 1, len(phenotype_vectors)):
                    dist = sum((a - b) ** 2 for a, b in zip(phenotype_vectors[i], phenotype_vectors[j])) ** 0.5
                    total_distance += dist
                    pair_count += 1
            phenotype_diversity = total_distance / pair_count if pair_count > 0 else 0.0
        else:
            phenotype_diversity = 0.0

        diversity_metrics = {
            'fitness_diversity': fitness_diversity,
            'genetic_diversity': genetic_diversity,
            'phenotype_diversity': phenotype_diversity
        }

        self.logger.debug(f"Diversity metrics calculated: {diversity_metrics}")

        return diversity_metrics

    def should_stop_early(self, population: Population, generation: int,
                         plateau_window: int = 5, min_improvement: float = 0.001) -> bool:
        """Determine if evolution should stop early.
        
        Args:
            population: Current population
            generation: Current generation number
            plateau_window: Number of generations to check for plateau
            min_improvement: Minimum improvement required to continue
            
        Returns:
            True if evolution should stop early
        """
        if generation < plateau_window:
            return False

        # Get recent fitness history
        if len(self.generation_history['best_fitness']) < plateau_window:
            return False

        recent_fitness = self.generation_history['best_fitness'][-plateau_window:]

        # Check for fitness plateau
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        is_plateau = fitness_improvement < min_improvement

        # Check for diversity loss
        recent_diversity = self.generation_history['diversity_scores'][-plateau_window:]
        diversity_decline = len([d for d in recent_diversity if d < 0.1]) >= plateau_window // 2

        should_stop = is_plateau and diversity_decline

        if should_stop:
            self.logger.info(f"Early stopping triggered: generation={generation}, fitness_improvement={fitness_improvement}, min_improvement={min_improvement}, diversity_decline={diversity_decline}")

        return should_stop

    def record_generation_stats(self, population: Population) -> None:
        """Record statistics for the current generation.
        
        Args:
            population: Current population
        """
        # Record best fitness
        best_fitness = 0.0
        if population.size() > 0:
            try:
                best_genome = population.best()
                best_fitness = best_genome.fitness
            except:
                pass

        self.generation_history['best_fitness'].append(best_fitness)

        # Record diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(population)
        avg_diversity = sum(diversity_metrics.values()) / len(diversity_metrics)
        self.generation_history['diversity_scores'].append(avg_diversity)

        # Record population size
        self.generation_history['population_sizes'].append(population.size())

        self.current_generation += 1

        self.logger.info(f"Generation stats recorded: generation={self.current_generation}, population_size={population.size()}, best_fitness={best_fitness}, avg_diversity={avg_diversity}")
