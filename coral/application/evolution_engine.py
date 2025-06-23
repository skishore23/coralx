###############################################################################
# Orchestration - NO FALLBACKS, strict config-driven
###############################################################################
from dataclasses import dataclass
from typing import Dict, Any, Callable, List
from random import Random, choice
from concurrent.futures import Future
import time
from pathlib import Path

from coral.domain.experiment import create_initial_population

from ..domain.ca import evolve
from ..domain.feature_extraction import extract_features
from ..domain.mapping import map_features_to_lora_config, EvolutionConfig
from ..domain.genome import Genome, MultiObjectiveScores
from ..domain.neat import Population, select, mutate, crossover
from ..domain.threshold_gate import (
    ThresholdConfig, calculate_dynamic_thresholds, 
    filter_population_by_thresholds
)
from ..domain.pareto_selection import nsga2_select
from ..ports.interfaces import FitnessFn, Executor, ModelRunner, DatasetProvider
from ..domain.genetic_operations_tracker import GeneticOperationsTracker


@dataclass(frozen=True)
class CoralConfig:
    """Main configuration container - NO DEFAULTS."""
    evo: EvolutionConfig
    threshold: ThresholdConfig
    seed: int
    execution: Dict[str, Any]
    infra: Dict[str, Any]
    experiment: Dict[str, Any]
    cache: Dict[str, Any]
    evaluation: Dict[str, Any]  # Include evaluation configuration (adaptive testing, fitness weights)
    adapter_type: str = "lora"  # Adapter type: "lora" or "dora"


class EvolutionEngine:
    def __init__(self,
                 cfg: CoralConfig,
                 fitness_fn: FitnessFn,
                 executor: Executor,
                 model_factory: Callable,
                 dataset: DatasetProvider,
                 run_id: str = None):
        self.cfg = cfg
        self.fitness_fn = fitness_fn
        self.executor = executor
        self.model_factory = model_factory
        self.dataset = dataset
        self.run_id = run_id  # Store run_id for genome creation
        
        # Validate configuration
        self._validate_configuration()
        
        # Generation tracking for threshold gates
        self.current_generation = 0
        self.max_generations = self.cfg.execution['generations']
        
        # Dynamic diversity tracking
        self.generation_history = {
            'best_fitness': [],
            'cache_hit_rates': [],
            'diversity_strengths': []
        }
        
        # Initialize genetic operations tracker
        genetic_tracking_dir = self.cfg.execution.get('genetic_tracking_dir', 'results/genetic_tracking')
        self.genetic_tracker = GeneticOperationsTracker(output_dir=genetic_tracking_dir)
    
    def _validate_configuration(self):
        """Validate that all required configuration is present - FAIL-FAST."""
        if 'generations' not in self.cfg.execution:
            raise ValueError("FAIL-FAST: 'generations' missing from execution configuration")
        
        if 'population_size' not in self.cfg.execution:
            raise ValueError("FAIL-FAST: 'population_size' missing from execution configuration")
        
        if self.cfg.execution['generations'] <= 0:
            raise ValueError("FAIL-FAST: 'generations' must be positive")
        
        if self.cfg.execution['population_size'] <= 0:
            raise ValueError("FAIL-FAST: 'population_size' must be positive")
    
    def run(self, init_pop: Population) -> Population:
        """Main evolution loop with CORAL-X enhancements - FAIL-FAST."""
        pop = init_pop
        
        print(f"🚀 Starting CORAL-X Evolution")
        print(f"   📊 Population: {pop.size()} genomes")
        print(f"   🔄 Generations: {self.max_generations}")
        print(f"   🧬 Evolution Path: CA → Features → LoRA (Full CORAL-X)")
        print("=" * 60)
        
        for gen in range(self.max_generations):
            self.current_generation = gen
            print(f"\n🧬 GENERATION {gen + 1}/{self.max_generations}")
            print("=" * 50)
            
            # Show population overview
            evaluated_count = len([g for g in pop.genomes if g.is_evaluated()])
            print(f"📋 Population Overview:")
            print(f"   • Total genomes: {pop.size()}")
            print(f"   • Already evaluated: {evaluated_count}")
            print(f"   • Need evaluation: {pop.size() - evaluated_count}")
            
            # Evaluate population fitness
            pop = self._evaluate_population(pop)
            
            # Apply threshold gate filtering
            pop = self._apply_threshold_gate(pop, gen)
            
            # Check population size after filtering - FAIL-FAST
            if pop.size() < 2:
                raise RuntimeError(
                    f"FAIL-FAST: Population too small ({pop.size()}) after threshold gate at generation {gen}. "
                    f"Cannot continue evolution with insufficient genomes."
                )
            
            # Show best genome stats
            try:
                best = pop.best()
                if best.has_multi_scores():
                    scores = best.multi_scores
                    print(f"🏆 Best Genome: {best.id}")
                    print(f"   📈 Scores: B:{scores.bugfix:.3f} S:{scores.style:.3f} "
                          f"Sec:{scores.security:.3f} R:{scores.runtime:.3f}")
                    print(f"   🎯 Overall: {best.fitness:.3f}")
            except:
                print(f"⚠️  No evaluated genomes yet")
            
            # Early stopping check
            if self._should_stop_early(pop, gen):
                print(f"🛑 Early stopping triggered at generation {gen + 1}")
                break
            
            # Selection and reproduction
            if gen < self.max_generations - 1:  # Don't evolve on last generation
                print(f"🔄 Evolving population for next generation...")
                pop = self._select_and_mutate(pop, gen)
                print(f"✅ Generation {gen + 2} population ready: {pop.size()} genomes")
            
            # Detect genetic patterns and save tracking data
            self._process_genetic_tracking(gen)
        
        print(f"\n🎉 Evolution Complete!")
        print(f"   🏁 Final generation: {self.current_generation + 1}")
        print(f"   👥 Final population: {pop.size()} genomes")
        
        # Store completion info for result processing
        self.generations_completed = self.current_generation + 1
        self.evolution_completed_fully = (self.current_generation + 1) >= self.max_generations
        
        return pop
    
    def _evaluate_population(self, pop: Population) -> Population:
        """
        Dispatch genome evaluations via Executor with caching support - FAIL-FAST.
        """
        # Only evaluate genomes that haven't been evaluated yet
        unevaluated = [g for g in pop.genomes if not g.is_evaluated()]
        evaluated = [g for g in pop.genomes if g.is_evaluated()]
        
        if not unevaluated:
            print(f"✅ All genomes already evaluated")
            return pop
        
        # Group by heavy genes for cache optimization
        heavy_gene_groups = self._group_by_heavy_genes(unevaluated)
        
        print(f"🔍 GENOME EVALUATION")
        print(f"   📊 Genomes to evaluate: {len(unevaluated)}")
        print(f"   📊 Already evaluated: {len(evaluated)}")
        print(f"   🔄 Cache groups: {len(heavy_gene_groups)}")
        
        # Debug: Show cache efficiency
        if len(heavy_gene_groups) > 0:
            cache_efficiency = len(unevaluated) / len(heavy_gene_groups)
            print(f"   ⚡ Cache efficiency: {cache_efficiency:.1f}x reuse")
            
            # Show group sizes for debugging
            group_sizes = [len(genomes) for genomes in heavy_gene_groups.values()]
            print(f"   📦 Group sizes: {group_sizes}")
            
            # Show sample genome IDs being evaluated
            sample_ids = [g.id for g in unevaluated[:5]]
            if len(unevaluated) > 5:
                sample_ids.append(f"... and {len(unevaluated) - 5} more")
            print(f"   🧬 Evaluating: {', '.join(sample_ids)}")
        
        # Submit evaluation tasks
        futures = []
        for heavy_key, genomes in heavy_gene_groups.items():
            for genome in genomes:
                future = self.executor.submit(self._evaluate_single_genome, genome)
                futures.append(future)
        
        # Collect results
        newly_evaluated = [future.result() for future in futures]
        
        print(f"✅ Evaluation complete: {len(newly_evaluated)} genomes processed")
        
        # Combine with already evaluated genomes
        all_genomes = tuple(evaluated + newly_evaluated)
        return Population(all_genomes)
    
    def _group_by_heavy_genes(self, genomes: List[Genome]) -> Dict[tuple, List[Genome]]:
        """Group genomes by heavy genes for cache optimization."""
        groups = {}
        for genome in genomes:
            heavy_key = genome.get_heavy_genes_key()
            if heavy_key not in groups:
                groups[heavy_key] = []
            groups[heavy_key].append(genome)
        return groups
    
    def _evaluate_single_genome(self, genome: Genome) -> Genome:
        """Enhanced genome evaluation with multi-objective scoring - FAIL-FAST."""
        # 🔥 FIX: Compute CA features ONCE per genome, not per problem
        print(f"🌊 Computing CA features for genome {genome.id}...")
        hist = evolve(genome.seed)
        ca_features = extract_features(hist)
        print(f"   ✅ CA features computed once: complexity={ca_features.complexity:.3f}")
        
        # Get LoRA config (either from genome or derive from features)
        lora_cfg = genome.lora_cfg
        
        # Bridge to plugin world — model + dataset live there
        # 🔥 FIX: Pass both genome AND pre-computed CA features to avoid re-evolution
        model = self.model_factory(lora_cfg, genome=genome)
        problems = list(self.dataset.problems())
        
        # FAIL-FAST: Require multi-objective fitness functions
        if not hasattr(self.fitness_fn, 'evaluate_multi_objective'):
            raise ValueError(
                f"FAIL-FAST: Fitness function must implement evaluate_multi_objective. "
                f"Single-objective fallbacks removed."
            )
        
        # 🔥 FIX: Pass pre-computed CA features to prevent double evolution
        multi_scores = self.fitness_fn.evaluate_multi_objective(
            genome=genome, 
            model=model, 
            problems=problems,
            ca_features=ca_features  # 🔥 NEW: Pass pre-computed features
        )
        
        # Calculate composite fitness for genetic tracking
        composite_fitness = (multi_scores.bugfix + multi_scores.style + 
                           multi_scores.security + multi_scores.runtime + 
                           multi_scores.syntax) / 5.0
        
        # Update genetic tracker with fitness outcome
        self.genetic_tracker.update_fitness_outcomes(
            genome_id=genome.id,
            fitness=composite_fitness,
            multi_scores={
                'bugfix': multi_scores.bugfix,
                'style': multi_scores.style, 
                'security': multi_scores.security,
                'runtime': multi_scores.runtime,
                'syntax': multi_scores.syntax
            }
        )
        
        return genome.with_multi_scores(multi_scores)
    
    def _apply_threshold_gate(self, pop: Population, gen: int) -> Population:
        """Apply population filtering - either threshold gates or Pareto selection."""
        # Check if Pareto selection is enabled
        selection_mode = self.cfg.execution.get('selection_mode', 'threshold')
        target_size = self.cfg.execution.get('population_size', len(pop.genomes))
        
        if selection_mode == 'pareto':
            print(f"🎯 PARETO SELECTION")
            print(f"   Population: {len(pop.genomes)} → {target_size} genomes")
            survivors = nsga2_select(pop, target_size)
            print(f"   ✅ Selected {len(survivors.genomes)} genomes from Pareto fronts")
            return survivors
        
        else:  # Default threshold gate behavior
            if not self.cfg.threshold:
                raise ValueError("FAIL-FAST: Threshold configuration is required but missing")
            
            # Calculate current thresholds
            current_thresholds = calculate_dynamic_thresholds(
                gen, self.max_generations, self.cfg.threshold
            )
            
            print(f"Generation {gen} thresholds: "
                  f"bugfix={current_thresholds.bugfix:.3f}, "
                  f"style={current_thresholds.style:.3f}, "
                  f"security={current_thresholds.security:.3f}, "
                  f"runtime={current_thresholds.runtime:.3f}, "
                  f"syntax={current_thresholds.syntax:.3f}")
            
            # Filter population
            def score_extractor(genome: Genome) -> MultiObjectiveScores:
                if genome.has_multi_scores():
                    return genome.multi_scores
                else:
                    # FAIL-FAST: All genomes must have multi-objective scores
                    raise ValueError(
                        f"FAIL-FAST: Genome {genome} missing multi-objective scores. "
                        f"Single-objective fallbacks removed."
                    )
            
            survivors = filter_population_by_thresholds(
                list(pop.genomes), score_extractor, current_thresholds
            )
            
            print(f"Threshold gate: {len(pop.genomes)} → {len(survivors)} genomes")
            
            return Population(tuple(survivors))
    
    def _select_and_mutate(self, pop: Population, gen: int) -> Population:
        """Selection and reproduction step - FAIL-FAST."""
        population_size = self.cfg.execution['population_size']
        
        # Get survival rate from config or use calculated value
        if 'survival_rate' in self.cfg.execution:
            survival_rate = self.cfg.execution['survival_rate']
        else:
            # Calculate reasonable survival rate based on population size
            survival_rate = max(0.2, min(0.7, 10.0 / population_size))
        
        # Get crossover rate from config
        crossover_rate = self.cfg.execution.get('crossover_rate', 0.7)
        
        # Select survivors
        num_survivors = max(1, int(population_size * survival_rate))
        survivors = select(pop, num_survivors)
        
        print(f"🧬 REPRODUCTION")
        print(f"   🎯 Survivors: {len(survivors.genomes)} (top {survival_rate*100:.1f}%)")
        print(f"   🧪 Crossover rate: {crossover_rate*100:.1f}%")
        print(f"   👶 Children needed: {population_size - len(survivors.genomes)}")
        
        # Generate offspring with generation tracking for IDs
        rng = Random(self.cfg.seed + gen)
        children = []
        child_counter = 0
        crossover_count = 0
        mutation_count = 0
        
        # Calculate dynamic diversity strength for this generation
        diversity_strength = self._calculate_generation_diversity_strength(gen, pop)
        
        # Prepare config dict for dynamic diversity mapping
        config_dict = {
            'evo': {
                'rank_candidates': list(self.cfg.evo.rank_candidates),
                'alpha_candidates': list(self.cfg.evo.alpha_candidates),
                'dropout_candidates': list(self.cfg.evo.dropout_candidates),
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
                'diversity': {
                    'mode': 'adaptive',
                    'base_strength': 1.0,
                    'max_strength': 2.0,
                    'min_strength': 0.3,
                    'cache_threshold': 0.8,
                    'plateau_threshold': 0.05,
                    'plateau_window': 3
                }
            },
            # 🔥 FIX: Include adapter_type for mutations and crossovers
            'adapter_type': getattr(self.cfg, 'adapter_type', 'lora'),  # Extract from main config
            'run_id': self.run_id  # Include run_id for genome creation
        }
        
        while len(children) + len(survivors.genomes) < population_size:
            if len(survivors.genomes) >= 2 and rng.random() < crossover_rate:
                # Crossover with dynamic diversity
                parent1 = choice(survivors.genomes)
                parent2 = choice(survivors.genomes)
                child = crossover(parent1, parent2, self.cfg.evo, rng, generation=gen,
                                diversity_strength=diversity_strength, config_dict=config_dict, run_id=self.run_id)
                
                # Track crossover operation
                self.genetic_tracker.track_crossover(
                    child=child,
                    parent1=parent1,
                    parent2=parent2,
                    generation=gen,
                    diversity_strength=diversity_strength
                )
                
                crossover_count += 1
            else:
                # Mutation with dynamic diversity
                parent = choice(survivors.genomes)
                # Determine mutation type (this will be implemented in neat.py)
                mutation_type = "ca_mutation" if rng.random() < 0.7 else "lora_mutation"
                child = mutate(parent, self.cfg.evo, rng, generation=gen,
                             diversity_strength=diversity_strength, config_dict=config_dict, run_id=self.run_id)
                
                # Track mutation operation
                self.genetic_tracker.track_mutation(
                    child=child,
                    parent=parent,
                    generation=gen,
                    mutation_type=mutation_type,
                    diversity_strength=diversity_strength
                )
                
                mutation_count += 1
            
            children.append(child)
            child_counter += 1
        
        print(f"   🔀 Crossovers: {crossover_count}")
        print(f"   🧬 Mutations: {mutation_count}")
        
        # Show sample new genome IDs
        if children:
            sample_new_ids = [c.id for c in children[:3]]
            if len(children) > 3:
                sample_new_ids.append(f"... +{len(children) - 3} more")
            print(f"   🆕 New genomes: {', '.join(sample_new_ids)}")
        
        # Combine survivors and children
        new_genomes = survivors.genomes + tuple(children)
        return Population(new_genomes[:population_size])
    
    def _should_stop_early(self, pop: Population, gen: int) -> bool:
        """Enhanced early stopping with multi-objective awareness and plateau detection - FAIL-FAST."""
        if pop.size() == 0:
            raise RuntimeError("FAIL-FAST: Empty population encountered during evolution")
        
        # Get early stopping config (enable simple plateau detection by default)
        early_stop_config = self.cfg.execution.get('early_stopping', {'enabled': True})
        if not early_stop_config.get('enabled', True):
            return False
        
        try:
            best = pop.best()
            
            # Track fitness history for plateau detection
            if hasattr(best, 'fitness'):
                current_fitness = best.fitness
            else:
                # Calculate composite fitness from multi-objective scores
                if best.has_multi_scores():
                    scores = best.multi_scores
                    current_fitness = (scores.bugfix + scores.style + scores.security + scores.runtime + scores.syntax) / 5.0
                else:
                    current_fitness = 0.0
            
            # Add to fitness history
            self.generation_history['best_fitness'].append(current_fitness)
            
            print(f"🏆 Generation {gen + 1} best fitness: {current_fitness:.4f}")
            
            # IMPROVED PLATEAU DETECTION: Require minimum generations before early stopping
            PLATEAU_WINDOW = 5  # Increased from 3
            PLATEAU_THRESHOLD = 0.01  # Reduced from 2% to 1% (more lenient)
            MIN_GENERATIONS_BEFORE_EARLY_STOP = max(8, self.max_generations // 3)  # At least 8 generations or 1/3 of total
            
            # Don't consider early stopping until minimum generations completed
            if gen < MIN_GENERATIONS_BEFORE_EARLY_STOP:
                print(f"📊 Early stopping check: {gen + 1}/{MIN_GENERATIONS_BEFORE_EARLY_STOP} minimum generations completed")
                return False
            
            if len(self.generation_history['best_fitness']) >= PLATEAU_WINDOW + 1:
                recent_fitness = self.generation_history['best_fitness'][-PLATEAU_WINDOW:]
                older_fitness = self.generation_history['best_fitness'][-(PLATEAU_WINDOW+1):-1]
                
                recent_best = max(recent_fitness)
                older_best = max(older_fitness) if older_fitness else 0.0
                improvement = recent_best - older_best
                
                print(f"📈 Improvement over last {PLATEAU_WINDOW} generations: {improvement:.4f}")
                
                if improvement < PLATEAU_THRESHOLD:
                    print(f"🛑 EARLY STOPPING: Fitness plateau detected!")
                    print(f"   • No improvement ≥{PLATEAU_THRESHOLD:.1%} for {PLATEAU_WINDOW} generations")
                    print(f"   • Recent fitness: {[f'{f:.3f}' for f in recent_fitness]}")
                    print(f"   • Stopping at generation {gen + 1}/{self.max_generations}")
                    return True
            
            # Multi-objective threshold early stopping (optional)
            if best.has_multi_scores():
                scores = best.multi_scores
                
                # Get thresholds from config (more aggressive defaults for QuixBugs)
                thresholds = early_stop_config.get('thresholds', {})
                bugfix_threshold = thresholds.get('bugfix', 0.95)
                style_threshold = thresholds.get('style', 0.95)
                security_threshold = thresholds.get('security', 0.90)
                runtime_threshold = thresholds.get('runtime', 0.80)
                syntax_threshold = thresholds.get('syntax', 0.80)
                
                # Stop if all objectives meet high thresholds
                if (scores.bugfix >= bugfix_threshold and scores.style >= style_threshold and 
                    scores.security >= security_threshold and scores.runtime >= runtime_threshold and
                    scores.syntax >= syntax_threshold):
                    print(f"🛑 EARLY STOPPING: All objectives meet high thresholds!")
                    print(f"   • B:{scores.bugfix:.3f}≥{bugfix_threshold:.3f}, S:{scores.style:.3f}≥{style_threshold:.3f}")
                    print(f"   • Sec:{scores.security:.3f}≥{security_threshold:.3f}, R:{scores.runtime:.3f}≥{runtime_threshold:.3f}")
                    print(f"   • Syn:{scores.syntax:.3f}≥{syntax_threshold:.3f}")
                    return True
                
        except ValueError as e:
            # No evaluated genomes yet - continue evolution
            if "No evaluated genomes" in str(e):
                return False
            else:
                raise e
        
        return False
    
    def _calculate_generation_diversity_strength(self, generation: int, population: Population) -> float:
        """Calculate dynamic diversity strength based on generation state."""
        from coral.domain.mapping import calculate_dynamic_diversity_strength
        
        # Calculate cache hit rate for current population
        lora_signatures = set()
        for genome in population.genomes:
            signature = f"r{genome.lora_cfg.r}_a{genome.lora_cfg.alpha}_d{genome.lora_cfg.dropout}"
            lora_signatures.add(signature)
        
        # Cache hit rate approximation: how much sharing is happening
        cache_hit_rate = 1.0 - (len(lora_signatures) / len(population.genomes))
        
        # Calculate recent fitness improvements
        recent_improvements = []
        if len(self.generation_history['best_fitness']) >= 2:
            for i in range(1, min(4, len(self.generation_history['best_fitness']))):
                current = self.generation_history['best_fitness'][-1]
                previous = self.generation_history['best_fitness'][-1-i]
                improvement = max(0, current - previous)
                recent_improvements.append(improvement)
        else:
            # No history yet, assume good exploration
            recent_improvements = [0.1]  # Above plateau threshold
        
        # Prepare config dict for calculation
        config_dict = {
            'evo': {
                'diversity': {
                    'mode': 'adaptive',
                    'base_strength': 1.0,
                    'max_strength': 2.0,
                    'min_strength': 0.3,
                    'cache_threshold': 0.8,
                    'plateau_threshold': 0.05,
                    'plateau_window': 3
                }
            }
        }
        
        # Calculate dynamic diversity strength
        diversity_strength = calculate_dynamic_diversity_strength(
            cache_hit_rate, recent_improvements, config_dict
        )
        
        # Update generation history (FITNESS ALREADY ADDED IN _should_stop_early - DON'T DUPLICATE)
        # Fitness tracking moved to _should_stop_early to prevent duplicate entries
        
        self.generation_history['cache_hit_rates'].append(cache_hit_rate)
        self.generation_history['diversity_strengths'].append(diversity_strength)
        
        print(f"🎚️  DYNAMIC DIVERSITY - Generation {generation}")
        print(f"   • Cache hit rate: {cache_hit_rate:.1%}")
        print(f"   • Recent improvements: {recent_improvements}")
        print(f"   • Diversity strength: {diversity_strength:.2f}")
        
        return diversity_strength
    
    def _process_genetic_tracking(self, generation: int):
        """Process genetic operations tracking for current generation."""
        
        # Detect genetic patterns
        patterns = self.genetic_tracker.detect_genetic_patterns(generation)
        
        # Show generation summary
        summary = self.genetic_tracker.get_generation_summary(generation)
        print(f"\n🧬 GENETIC OPERATIONS SUMMARY - Generation {generation + 1}")
        print(f"   🔀 Crossovers: {summary['crossovers']} (success rate: {summary['crossover_success_rate']:.1f}%)")
        print(f"   🧬 Mutations: {summary['mutations']} (success rate: {summary['mutation_success_rate']:.1f}%)")
        print(f"      • CA mutations: {summary['ca_mutations']}")
        print(f"      • LoRA mutations: {summary['lora_mutations']}")
        print(f"   🎯 Patterns detected: {summary['patterns_detected']}")
        
        # Save tracking data every few generations
        if generation % 3 == 0 or generation == self.max_generations - 1:
            self.genetic_tracker.save_tracking_data(generation) 

# Legacy evolution functions removed - use EvolutionEngine.run() instead 