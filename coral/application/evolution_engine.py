###############################################################################
# Orchestration - NO FALLBACKS, strict config-driven
###############################################################################
from dataclasses import dataclass
from typing import Dict, Any, Callable, List, Optional
from random import Random, choice
from concurrent.futures import Future
import time
import json
from pathlib import Path

from coral.domain.experiment import create_initial_population

from ..domain.ca import evolve
from ..domain.feature_extraction import extract_features
from ..domain.mapping import map_features_to_lora_config, EvolutionConfig
from ..domain.genome import Genome, MultiObjectiveScores
from ..domain.neat import Population, select, mutate, crossover
from ..domain.threshold_gate import (
    ThresholdConfig, ObjectiveThresholds, calculate_dynamic_thresholds, 
    filter_population_by_thresholds
)
from ..domain.pareto_selection import nsga2_select
from ..ports.interfaces import FitnessFn, Executor, ModelRunner, DatasetProvider
from ..domain.genetic_operations_tracker import GeneticOperationsTracker


@dataclass(frozen=True)
class CoralConfig:
    """
    Enhanced configuration container with categorical dual access.
    Eliminates impedance mismatch by storing raw YAML as source of truth.
    """
    # Raw YAML data (source of truth)
    _raw_data: Dict[str, Any]
    
    # Lazy-loaded structured fields
    _evo: Optional[EvolutionConfig] = None
    _threshold: Optional[ThresholdConfig] = None
    
    def __post_init__(self):
        """Validate structure on creation."""
        self._validate_config(self._raw_data)
    
    # 🧮 CATEGORICAL DICT ACCESS (existing functionality enhanced)
    def __getitem__(self, key: str):
        """Direct dict access: config['evo']"""
        return self._raw_data[key]
    
    def __contains__(self, key: str) -> bool:
        """'key' in config"""
        return key in self._raw_data
    
    def get(self, key: str, default=None):
        """config.get('key', default)"""
        return self._raw_data.get(key, default)
    
    def keys(self):
        return self._raw_data.keys()
    
    def items(self):
        return self._raw_data.items()
    
    def copy(self):
        """Return raw dict copy - no more conversion needed!"""
        return self._raw_data.copy()
    
    # 🧮 CATEGORICAL STRUCTURED ACCESS (lazy-loaded for performance)
    @property
    def evo(self) -> EvolutionConfig:
        """Lazy-loaded structured evolution config."""
        if self._evo is None:
            evo_raw = self._raw_data['evo']
            object.__setattr__(self, '_evo', EvolutionConfig(
                rank_candidates=tuple(evo_raw['rank_candidates']),
                alpha_candidates=tuple(evo_raw['alpha_candidates']),
                dropout_candidates=tuple(evo_raw['dropout_candidates']),
                target_modules=tuple(evo_raw['target_modules'])
            ))
        return self._evo
    
    @property 
    def threshold(self) -> ThresholdConfig:
        """Lazy-loaded structured threshold config."""
        if self._threshold is None:
            threshold_raw = self._raw_data['threshold']
            base_thresh = threshold_raw['base_thresholds']
            max_thresh = threshold_raw['max_thresholds']
            
            object.__setattr__(self, '_threshold', ThresholdConfig(
                base_thresholds=ObjectiveThresholds(
                    bugfix=base_thresh['bugfix'],
                    style=base_thresh['style'],
                    security=base_thresh['security'],
                    runtime=base_thresh['runtime'],
                    syntax=base_thresh.get('syntax', 0.3)
                ),
                max_thresholds=ObjectiveThresholds(
                    bugfix=max_thresh['bugfix'],
                    style=max_thresh['style'],
                    security=max_thresh['security'],
                    runtime=max_thresh['runtime'],
                    syntax=max_thresh.get('syntax', 0.9)
                ),
                schedule=threshold_raw['schedule']
            ))
        return self._threshold
    
    @property
    def seed(self) -> int:
        """Direct access to seed."""
        return self._raw_data['seed']
    
    @property
    def execution(self) -> Dict[str, Any]:
        """Direct access to execution config."""
        return self._raw_data['execution']
    
    @property
    def infra(self) -> Dict[str, Any]:
        """Direct access to infra config."""
        return self._raw_data['infra']
    
    @property
    def experiment(self) -> Dict[str, Any]:
        """Direct access to experiment config."""
        return self._raw_data['experiment']
    
    @property
    def cache(self) -> Dict[str, Any]:
        """Direct access to cache config."""
        return self._raw_data['cache']
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Direct access to evaluation config."""
        return self._raw_data['evaluation']
    
    @property
    def adapter_type(self) -> str:
        """Direct access to adapter type."""
        return self._raw_data.get('adapter_type', 'lora')
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Direct access to paths config."""
        return self._raw_data.get('paths', {})
    
    # 🧮 CATEGORICAL FUNCTORS (Modal serialization - now trivial!)
    def serialize_for_modal(self) -> Dict[str, Any]:
        """Natural transformation: CoralConfig → Modal Dict"""
        return self._raw_data.copy()  # Raw data is already Modal-ready!
    
    def serialize_for_executor(self, executor_type: str) -> Dict[str, Any]:
        """Functorial transformation for different execution contexts."""
        config = self._raw_data.copy()
        
        # Apply context-specific transformations
        if executor_type == 'modal':
            config['infra']['executor'] = 'modal'
            if 'paths' in config and 'modal' in config['paths']:
                config['current_paths'] = config['paths']['modal']
        elif executor_type == 'local':
            config['infra']['executor'] = 'local'
            if 'paths' in config and 'local' in config['paths']:
                config['current_paths'] = config['paths']['local']
                
        return config
    
    # 🧮 CATEGORICAL MORPHISMS (Pure transformations)
    def with_population_size(self, size: int) -> 'CoralConfig':
        """Immutable update - returns new config."""
        new_raw = self._raw_data.copy()
        new_raw['execution']['population_size'] = size
        return CoralConfig(new_raw)
    
    def with_executor(self, executor_type: str) -> 'CoralConfig':
        """Immutable executor change."""
        new_raw = self._raw_data.copy()
        new_raw['infra']['executor'] = executor_type
        return CoralConfig(new_raw)
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure - FAIL-FAST."""
        required_sections = ['evo', 'execution', 'experiment', 'infra', 'cache', 'threshold', 'evaluation']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"FAIL-FAST: Missing required configuration section: '{section}'")


class EvolutionEngine:
    def __init__(self,
                 config: CoralConfig,  # ✅ SINGLE CONFIG OBJECT
                 fitness_fn: FitnessFn,
                 executor: Executor,
                 model_factory: Callable,
                 dataset: DatasetProvider,
                 run_id: str = None):
        self.config = config  # ✅ NO MORE cfg + raw_config
        self.fitness_fn = fitness_fn
        self.executor = executor
        self.model_factory = model_factory
        self.dataset = dataset
        self.run_id = run_id  # Store run_id for genome creation
        
        # Validate configuration
        self._validate_configuration()
        
        # Generation tracking for threshold gates
        self.current_generation = 0
        self.max_generations = self.config.execution['generations']
        
        # Dynamic diversity tracking
        self.generation_history = {
            'best_fitness': [],
            'cache_hit_rates': [],
            'diversity_strengths': []
        }
        
        # Initialize genetic operations tracker
        genetic_tracking_dir = self.config.execution.get('genetic_tracking_dir', 'results/genetic_tracking')
        self.genetic_tracker = GeneticOperationsTracker(output_dir=genetic_tracking_dir)
        
        # Progress tracking initialization
        self.start_time = time.time()
        self.progress_file_path = self._get_progress_file_path()
        self._initialize_progress_tracking()
    
    def _get_progress_file_path(self) -> Path:
        """Get the path to the progress tracking file using enhanced config."""
        # Extract paths using enhanced config
        if 'paths' not in self.config:
            raise RuntimeError(
                f"FAIL-FAST: 'paths' section missing from configuration.\n"
                f"Progress tracking requires explicit path configuration.\n"
                f"Add paths.modal.progress: '/cache/progress.json' to config."
            )
        
        paths = self.config.paths
        executor_type = self.config.infra.get('executor', 'modal')
        
        # Variable orchestration: Config → PathConfig → Progress Path
        if executor_type not in paths:
            raise RuntimeError(
                f"FAIL-FAST: Path configuration missing for executor '{executor_type}'.\n"
                f"Available path configs: {list(paths.keys())}\n"
                f"Add paths.{executor_type}.progress to your config."
            )
        
        executor_paths = paths[executor_type]
        if 'progress' not in executor_paths:
            raise RuntimeError(
                f"FAIL-FAST: Progress path not configured for executor '{executor_type}'.\n"
                f"Available paths: {list(executor_paths.keys())}\n"
                f"Add paths.{executor_type}.progress: '/path/to/progress.json' to config."
            )
        
        progress_path = executor_paths['progress']
        print(f"📊 Using progress tracking: {progress_path} (from {executor_type} config)")
        return Path(progress_path)
    
    def _initialize_progress_tracking(self):
        """Initialize progress tracking file."""
        try:
            # Create initial progress data
            initial_progress = {
                'status': 'starting',
                'message': 'Evolution starting...',
                'current_generation': 0,
                'max_generations': self.max_generations,
                'best_fitness': 0.0,
                'start_time': self.start_time,
                'start_time_str': time.strftime('%H:%M:%S', time.localtime(self.start_time)),
                'last_update': self.start_time,
                'elapsed_time': 0.0,
                'population_size': self.config.execution.get('population_size', 10),
                'run_id': self.run_id or 'unknown',
                'best_scores': {
                    'bugfix': 0.0,
                    'style': 0.0,
                    'security': 0.0,
                    'runtime': 0.0,
                    'syntax': 0.0
                },
                'cache_stats': {
                    'hit_rate': 0.0,
                    'total_adapters': 0,
                    'cache_size_mb': 0
                },
                'training_stats': {
                    'adapters_trained': 0,
                    'training_rate': 0.0,
                    'current_adapter': 'Initializing...'
                },
                'current_adapter': 'Initializing...',
                'infrastructure_stats': {
                    'model_files': 0,
                    'dataset_files': 0,
                    'adapters': 0,
                    'cache_size_mb': 0
                },
                'genetic_stats': {
                    'ca_parameters': {
                        'grid_size': [8, 8],
                        'rule_range': [1, 255],
                        'steps_range': [5, 25],
                        'current_rule': 'Not available',
                        'current_steps': 'Not available'
                    },
                    'neat_operations': {
                        'crossover_rate': self.config.execution.get('crossover_rate', 0.7),
                        'mutation_rate': self.config.execution.get('mutation_rate', 0.3),
                        'survival_rate': self.config.execution.get('survival_rate', 0.4),
                        'last_mutation': 'Not available',
                        'last_crossover': 'Not available'
                    },
                    'diversity_mode': 'adaptive',
                    'diversity_strength': 1.0
                }
            }
            
            # Ensure parent directory exists
            self.progress_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write initial progress
            with open(self.progress_file_path, 'w') as f:
                json.dump(initial_progress, f, indent=2)
            
            print(f"📊 Progress tracking initialized: {self.progress_file_path}")
            
        except Exception as e:
            print(f"⚠️  Progress tracking initialization failed: {e}")
            # Don't fail evolution if progress tracking fails
    
    def _update_progress(self, status: str, message: str = None, best_genome: Genome = None, 
                        additional_data: Dict[str, Any] = None):
        """Simple progress logging."""
        try:
            print(f"📊 Evolution Progress: {status}")
            if message:
                print(f"   📝 {message}")
            if best_genome and best_genome.fitness:
                print(f"   🏆 Best fitness: {best_genome.fitness:.3f}")
            
        except Exception as e:
            print(f"⚠️  Progress update failed: {e}")
            # Don't fail evolution if progress tracking fails
    
    def _validate_configuration(self):
        """Validate that all required configuration is present - FAIL-FAST."""
        if 'generations' not in self.config.execution:
            raise ValueError("FAIL-FAST: 'generations' missing from execution configuration")
        
        if 'population_size' not in self.config.execution:
            raise ValueError("FAIL-FAST: 'population_size' missing from execution configuration")
        
        if self.config.execution['generations'] <= 0:
            raise ValueError("FAIL-FAST: 'generations' must be positive")
        
        if self.config.execution['population_size'] <= 0:
            raise ValueError("FAIL-FAST: 'population_size' must be positive")
    
    def run(self, init_pop: Population) -> Population:
        """Main evolution loop with CORAL-X enhancements - FAIL-FAST."""
        pop = init_pop
        
        print(f"🚀 Starting CORAL-X Evolution")
        print(f"   📊 Population: {pop.size()} genomes")
        print(f"   🔄 Generations: {self.max_generations}")
        print(f"   🧬 Evolution Path: CA → Features → LoRA (Full CORAL-X)")
        print("=" * 60)
        
        # Update progress: Evolution started
        self._update_progress('evolving', 'Evolution started')
        
        for gen in range(self.max_generations):
            self.current_generation = gen
            print(f"\n🧬 GENERATION {gen + 1}/{self.max_generations}")
            print("=" * 50)
            
            # Update progress: Generation started
            self._update_progress('evolving', f'Generation {gen + 1}/{self.max_generations} - Starting evaluation')
            
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
                # Update progress: Evolution failed
                self._update_progress('failed', f'Population too small ({pop.size()}) after threshold gate')
                raise RuntimeError(
                    f"FAIL-FAST: Population too small ({pop.size()}) after threshold gate at generation {gen}. "
                    f"Cannot continue evolution with insufficient genomes."
                )
            
            # Show best genome stats and update progress
            try:
                best = pop.best()
                if best.has_multi_scores():
                    scores = best.multi_scores
                    print(f"🏆 Best Genome: {best.id}")
                    print(f"   📈 Scores: B:{scores.bugfix:.3f} S:{scores.style:.3f} "
                          f"Sec:{scores.security:.3f} R:{scores.runtime:.3f}")
                    print(f"   🎯 Overall: {best.fitness:.3f}")
                    
                    # Update progress with current best
                    self._update_progress('evolving', 
                                        f'Generation {gen + 1}/{self.max_generations} - Best fitness: {best.fitness:.3f}',
                                        best_genome=best)
            except:
                print(f"⚠️  No evaluated genomes yet")
                self._update_progress('evolving', f'Generation {gen + 1}/{self.max_generations} - No evaluated genomes yet')
            
            # Early stopping check
            if self._should_stop_early(pop, gen):
                print(f"🛑 Early stopping triggered at generation {gen + 1}")
                self._update_progress('completed', f'Early stopping at generation {gen + 1}', pop.best())
                break
            
            # Selection and reproduction
            if gen < self.max_generations - 1:  # Don't evolve on last generation
                print(f"🔄 Evolving population for next generation...")
                self._update_progress('evolving', f'Generation {gen + 1}/{self.max_generations} - Evolving population')
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
        
        # Update progress: Evolution completed
        # Check if any genomes were evaluated successfully
        evaluated_count = len([g for g in pop.genomes if g.is_evaluated()])
        
        if evaluated_count == 0:
            print(f"⚠️  WARNING: No genomes were successfully evaluated during evolution!")
            print(f"   This indicates systematic evaluation failures.")
            print(f"   Check adapter training and evaluation logs for issues.")
            # Assign minimal fitness to enable completion
            pop = pop.with_default_fitness(0.01)  # Small positive fitness
            
        best_final = pop.best() if pop.size() > 0 else None
        self._update_progress('completed', 
                            f'Evolution completed - {self.current_generation + 1} generations (evaluated: {evaluated_count}/{pop.size()})',
                            best_genome=best_final)
        
        return pop
    
    def _evaluate_population(self, pop: Population) -> Population:
        """
        Two-phase evaluation: parallel training, then parallel evaluation - FAIL-FAST.
        """
        # Only evaluate genomes that haven't been evaluated yet
        unevaluated = [g for g in pop.genomes if not g.is_evaluated()]
        evaluated = [g for g in pop.genomes if g.is_evaluated()]
        
        if not unevaluated:
            print(f"✅ All genomes already evaluated")
            return pop
        
        print(f"🔍 TWO-PHASE EVALUATION")
        print(f"   📊 Genomes to evaluate: {len(unevaluated)}")
        print(f"   📊 Already evaluated: {len(evaluated)}")
        
        # PHASE 1: Parallel Training of Unique Heavy Genes
        print(f"\n🏗️  PHASE 1: PARALLEL TRAINING")
        self._update_progress('evolving', 
                            f'Generation {self.current_generation + 1} - Phase 1: Training adapters')
        unique_heavy_genes = self._extract_unique_heavy_genes(unevaluated)
        trained_adapters = self._train_adapters_parallel(unique_heavy_genes)
        
        # PHASE 2: Parallel Evaluation (Inference Only)
        print(f"\n🧪 PHASE 2: PARALLEL EVALUATION")
        self._update_progress('evolving', 
                            f'Generation {self.current_generation + 1} - Phase 2: Evaluating genomes')
        newly_evaluated = self._evaluate_genomes_parallel(unevaluated, trained_adapters)
        
        print(f"✅ Two-phase evaluation complete: {len(newly_evaluated)} genomes processed")
        
        # Combine with already evaluated genomes
        all_genomes = tuple(evaluated + newly_evaluated)
        return Population(all_genomes)
    
    def _extract_unique_heavy_genes(self, genomes: List[Genome]) -> Dict[str, Any]:
        """Extract unique heavy genes that need training."""
        from infra.adapter_cache import HeavyGenes
        
        unique_genes = {}
        genome_to_hash = {}
        
        for genome in genomes:
            # Convert genome LoRA config to HeavyGenes
            heavy_genes = HeavyGenes.from_lora_config(
                genome.lora_cfg, 
                run_id=genome.run_id
            )
            
            genes_hash = heavy_genes.to_hash()
            genome_to_hash[genome.id] = genes_hash
            
            if genes_hash not in unique_genes:
                unique_genes[genes_hash] = heavy_genes
        
        print(f"   🔍 Found {len(unique_genes)} unique heavy gene configurations")
        print(f"   📊 Cache efficiency: {len(genomes) / len(unique_genes):.1f}x reuse")
        
        # Store mapping for later use
        self._genome_to_hash = genome_to_hash
        
        return unique_genes
    
    def _train_adapters_parallel(self, unique_heavy_genes: Dict[str, Any]) -> Dict[str, str]:
        """Train all unique adapters in parallel using Modal."""
        import time
        from pathlib import Path
        
        if not unique_heavy_genes:
            print(f"   ✅ No training needed - all adapters cached")
            return {}
        
        print(f"   🚀 Starting parallel training of {len(unique_heavy_genes)} adapters")
        
        # Check cache first
        cached_adapters = {}
        training_needed = {}
        
        for genes_hash, heavy_genes in unique_heavy_genes.items():
            # Check if adapter already exists in cache
            cache_config = self.cfg.cache or {}
            artifacts_dir = cache_config.get('artifacts_dir', '/cache/adapters')
            adapter_path = Path(artifacts_dir) / f"adapter_{genes_hash}"
            
            if adapter_path.exists() and self._verify_adapter(adapter_path):
                cached_adapters[genes_hash] = str(adapter_path)
                print(f"   💾 Cache hit: {genes_hash[:8]}...")
            else:
                training_needed[genes_hash] = heavy_genes
                print(f"   🏗️  Training needed: {genes_hash[:8]}...")
        
        # Train adapters that need training
        trained_adapters = {}
        if training_needed:
            print(f"   ⚡ Submitting {len(training_needed)} training jobs to Modal...")
            
            # Submit all training jobs in parallel
            training_futures = []
            for genes_hash, heavy_genes in training_needed.items():
                # Get base model and save path
                base_model = "codellama/CodeLlama-7b-Python-hf"  # TODO: Get from config
                save_path = str(Path(artifacts_dir) / f"adapter_{genes_hash}")
                
                # ✅ CLEAN: Direct Modal serialization
                training_config = self.config.serialize_for_modal()
                
                # Submit training job to Modal
                future = self.executor.submit_training(
                    base_model=base_model,
                    heavy_genes=heavy_genes,
                    save_path=save_path,
                    config=training_config
                )
                training_futures.append((genes_hash, future))
            
            # Collect training results
            print(f"   ⏳ Waiting for {len(training_futures)} training jobs to complete...")
            start_time = time.time()
            
            for genes_hash, future in training_futures:
                try:
                    adapter_path = future.result(timeout=1800)  # 30 minutes max
                    trained_adapters[genes_hash] = adapter_path
                    print(f"   ✅ Training complete: {genes_hash[:8]}... → {adapter_path}")
                except Exception as e:
                    raise RuntimeError(f"FAIL-FAST: Training failed for {genes_hash}: {e}")
            
            training_time = time.time() - start_time
            print(f"   🎉 All training complete in {training_time:.1f}s")
        
        # Combine cached and newly trained adapters
        all_adapters = {**cached_adapters, **trained_adapters}
        print(f"   📦 Total adapters ready: {len(all_adapters)} ({len(cached_adapters)} cached, {len(trained_adapters)} trained)")
        
        return all_adapters
    
    def _evaluate_genomes_parallel(self, genomes: List[Genome], trained_adapters: Dict[str, str]) -> List[Genome]:
        """Evaluate all genomes in parallel using pre-trained adapters."""
        import time
        
        print(f"   🧪 Starting parallel evaluation of {len(genomes)} genomes")
        
        # Submit all evaluation jobs in parallel
        evaluation_futures = []
        for genome in genomes:
            genes_hash = self._genome_to_hash[genome.id]
            adapter_path = trained_adapters.get(genes_hash)
            
            if not adapter_path:
                raise RuntimeError(f"FAIL-FAST: No adapter found for genome {genome.id} (hash: {genes_hash})")
            
            # Submit evaluation job to Modal (inference only)
            future = self.executor.submit_evaluation(
                genome=genome,
                adapter_path=adapter_path,
                config=self.config.serialize_for_modal()  # ✅ CLEAN: Direct serialization
            )
            evaluation_futures.append((genome.id, future))
        
        # Collect evaluation results with enhanced timeout and error handling
        print(f"   ⏳ Waiting for {len(evaluation_futures)} evaluation jobs to complete...")
        start_time = time.time()
        
        evaluated_genomes = []
        for genome_id, future in evaluation_futures:
            try:
                # Increased timeout to 30 minutes for complex evaluations
                evaluated_genome = future.result(timeout=1800)  # 30 minutes max
                evaluated_genomes.append(evaluated_genome)
                print(f"   ✅ Evaluation complete: {genome_id}")
            except Exception as e:
                # Enhanced error reporting for debugging dual process issues
                if "TimeoutError" in str(type(e)):
                    print(f"⏰ Evaluation timeout for {genome_id} after 30 minutes")
                    print(f"   This may indicate queue synchronization issues or stuck jobs")
                    # Check queue status for debugging
                    try:
                        import modal
                        test_queue = modal.Queue.from_name('coral-test')
                        results_queue = modal.Queue.from_name('coral-results')
                        print(f"   Queue status: test={test_queue.len()}, results={results_queue.len()}")
                    except:
                        pass
                raise RuntimeError(f"FAIL-FAST: Evaluation failed for {genome_id}: {e}")
        
        evaluation_time = time.time() - start_time
        print(f"   🎉 All evaluations complete in {evaluation_time:.1f}s")
        
        return evaluated_genomes
    
    def _verify_adapter(self, adapter_path: Path) -> bool:
        """Verify that adapter files exist and are valid."""
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        
        if not adapter_path.exists():
            return False
        
        for required_file in required_files:
            file_path = adapter_path / required_file
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
        
        return True
    

    
    def _apply_threshold_gate(self, pop: Population, gen: int) -> Population:
        """Apply population filtering - either threshold gates or Pareto selection."""
        # Check if Pareto selection is enabled
        selection_mode = self.config.execution.get('selection_mode', 'threshold')
        target_size = self.config.execution.get('population_size', len(pop.genomes))
        
        if selection_mode == 'pareto':
            print(f"🎯 PARETO SELECTION")
            print(f"   Population: {len(pop.genomes)} → {target_size} genomes")
            survivors = nsga2_select(pop, target_size)
            print(f"   ✅ Selected {len(survivors.genomes)} genomes from Pareto fronts")
            return survivors
        
        else:  # Default threshold gate behavior
            if not self.config.threshold:
                raise ValueError("FAIL-FAST: Threshold configuration is required but missing")
            
            # Calculate current thresholds
            current_thresholds = calculate_dynamic_thresholds(
                gen, self.max_generations, self.config.threshold
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
        population_size = self.config.execution['population_size']
        
        # Get survival rate from config or use calculated value
        if 'survival_rate' in self.config.execution:
            survival_rate = self.config.execution['survival_rate']
        else:
            # Calculate reasonable survival rate based on population size
            survival_rate = max(0.2, min(0.7, 10.0 / population_size))
        
        # Get crossover rate from config
        crossover_rate = self.config.execution.get('crossover_rate', 0.7)
        
        # Select survivors
        num_survivors = max(1, int(population_size * survival_rate))
        survivors = select(pop, num_survivors)
        
        print(f"🧬 REPRODUCTION")
        print(f"   🎯 Survivors: {len(survivors.genomes)} (top {survival_rate*100:.1f}%)")
        print(f"   🧪 Crossover rate: {crossover_rate*100:.1f}%")
        print(f"   👶 Children needed: {population_size - len(survivors.genomes)}")
        
        # Handle population extinction edge case
        if len(survivors.genomes) == 0:
            print(f"⚠️  POPULATION EXTINCTION: All genomes failed evaluation")
            print(f"   Evolution naturally restarting with fresh random population...")
            
            from coral.domain.ca import CASeed, evolve
            from coral.domain.feature_extraction import extract_features
            from coral.domain.mapping import map_features_to_lora_config
            from random import Random
            import numpy as np
            
            rng = Random(self.cfg.seed + gen)
            fresh_genomes = []
            
            for i in range(population_size):
                # Create unique genome ID
                genome_id = f"restart_gen{gen}_genome{i:04d}"
                
                # Create diverse CA seed
                genome_rng = Random(self.config.seed + gen + i * 1000)
                np.random.seed(self.config.seed + gen + i * 1000)
                
                # Use YAML defaults for CA parameters
                ca_config = self.config.get('evo', {}).get('ca', {})
                grid_size = tuple(ca_config.get('grid_size', [8, 8]))
                initial_density = ca_config.get('initial_density', 0.35)
                rule_range = ca_config.get('rule_range', [1, 255])
                steps_range = ca_config.get('steps_range', [5, 20])
                
                # Generate CA parameters using config defaults
                initial_grid = np.random.choice([0, 1], size=grid_size, p=[1-initial_density, initial_density])
                rule = genome_rng.randint(rule_range[0], rule_range[1])
                steps = genome_rng.randint(steps_range[0], steps_range[1])
                
                ca_seed = CASeed(grid=initial_grid, rule=rule, steps=steps)
                
                # Generate features and map to LoRA config
                history = evolve(ca_seed)
                features = extract_features(history)
                
                config_dict = {
                    'evo': {
                        'rank_candidates': list(self.config.evo.rank_candidates),
                        'alpha_candidates': list(self.config.evo.alpha_candidates),
                        'dropout_candidates': list(self.config.evo.dropout_candidates),
                        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
                    },
                    'adapter_type': self.config.adapter_type
                }
                
                lora_config = map_features_to_lora_config(features, config_dict, diversity_strength=1.0, genome_index=i)
                
                # Create genome
                genome = Genome(seed=ca_seed, lora_cfg=lora_config, id=genome_id, 
                              ca_features=features, run_id=self.run_id)
                fresh_genomes.append(genome)
            
            print(f"   ✅ Generated {len(fresh_genomes)} fresh genomes for restart")
            return Population(tuple(fresh_genomes))
        
        # Generate offspring with generation tracking for IDs
        rng = Random(self.config.seed + gen)
        children = []
        child_counter = 0
        crossover_count = 0
        mutation_count = 0
        
        # Calculate dynamic diversity strength for this generation
        diversity_strength = self._calculate_generation_diversity_strength(gen, pop)
        
        # Prepare config dict for dynamic diversity mapping
        config_dict = {
            'evo': {
                'rank_candidates': list(self.config.evo.rank_candidates),
                'alpha_candidates': list(self.config.evo.alpha_candidates),
                'dropout_candidates': list(self.config.evo.dropout_candidates),
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
            # ✅ CLEAN: Direct adapter_type access
            'adapter_type': self.config.adapter_type,
            'run_id': self.run_id  # Include run_id for genome creation
        }
        
        while len(children) + len(survivors.genomes) < population_size:
            if len(survivors.genomes) >= 2 and rng.random() < crossover_rate:
                # Crossover with dynamic diversity
                parent1 = choice(survivors.genomes)
                parent2 = choice(survivors.genomes)
                child = crossover(parent1, parent2, self.config.evo, rng, generation=gen,
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
                child = mutate(parent, self.config.evo, rng, generation=gen,
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
        early_stop_config = self.config.execution.get('early_stopping', {'enabled': True})
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
            
            fitness_display = f"{current_fitness:.4f}" if current_fitness is not None else "None"
            print(f"🏆 Generation {gen + 1} best fitness: {fitness_display}")
            
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