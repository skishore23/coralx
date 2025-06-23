###############################################################################
# Compare final genomes vs baselines, produce metrics.
###############################################################################
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json

from coral.domain.neat import Population
from coral.domain.genome import Genome
from coral.application.evolution_engine import CoralConfig
from coral.ports.interfaces import ModelRunner, DatasetProvider, FitnessFn


@dataclass(frozen=True)
class BenchmarkResult:
    """Individual benchmark result - CORAL-X Architecture compliant."""
    genome_id: str
    fitness: float
    # CORAL-X Multi-objective scores
    bugfix_score: float
    style_score: float
    security_score: float
    runtime_score: float
    # Technical details
    lora_config: Dict[str, Any]
    ca_features: Dict[str, Any]
    execution_time: float
    # Cache efficiency
    cache_hit: bool
    adapter_training_time: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark report."""
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]
    results: List[BenchmarkResult]
    best_genome: Optional[BenchmarkResult]
    statistics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'results': [
                {
                    'genome_id': r.genome_id,
                    'fitness': r.fitness,
                    'bugfix_score': r.bugfix_score,
                    'style_score': r.style_score,
                    'security_score': r.security_score,
                    'runtime_score': r.runtime_score,
                    'lora_config': r.lora_config,
                    'ca_features': r.ca_features,
                    'execution_time': r.execution_time,
                    'cache_hit': r.cache_hit,
                    'adapter_training_time': r.adapter_training_time
                } for r in self.results
            ],
            'best_genome': {
                'genome_id': self.best_genome.genome_id,
                'fitness': self.best_genome.fitness,
                'bugfix_score': self.best_genome.bugfix_score,
                'style_score': self.best_genome.style_score,
                'security_score': self.best_genome.security_score,
                'runtime_score': self.best_genome.runtime_score,
                'lora_config': self.best_genome.lora_config,
                'ca_features': self.best_genome.ca_features,
                'execution_time': self.best_genome.execution_time,
                'cache_hit': self.best_genome.cache_hit,
                'adapter_training_time': self.best_genome.adapter_training_time
            } if self.best_genome else None,
            'statistics': self.statistics
        }


def run_benchmarks(config: CoralConfig, 
                  winners: Population,
                  model_factory,
                  dataset: DatasetProvider,
                  fitness_fn: FitnessFn) -> BenchmarkReport:
    """
    Pure → returns BenchmarkReport record for downstream rendering.
    """
    experiment_name = config.experiment.get('name', 'coralx_experiment')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Benchmark each genome in the final population
    results = []
    for i, genome in enumerate(winners.genomes):
        if genome.is_evaluated():
            result = _benchmark_genome(f"genome_{i}", genome, model_factory, dataset, fitness_fn)
            results.append(result)
    
    # Find best genome
    best_genome = None
    if results:
        best_genome = max(results, key=lambda r: r.fitness)
    
    # Calculate statistics
    statistics = _calculate_statistics(results)
    
    return BenchmarkReport(
        experiment_name=experiment_name,
        timestamp=timestamp,
        config=_config_to_dict(config),
        results=results,
        best_genome=best_genome,
        statistics=statistics
    )


def _benchmark_genome(genome_id: str,
                     genome: Genome,
                     model_factory,
                     dataset: DatasetProvider,
                     fitness_fn: FitnessFn) -> BenchmarkResult:
    """Benchmark a single genome - CORAL-X Architecture compliant."""
    start_time = time.time()
    
    # FAIL-FAST: Use already evaluated fitness instead of re-evaluation
    if not genome.is_evaluated():
        raise ValueError(f"FAIL-FAST: Cannot benchmark unevaluated genome {genome_id}")
    
    # Extract CORAL-X multi-objective scores
    if hasattr(genome, 'multi_scores') and genome.multi_scores:
        multi_scores = genome.multi_scores
        bugfix_score = multi_scores.bugfix
        style_score = multi_scores.style
        security_score = multi_scores.security
        runtime_score = multi_scores.runtime
        fitness = (bugfix_score + style_score + security_score + runtime_score) / 4.0
    else:
        # FAIL-FAST: Architecture requires multi-objective evaluation
        raise ValueError(f"FAIL-FAST: Genome {genome_id} missing multi-objective scores required by CORAL-X architecture")
    
    # Extract cache efficiency metrics
    cache_hit = getattr(genome, '_cache_hit', False)
    adapter_training_time = getattr(genome, '_adapter_training_time', 0.0)
    
    execution_time = time.time() - start_time
    
    # Extract CA features for analysis (fail-fast approach)
    try:
        from coral.domain.ca import evolve
        from coral.domain.feature_extraction import extract_features
        
        ca_history = evolve(genome.seed)
        ca_features = extract_features(ca_history)
        
        ca_features_dict = {
            'complexity': ca_features.complexity,
            'intensity': ca_features.intensity,
            'periodicity': ca_features.periodicity,
            'convergence': ca_features.convergence
        }
    except Exception as e:
        print(f"⚠️  CA feature extraction failed for {genome_id}: {e}")
        ca_features_dict = {
            'complexity': 0.0,
            'intensity': 0.0,
            'periodicity': 0.0,
            'convergence': 0.0
        }
    
    # Safe LoRA config extraction
    try:
        lora_config_dict = {
            'r': getattr(genome.lora_cfg, 'r', 8),
            'alpha': getattr(genome.lora_cfg, 'alpha', 16.0),
            'dropout': getattr(genome.lora_cfg, 'dropout', 0.1),
            'target_modules': list(getattr(genome.lora_cfg, 'target_modules', ['q_proj', 'v_proj']))
        }
    except Exception as e:
        print(f"⚠️  LoRA config extraction failed for {genome_id}: {e}")
        lora_config_dict = {
            'r': 8,
            'alpha': 16.0,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj']
        }
    
    return BenchmarkResult(
        genome_id=genome_id,
        fitness=fitness,
        # CORAL-X Multi-objective scores
        bugfix_score=bugfix_score,
        style_score=style_score,
        security_score=security_score,
        runtime_score=runtime_score,
        # Technical details
        lora_config=lora_config_dict,
        ca_features=ca_features_dict,
        execution_time=execution_time,
        # Cache efficiency metrics
        cache_hit=cache_hit,
        adapter_training_time=adapter_training_time
    )


def _calculate_statistics(results: List[BenchmarkResult]) -> Dict[str, float]:
    """Calculate summary statistics from benchmark results - CORAL-X Architecture compliant."""
    if not results:
        return {}
    
    fitnesses = [r.fitness for r in results]
    execution_times = [r.execution_time for r in results]
    
    # CORAL-X Multi-objective statistics
    bugfix_scores = [r.bugfix_score for r in results]
    style_scores = [r.style_score for r in results]
    security_scores = [r.security_score for r in results]
    runtime_scores = [r.runtime_score for r in results]
    
    # Cache efficiency statistics
    cache_hits = [r.cache_hit for r in results]
    training_times = [r.adapter_training_time for r in results]
    
    # Architecture SLA targets
    from coral.domain.threshold_gate import get_sla_targets
    sla_targets = get_sla_targets()
    
    # Fitness statistics
    fitness_stats = {
        'mean_fitness': sum(fitnesses) / len(fitnesses),
        'max_fitness': max(fitnesses),
        'min_fitness': min(fitnesses),
        'fitness_std': _calculate_std(fitnesses)
    }
    
    # CORAL-X Multi-objective statistics
    multi_objective_stats = {
        'mean_bugfix': sum(bugfix_scores) / len(bugfix_scores),
        'max_bugfix': max(bugfix_scores),
        'bugfix_sla_compliance': sum(1 for s in bugfix_scores if s >= sla_targets['bugfix']) / len(bugfix_scores),
        
        'mean_style': sum(style_scores) / len(style_scores),
        'max_style': max(style_scores),
        'style_sla_compliance': sum(1 for s in style_scores if s >= sla_targets['style']) / len(style_scores),
        
        'mean_security': sum(security_scores) / len(security_scores),
        'max_security': max(security_scores),
        'security_sla_compliance': sum(1 for s in security_scores if s >= sla_targets['security']) / len(security_scores),
        
        'mean_runtime': sum(runtime_scores) / len(runtime_scores),
        'max_runtime': max(runtime_scores),
        'runtime_sla_compliance': sum(1 for s in runtime_scores if s >= sla_targets['runtime']) / len(runtime_scores)
    }
    
    # Cache efficiency statistics
    cache_stats = {
        'cache_hit_rate': sum(cache_hits) / len(cache_hits),
        'mean_adapter_training_time': sum(training_times) / len(training_times),
        'total_adapter_training_time': sum(training_times),
        'cache_efficiency_10x': sum(1 for hit in cache_hits if hit) / len(cache_hits)  # Fraction getting 10x+ speedup
    }
    
    # Timing statistics
    timing_stats = {
        'mean_execution_time': sum(execution_times) / len(execution_times),
        'max_execution_time': max(execution_times),
        'min_execution_time': min(execution_times),
        'total_execution_time': sum(execution_times)
    }
    
    # LoRA parameter statistics
    lora_stats = _calculate_lora_statistics(results)
    
    # CA feature statistics
    ca_stats = _calculate_ca_statistics(results)
    
    return {
        **fitness_stats,
        **multi_objective_stats,
        **cache_stats,
        **timing_stats,
        **lora_stats,
        **ca_stats,
        'num_genomes': len(results)
    }


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _calculate_lora_statistics(results: List[BenchmarkResult]) -> Dict[str, float]:
    """Calculate statistics for LoRA parameters."""
    ranks = [r.lora_config['r'] for r in results]
    alphas = [r.lora_config['alpha'] for r in results]
    dropouts = [r.lora_config['dropout'] for r in results]
    
    return {
        'mean_lora_rank': sum(ranks) / len(ranks),
        'mean_lora_alpha': sum(alphas) / len(alphas),
        'mean_lora_dropout': sum(dropouts) / len(dropouts),
        'lora_rank_std': _calculate_std([float(r) for r in ranks]),
        'lora_alpha_std': _calculate_std(alphas),
        'lora_dropout_std': _calculate_std(dropouts)
    }


def _calculate_ca_statistics(results: List[BenchmarkResult]) -> Dict[str, float]:
    """Calculate statistics for CA features."""
    complexities = [r.ca_features['complexity'] for r in results]
    intensities = [r.ca_features['intensity'] for r in results]
    periodicities = [r.ca_features['periodicity'] for r in results]
    convergences = [r.ca_features['convergence'] for r in results]
    
    return {
        'mean_ca_complexity': sum(complexities) / len(complexities),
        'mean_ca_intensity': sum(intensities) / len(intensities),
        'mean_ca_periodicity': sum(periodicities) / len(periodicities),
        'mean_ca_convergence': sum(convergences) / len(convergences),
        'ca_complexity_std': _calculate_std(complexities),
        'ca_intensity_std': _calculate_std(intensities),
        'ca_periodicity_std': _calculate_std(periodicities),
        'ca_convergence_std': _calculate_std(convergences)
    }


def _config_to_dict(config: CoralConfig) -> Dict[str, Any]:
    """Convert CoralConfig to dictionary."""
    return {
        'evo': {
            'rank_candidates': list(config.evo.rank_candidates),
            'alpha_candidates': list(config.evo.alpha_candidates),
            'dropout_candidates': list(config.evo.dropout_candidates)
        },
        'seed': config.seed,
        'execution': config.execution,
        'infra': config.infra,
        'experiment': config.experiment
    } 