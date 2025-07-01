###############################################################################
# Dashboard State — Pure Mathematical Functions (Domain Category)
# All dashboard state calculations - NO side effects, pure functions only
###############################################################################
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from pathlib import Path

from .genome import Genome, MultiObjectiveScores
from .neat import Population
from .ca import CASeed
from .feature_extraction import CAFeatures
from .mapping import LoRAConfig


@dataclass(frozen=True)
class DashboardState:
    """Immutable dashboard state snapshot."""
    timestamp: float
    run_id: str
    
    # Evolution Progress
    current_generation: int
    max_generations: int
    population_size: int
    completion_percentage: float
    
    # Performance Metrics
    best_scores: MultiObjectiveScores
    fitness_progression: List[float]
    
    # Adapter Status
    adapters_trained: int
    cache_hit_rate: float
    training_rate: float
    
    # Genetic Operations
    crossover_count: int
    mutation_count: int
    crossover_success_rate: float
    mutation_success_rate: float
    
    # CA Metrics
    avg_ca_complexity: float
    avg_ca_intensity: float
    avg_ca_period: float
    avg_ca_convergence: float
    
    # Emergent Behavior
    emergent_active: bool
    total_behaviors: int
    detection_rate: float
    recent_behaviors: List[str]
    
    # Infrastructure
    models_cached: int
    dataset_files: int
    modal_status: str
    gpu_type: str
    
    # Configuration
    cheap_knobs: Dict[str, Tuple[float, float]]
    threshold_progress: Dict[str, float]
    
    # Queue Status (Category Theory Queue System)
    queue_status: Dict[str, int]
    
    # Activity Status
    status: str  # 'starting', 'evolving', 'completed', 'failed'
    current_activity: str
    elapsed_time: float


def calculate_evolution_progress(generation: int, max_generations: int, 
                               population: Optional[Population]) -> Dict[str, Any]:
    """Calculate evolution progress metrics — pure function."""
    completion = (generation / max_generations) * 100 if max_generations > 0 else 0.0
    
    pop_size = population.size() if population else 0
    evaluated_count = len([g for g in population.genomes if g.is_evaluated()]) if population else 0
    
    return {
        'completion_percentage': completion,
        'population_size': pop_size,
        'evaluated_count': evaluated_count,
        'progress_bar': _create_progress_bar(completion)
    }


def calculate_performance_metrics(population: Optional[Population]) -> Dict[str, Any]:
    """Calculate multi-objective performance metrics — pure function."""
    if not population or population.size() == 0:
        return _empty_performance_metrics()
    
    try:
        best = population.best()
        if not best.has_multi_scores():
            return _empty_performance_metrics()
        
        scores = best.multi_scores
        
        # Calculate grades based on score ranges
        grades = {
            'bugfix': _score_to_grade(scores.bugfix),
            'style': _score_to_grade(scores.style),
            'security': _score_to_grade(scores.security),
            'runtime': _score_to_grade(scores.runtime),
            'syntax': _score_to_grade(scores.syntax) if hasattr(scores, 'syntax') else 'N/A'
        }
        
        return {
            'best_scores': scores,
            'grades': grades,
            'overall_fitness': best.fitness
        }
        
    except Exception:
        return _empty_performance_metrics()


def calculate_ca_metrics(population: Optional[Population]) -> Dict[str, float]:
    """Calculate aggregated CA feature metrics — pure function."""
    if not population or population.size() == 0:
        return {'complexity': 0.0, 'intensity': 0.0, 'periodicity': 0.0, 'convergence': 0.0}
    
    complexities = []
    intensities = []
    periodicities = []
    convergences = []
    
    for genome in population.genomes:
        if hasattr(genome, 'ca_features') and genome.ca_features:
            complexities.append(genome.ca_features.complexity)
            intensities.append(genome.ca_features.intensity)
            periodicities.append(genome.ca_features.periodicity)
            convergences.append(genome.ca_features.convergence)
    
    if not complexities:
        return {'complexity': 0.0, 'intensity': 0.0, 'periodicity': 0.0, 'convergence': 0.0}
    
    return {
        'complexity': np.mean(complexities),
        'intensity': np.mean(intensities),
        'periodicity': np.mean(periodicities),
        'convergence': np.mean(convergences)
    }


def calculate_genetic_metrics(genetic_stats: Dict[int, Dict[str, Any]], 
                            current_generation: int) -> Dict[str, Any]:
    """Calculate genetic operations metrics — pure function."""
    if not genetic_stats or current_generation not in genetic_stats:
        return {
            'crossover_count': 0,
            'mutation_count': 0,
            'crossover_success_rate': 0.0,
            'mutation_success_rate': 0.0
        }
    
    gen_stats = genetic_stats[current_generation]
    
    crossovers = gen_stats.get('crossovers', 0)
    mutations = gen_stats.get('mutations', 0)
    successful_crossovers = gen_stats.get('successful_crossovers', 0)
    successful_mutations = gen_stats.get('successful_mutations', 0)
    
    crossover_rate = (successful_crossovers / crossovers * 100) if crossovers > 0 else 0.0
    mutation_rate = (successful_mutations / mutations * 100) if mutations > 0 else 0.0
    
    return {
        'crossover_count': crossovers,
        'mutation_count': mutations,
        'crossover_success_rate': crossover_rate,
        'mutation_success_rate': mutation_rate
    }


def calculate_threshold_progress(current_scores: Optional[MultiObjectiveScores],
                               generation: int, max_generations: int,
                               base_thresholds: Dict[str, float],
                               max_thresholds: Dict[str, float]) -> Dict[str, float]:
    """Calculate progress towards dynamic thresholds — pure function."""
    from .threshold_gate import calculate_sigma
    
    if not current_scores:
        return {obj: 0.0 for obj in base_thresholds.keys()}
    
    # Calculate current threshold targets
    sigma = calculate_sigma(generation, max_generations, "sigmoid")
    
    scores_dict = current_scores.to_dict()
    progress = {}
    
    for objective in base_thresholds.keys():
        base = base_thresholds[objective]
        max_thresh = max_thresholds[objective]
        current_target = base + sigma * (max_thresh - base)
        
        current_score = scores_dict.get(objective, 0.0)
        progress_pct = (current_score / current_target * 100) if current_target > 0 else 0.0
        progress[objective] = min(100.0, progress_pct)
    
    return progress


def extract_cheap_knobs_ranges(config: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Extract cheap knobs configuration ranges — pure function."""
    cheap_knobs = config.get('cheap_knobs', {})
    
    def extract_range(range_list):
        if isinstance(range_list, list) and len(range_list) >= 2:
            return (min(range_list), max(range_list))
        return (0.0, 1.0)
    
    return {
        'temperature': extract_range(cheap_knobs.get('temperature_range', [0.2, 0.8])),
        'top_p': extract_range(cheap_knobs.get('top_p_range', [0.75, 0.9])),
        'top_k': extract_range(cheap_knobs.get('top_k_range', [20, 50])),
        'repetition_penalty': extract_range(cheap_knobs.get('repetition_penalty_range', [1.0, 1.2])),
        'max_tokens': extract_range(cheap_knobs.get('max_tokens_range', [150, 300]))
    }


def calculate_training_rate(adapters_trained: int, elapsed_hours: float) -> float:
    """Calculate adapter training rate — pure function."""
    return adapters_trained / max(elapsed_hours, 0.1)


def create_dashboard_state(
    # Evolution data
    current_generation: int,
    max_generations: int,
    population: Optional[Population],
    
    # Tracking data
    genetic_stats: Dict[int, Dict[str, Any]],
    emergent_stats: Dict[str, Any],
    
    # Infrastructure data
    adapters_trained: int,
    cache_hit_rate: float,
    models_cached: int,
    dataset_files: int,
    
    # Configuration
    config: Dict[str, Any],
    
    # Runtime info
    run_id: str,
    start_time: float,
    status: str,
    current_activity: str,
    
    # Queue status (Category Theory)
    queue_status: Optional[Dict[str, int]] = None
    
) -> DashboardState:
    """Create complete dashboard state — pure function composition."""
    
    # Calculate all metrics using pure functions
    evolution_metrics = calculate_evolution_progress(current_generation, max_generations, population)
    performance_metrics = calculate_performance_metrics(population)
    ca_metrics = calculate_ca_metrics(population)
    genetic_metrics = calculate_genetic_metrics(genetic_stats, current_generation)
    
    # Extract configuration data
    cheap_knobs = extract_cheap_knobs_ranges(config)
    
    # Calculate threshold progress
    base_thresholds = config.get('threshold', {}).get('base_thresholds', {})
    max_thresholds = config.get('threshold', {}).get('max_thresholds', {})
    threshold_progress = calculate_threshold_progress(
        performance_metrics.get('best_scores'),
        current_generation, max_generations,
        base_thresholds, max_thresholds
    )
    
    # Calculate time metrics
    elapsed_time = datetime.now().timestamp() - start_time
    training_rate = calculate_training_rate(adapters_trained, elapsed_time / 3600)
    
    # Extract emergent behavior data
    emergent_active = emergent_stats.get('active', False)
    total_behaviors = emergent_stats.get('total_behaviors', 0)
    total_evaluations = emergent_stats.get('total_evaluations', 1)
    detection_rate = (total_behaviors / total_evaluations * 100) if total_evaluations > 0 else 0.0
    recent_behaviors = emergent_stats.get('recent_behaviors', [])
    
    # Infrastructure status
    modal_status = "READY" if models_cached > 0 and dataset_files > 0 else "NOT READY"
    gpu_type = config.get('infra', {}).get('modal', {}).get('functions', {}).get('evaluate_genome', {}).get('gpu', 'A100-40GB')
    
    return DashboardState(
        timestamp=datetime.now().timestamp(),
        run_id=run_id,
        
        # Evolution Progress
        current_generation=current_generation,
        max_generations=max_generations,
        population_size=evolution_metrics['population_size'],
        completion_percentage=evolution_metrics['completion_percentage'],
        
        # Performance Metrics
        best_scores=performance_metrics.get('best_scores'),
        fitness_progression=[],  # TODO: Extract from history
        
        # Adapter Status
        adapters_trained=adapters_trained,
        cache_hit_rate=cache_hit_rate,
        training_rate=training_rate,
        
        # Genetic Operations
        crossover_count=genetic_metrics['crossover_count'],
        mutation_count=genetic_metrics['mutation_count'],
        crossover_success_rate=genetic_metrics['crossover_success_rate'],
        mutation_success_rate=genetic_metrics['mutation_success_rate'],
        
        # CA Metrics
        avg_ca_complexity=ca_metrics['complexity'],
        avg_ca_intensity=ca_metrics['intensity'],
        avg_ca_period=ca_metrics['periodicity'],
        avg_ca_convergence=ca_metrics['convergence'],
        
        # Emergent Behavior
        emergent_active=emergent_active,
        total_behaviors=total_behaviors,
        detection_rate=detection_rate,
        recent_behaviors=recent_behaviors,
        
        # Infrastructure
        models_cached=models_cached,
        dataset_files=dataset_files,
        modal_status=modal_status,
        gpu_type=gpu_type,
        
        # Configuration
        cheap_knobs=cheap_knobs,
        threshold_progress=threshold_progress,
        
        # Queue Status
        queue_status=queue_status or {},
        
        # Activity Status
        status=status,
        current_activity=current_activity,
        elapsed_time=elapsed_time
    )


# Private helper functions

def _create_progress_bar(percentage: float, width: int = 20) -> str:
    """Create ASCII progress bar — pure function."""
    filled = int(percentage / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage:.1f}%"


def _score_to_grade(score: float) -> str:
    """Convert score to letter grade — pure function."""
    if score >= 0.9:
        return "A+"
    elif score >= 0.85:
        return "A"
    elif score >= 0.8:
        return "A-"
    elif score >= 0.75:
        return "B+"
    elif score >= 0.7:
        return "B"
    elif score >= 0.65:
        return "B-"
    elif score >= 0.6:
        return "C"
    else:
        return "D"


def _empty_performance_metrics() -> Dict[str, Any]:
    """Return empty performance metrics — pure function."""
    return {
        'best_scores': None,
        'grades': {'bugfix': 'N/A', 'style': 'N/A', 'security': 'N/A', 'runtime': 'N/A', 'syntax': 'N/A'},
        'overall_fitness': 0.0
    } 