"""
Pure experiment domain logic.
Contains immutable data structures and pure functions for experiments.
"""
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from coral.domain.genome import Genome
from coral.domain.neat import Population
from coral.domain.ca import CASeed, evolve
from coral.domain.feature_extraction import extract_features
from coral.domain.mapping import map_features_to_lora_config, EvolutionConfig
import numpy as np
from random import Random


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration."""
    population_size: int
    generations: int
    seed: int
    evolution_config: EvolutionConfig


@dataclass(frozen=True)
class ExperimentResults:
    """Immutable experiment results."""
    final_population: Population
    experiment_time: float
    best_fitness: float
    generations_completed: int
    success: bool
    error_message: str = ""


def create_experiment_config(raw_config: Dict[str, Any]) -> ExperimentConfig:
    """Pure function to create experiment config from raw dictionary - FAIL-FAST."""
    # Validate required sections
    if 'evo' not in raw_config:
        raise ValueError("FAIL-FAST: 'evo' section missing from configuration")
    if 'execution' not in raw_config:
        raise ValueError("FAIL-FAST: 'execution' section missing from configuration")
    
    evo_raw = raw_config['evo']
    execution_raw = raw_config['execution']
    
    # Validate required evolution parameters
    required_evo_fields = ['rank_candidates', 'alpha_candidates', 'dropout_candidates']
    for field in required_evo_fields:
        if field not in evo_raw:
            raise ValueError(f"FAIL-FAST: '{field}' missing from evolution configuration")
    
    # Validate required execution parameters
    required_exec_fields = ['population_size', 'generations']
    for field in required_exec_fields:
        if field not in execution_raw:
            raise ValueError(f"FAIL-FAST: '{field}' missing from execution configuration")
    
    if 'seed' not in raw_config:
        raise ValueError("FAIL-FAST: 'seed' missing from configuration")
    
    evolution_config = EvolutionConfig(
        rank_candidates=tuple(evo_raw['rank_candidates']),
        alpha_candidates=tuple(evo_raw['alpha_candidates']),
        dropout_candidates=tuple(evo_raw['dropout_candidates'])
    )
    
    return ExperimentConfig(
        population_size=execution_raw['population_size'],
        generations=execution_raw['generations'],
        seed=raw_config['seed'],
        evolution_config=evolution_config
    )


def create_initial_population(config: ExperimentConfig, diversity_strength: float = 1.0, raw_config: Dict[str, Any] = None, run_id: str = None) -> Population:
    """Pure function to create initial population with configurable diversity."""
    rng = Random(config.seed)
    genomes = []
    
    print(f"🧬 CREATING INITIAL POPULATION")
    print(f"   • Population size: {config.population_size}")
    print(f"   • Diversity strength: {diversity_strength:.2f}")
    print(f"   • Run ID: {run_id or 'None'}")
    
    for i in range(config.population_size):
        # Create unique genome ID
        genome_id = f"gen0_genome{i:04d}"
        
        # FIXED: Ensure each genome gets unique random state
        genome_rng = Random(config.seed + i * 1000)  # Large offset for distinctness
        np.random.seed(config.seed + i * 1000)  # Set numpy global seed per genome
        
        # Create diverse CA seed with proper randomization
        grid_size = (8, 8)
        initial_grid = np.random.randint(0, 2, grid_size, dtype=int)
        rule = genome_rng.randint(1, 255)
        steps = genome_rng.randint(5, 20)
        
        ca_seed = CASeed(grid=initial_grid, rule=rule, steps=steps)
        
        # Generate features and map to LoRA config with dynamic diversity
        history = evolve(ca_seed)
        features = extract_features(history)
        
        # ENHANCED: Add genome-specific entropy to mapping for guaranteed diversity
        config_dict = {
            'evo': {
                'rank_candidates': list(config.evolution_config.rank_candidates),
                'alpha_candidates': list(config.evolution_config.alpha_candidates),
                'dropout_candidates': list(config.evolution_config.dropout_candidates),
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
                'diversity': {
                    'mode': 'adaptive',
                    'base_strength': diversity_strength,
                    'max_strength': 2.0,
                    'min_strength': 0.3,
                    'cache_threshold': 0.8,
                    'plateau_threshold': 0.05,
                    'plateau_window': 3,
                    'genome_entropy': i  # ADDED: Genome-specific entropy for diversity
                }
            },
            # 🔥 FIX: Pass through adapter_type from raw config
            'adapter_type': raw_config.get('adapter_type', 'lora') if raw_config else 'lora'
        }
        
        # Apply dynamic diversity strength to LoRA mapping with genome index for guaranteed diversity
        lora_config = map_features_to_lora_config(features, config_dict, diversity_strength, i)
        
        genome = Genome(seed=ca_seed, lora_cfg=lora_config, id=genome_id, run_id=run_id)
        genomes.append(genome)
        
        # Debug: Show first few genome details for verification
        if i < 3:
            print(f"   • Genome {i}: grid_hash={hash(initial_grid.tobytes())}, rule={rule}, steps={steps}")
            print(f"     LoRA: r={lora_config.r}, α={lora_config.alpha}, dropout={lora_config.dropout}")
    
    # Show diversity analysis for initial population
    lora_signatures = set()
    ca_signatures = set()
    for genome in genomes:
        lora_sig = f"r{genome.lora_cfg.r}_a{genome.lora_cfg.alpha}_d{genome.lora_cfg.dropout}"
        ca_sig = f"rule{genome.seed.rule}_steps{genome.seed.steps}_grid{hash(genome.seed.grid.tobytes())}"
        lora_signatures.add(lora_sig)
        ca_signatures.add(ca_sig)
    
    lora_diversity = len(lora_signatures) / len(genomes) * 100
    ca_diversity = len(ca_signatures) / len(genomes) * 100
    cache_efficiency = len(genomes) / len(lora_signatures)
    
    print(f"   • Unique CA seeds: {len(ca_signatures)}/{len(genomes)} ({ca_diversity:.1f}%)")
    print(f"   • Unique LoRA configs: {len(lora_signatures)}/{len(genomes)} ({lora_diversity:.1f}%)")
    print(f"   • Expected cache efficiency: {cache_efficiency:.1f}x")
    
    # FAIL-FAST: Verify diversity was achieved
    if len(lora_signatures) < max(2, len(genomes) // 16):  # At least 1/16th should be unique
        raise RuntimeError(
            f"FAIL-FAST: Insufficient LoRA diversity in initial population. "
            f"Only {len(lora_signatures)} unique configs from {len(genomes)} genomes. "
            f"Check CA → LoRA mapping function for diversity issues."
        )
    
    return Population(tuple(genomes))


def calculate_experiment_metrics(population: Population, start_time: float, end_time: float) -> Tuple[float, float]:
    """Pure function to calculate experiment metrics."""
    experiment_time = end_time - start_time
    
    best_fitness = 0.0
    if population.size() > 0:
        try:
            best_genome = population.best()
            if best_genome.has_multi_scores():
                scores = best_genome.multi_scores
                best_fitness = (scores.bugfix + scores.style + scores.security + scores.runtime) / 4.0
            else:
                best_fitness = best_genome.fitness if hasattr(best_genome, 'fitness') else 0.0
        except:
            best_fitness = 0.0
    
    return experiment_time, best_fitness


def create_experiment_result(population: Population, start_time: float, end_time: float, 
                           generations_completed: int, success: bool, error_message: str = "") -> ExperimentResults:
    """Pure function to create experiment results."""
    experiment_time, best_fitness = calculate_experiment_metrics(population, start_time, end_time)
    
    return ExperimentResults(
        final_population=population,
        experiment_time=experiment_time,
        best_fitness=best_fitness,
        generations_completed=generations_completed,
        success=success,
        error_message=error_message
    ) 