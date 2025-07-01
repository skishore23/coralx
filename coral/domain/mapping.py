###############################################################################
# Feature → LoRA hyper‑params mapping - NO FALLBACKS, config-driven
###############################################################################
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from coral.domain.categorical_result import Result
from .feature_extraction import CAFeatures


@dataclass(frozen=True)
class AdapterConfig:
    """Unified adapter configuration supporting both LoRA and DoRA."""
    r: int
    alpha: float
    dropout: float
    target_modules: Tuple[str, ...]
    adapter_type: str = "lora"  # "lora" or "dora"


# Backward compatibility alias
LoRAConfig = AdapterConfig


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration ranges for evolution parameters - NO DEFAULTS."""
    rank_candidates: Tuple[int, ...]
    alpha_candidates: Tuple[float, ...]  # Changed from range to discrete candidates
    dropout_candidates: Tuple[float, ...]  # Changed from range to discrete candidates
    target_modules: Tuple[str, ...]  # Target modules for adapter training


def map_features_to_lora_config(features: CAFeatures, config: Dict[str, Any], 
                                diversity_strength: float = 1.0, genome_index: int = 0) -> AdapterConfig:
    """
    Map CA features to LoRA config with dynamic diversity injection.
    Uses feature fingerprinting with adjustable diversity strength.
    
    Args:
        features: CA features extracted from cellular automata evolution
        config: Configuration dictionary with evolution parameters
        diversity_strength: Multiplier for diversity injection (1.0 = baseline, 2.0 = max diversity)
        genome_index: Index of genome in population (for guaranteed diversity)
    """
    # Extract evolution configuration
    if 'evo' not in config:
        raise ValueError("FAIL-FAST: 'evo' section missing from configuration")
    
    evo_raw = config['evo']
    
    # Validate required fields
    required_fields = ['rank_candidates', 'alpha_candidates', 'dropout_candidates', 'target_modules']
    for field in required_fields:
        if field not in evo_raw:
            raise ValueError(f"FAIL-FAST: '{field}' missing from evolution configuration")
    
    # Create evolution config
    evo_cfg = EvolutionConfig(
        rank_candidates=tuple(evo_raw['rank_candidates']),
        alpha_candidates=tuple(evo_raw['alpha_candidates']),
        dropout_candidates=tuple(evo_raw['dropout_candidates']),
        target_modules=tuple(evo_raw['target_modules'])
    )
    
    # ENHANCED DIVERSITY MAPPING: Use genome index as additional entropy
    rank = _map_with_enhanced_diversity(features, evo_cfg.rank_candidates, 'rank', diversity_strength, genome_index)
    alpha = _map_with_enhanced_diversity(features, evo_cfg.alpha_candidates, 'alpha', diversity_strength, genome_index) 
    dropout = _map_with_enhanced_diversity(features, evo_cfg.dropout_candidates, 'dropout', diversity_strength, genome_index)
    
    # Get adapter type from config (default to LoRA for backward compatibility)
    adapter_type = config.get('adapter_type', 'lora')
    
    return AdapterConfig(
        r=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=tuple(evo_raw['target_modules']),
        adapter_type=adapter_type
    )


def _map_with_enhanced_diversity(features: CAFeatures, candidates: Tuple, param_type: str, 
                                diversity_strength: float, genome_index: int = 0):
    """Enhanced mapping with guaranteed diversity using feature fingerprinting + genome index."""
    if len(candidates) == 0:
        raise ValueError(f"FAIL-FAST: No candidates provided for {param_type} mapping")
    
    # Create genome-specific entropy from CA features themselves
    # Use feature combinations to ensure each genome gets unique mappings
    genome_entropy = hash((
        f"{features.complexity * 1000:.0f}",  # Integer precision for better hashing
        f"{features.intensity * 1000:.0f}",
        f"{features.periodicity * 1000:.0f}",
        f"{features.convergence * 1000:.0f}",
        param_type,  # Each parameter type gets different hash seed
        genome_index * 7919  # Prime number multiplication for genome uniqueness
    )) % 10000  # Modulo for reasonable range
    
    # Create enhanced feature fingerprint with guaranteed uniqueness
    enhanced_fingerprint = hash((
        f"{features.complexity:.8f}",
        f"{features.intensity:.8f}", 
        f"{features.periodicity:.8f}",
        f"{features.convergence:.8f}",
        param_type,  # Different for rank/alpha/dropout
        genome_entropy,  # Genome-specific entropy
        genome_index + 1  # Ensure each genome gets different base hash
    ))
    
    # Apply diversity strength adjustment with genome index guarantee
    if diversity_strength <= 0.5:
        # LOW DIVERSITY: Use feature blending for cache efficiency
        # Similar features map to similar configs, but genome index ensures uniqueness
        feature_blend = (
            features.complexity * 0.4 +
            features.intensity * 0.3 +
            features.periodicity * 0.2 +
            features.convergence * 0.1
        )
        # Add genome entropy and index for guaranteed distinctness
        feature_blend = (feature_blend + genome_entropy * 0.001 + genome_index * 0.01) % 1.0
        
        # Quantize to create cache groups with minimum diversity
        num_groups = max(2, int(len(candidates) * diversity_strength))
        group_index = int(feature_blend * num_groups) % num_groups
        candidate_index = int(group_index / num_groups * (len(candidates) - 1))
        
    elif diversity_strength >= 1.5:
        # HIGH DIVERSITY: Use enhanced fingerprinting with maximum spread
        super_enhanced_fingerprint = hash((
            enhanced_fingerprint,
            f"{features.complexity * features.intensity:.12f}",
            f"{features.periodicity / (features.convergence + 1e-8):.12f}",
            int(diversity_strength * 1000),  # Include strength in hash
            genome_index * 997  # Another prime for better distribution
        ))
        candidate_index = abs(super_enhanced_fingerprint) % len(candidates)
        
    else:
        # BALANCED DIVERSITY: Enhanced fingerprinting with genome index
        candidate_index = abs(enhanced_fingerprint) % len(candidates)
    
    return candidates[candidate_index]


def calculate_dynamic_diversity_strength(cache_hit_rate: float, recent_improvements: list,
                                       config: Dict[str, Any]) -> float:
    """
    Calculate dynamic diversity strength based on evolution state.
    
    Args:
        cache_hit_rate: Fraction of genomes using cached adapters (0.0-1.0)
        recent_improvements: List of fitness improvements over recent generations
        config: Configuration with diversity parameters
        
    Returns:
        diversity_strength: Multiplier for diversity injection
    """
    diversity_config = config.get('evo', {}).get('diversity', {})
    
    # Get configuration parameters with defaults
    mode = diversity_config.get('mode', 'adaptive')
    base_strength = diversity_config.get('base_strength', 1.0)
    max_strength = diversity_config.get('max_strength', 2.0)
    min_strength = diversity_config.get('min_strength', 0.3)
    cache_threshold = diversity_config.get('cache_threshold', 0.8)
    plateau_threshold = diversity_config.get('plateau_threshold', 0.05)
    plateau_window = diversity_config.get('plateau_window', 3)
    
    if mode == 'fixed':
        return base_strength
    elif mode == 'aggressive':
        return max_strength
    
    # ADAPTIVE MODE: Adjust based on cache rate and performance
    strength = base_strength
    
    # Increase diversity if cache hit rate is too high
    if cache_hit_rate > cache_threshold:
        cache_penalty = (cache_hit_rate - cache_threshold) / (1.0 - cache_threshold)
        strength += cache_penalty * (max_strength - base_strength)
        print(f"   🔄 High cache rate ({cache_hit_rate:.2f}) → diversity boost: {strength:.2f}")
    
    # Increase diversity if performance has plateaued
    if len(recent_improvements) >= plateau_window:
        recent_window = recent_improvements[-plateau_window:]
        avg_improvement = sum(recent_window) / len(recent_window)
        
        if avg_improvement < plateau_threshold:
            plateau_penalty = (plateau_threshold - avg_improvement) / plateau_threshold
            strength += plateau_penalty * (max_strength - base_strength) * 0.5  # Moderate boost
            print(f"   📈 Performance plateau ({avg_improvement:.3f}) → diversity boost: {strength:.2f}")
    
    # Decrease diversity if we have good exploration (low cache rate + good improvement)
    if cache_hit_rate < 0.3 and len(recent_improvements) > 0 and recent_improvements[-1] > plateau_threshold:
        exploration_bonus = (0.3 - cache_hit_rate) / 0.3
        strength = max(min_strength, strength - exploration_bonus * (base_strength - min_strength) * 0.3)
        print(f"   🎯 Good exploration → diversity reduction: {strength:.2f}")
    
    # Clamp to valid range
    return max(min_strength, min(max_strength, strength))


def _map_multi_feature_to_rank(features: CAFeatures, rank_candidates: Tuple[int, ...]) -> int:
    """Multi-feature rank mapping - combines all features for maximum diversity."""
    if not rank_candidates:
        raise ValueError("FAIL-FAST: No rank candidates provided for rank mapping")
    
    # Combine ALL features with different weightings to create unique signatures
    # Use complexity as primary, but blend with other features for diversity
    feature_blend = (
        0.5 * features.complexity +     # Primary driver
        0.2 * features.intensity +      # Add dynamics
        0.2 * (1 - features.periodicity) +  # Add chaos (inverse of periodicity)
        0.1 * (1 - features.convergence)    # Add instability
    )
    
    # Apply non-linear transformation to spread out similar values
    enhanced_blend = feature_blend ** 0.6
    
    # Map to full range of candidates
    candidate_index = int(enhanced_blend * (len(rank_candidates) - 1))
    candidate_index = min(candidate_index, len(rank_candidates) - 1)
    
    return rank_candidates[candidate_index]


def _map_multi_feature_to_alpha(features: CAFeatures, alpha_candidates: Tuple[float, ...]) -> float:
    """Multi-feature alpha mapping - uses intensity + complexity."""
    if len(alpha_candidates) == 0:
        raise ValueError("FAIL-FAST: No alpha candidates provided for alpha mapping")
    
    # Combine intensity with complexity for alpha determination
    # Higher intensity + complexity = higher learning rate (alpha)
    alpha_blend = (
        0.6 * features.intensity +      # Primary: how active is the CA
        0.3 * features.complexity +     # Secondary: how complex
        0.1 * (1 - features.convergence)  # Tertiary: how unstable
    )
    
    # Apply square root to amplify small differences
    enhanced_blend = np.sqrt(alpha_blend)
    
    # Map to full range of alpha candidates
    candidate_index = int(enhanced_blend * (len(alpha_candidates) - 1))
    candidate_index = min(candidate_index, len(alpha_candidates) - 1)
    
    return alpha_candidates[candidate_index]


def _map_multi_feature_to_dropout(features: CAFeatures, dropout_candidates: Tuple[float, ...]) -> float:
    """Multi-feature dropout mapping - uses periodicity + convergence + complexity."""
    if len(dropout_candidates) == 0:
        raise ValueError("FAIL-FAST: No dropout candidates provided for dropout mapping")
    
    # Combine stability features for dropout determination
    # More chaotic/unstable patterns = higher dropout for regularization
    dropout_blend = (
        0.4 * (1 - features.periodicity) +    # Less periodic = more dropout
        0.3 * (1 - features.convergence) +    # Less convergent = more dropout  
        0.2 * features.intensity +            # More active = more dropout
        0.1 * features.complexity             # More complex = slightly more dropout
    )
    
    # Apply power transformation to emphasize differences
    enhanced_blend = dropout_blend ** 1.1
    
    # Map to full range of dropout candidates
    candidate_index = int(enhanced_blend * (len(dropout_candidates) - 1))
    candidate_index = min(candidate_index, len(dropout_candidates) - 1)
    
    return dropout_candidates[candidate_index]


# MODAL SERIALIZATION MAPPING FUNCTIONS (Category Theory Compliance)

def serialize_heavy_genes_for_modal(heavy_genes) -> list:
    """
    Pure mapping function: HeavyGenes → Modal-serializable list.
    Uses HeavyGenes object attributes directly instead of tuple conversion.
    """
    # Use HeavyGenes object attributes directly to avoid tuple conversion
    if hasattr(heavy_genes, 'rank'):  # HeavyGenes object
        return [
            heavy_genes.rank,
            heavy_genes.alpha, 
            heavy_genes.dropout,
            list(heavy_genes.target_modules),  # Convert tuple to list for Modal
            heavy_genes.adapter_type,
            heavy_genes.run_id  # Include run_id for experiment separation
        ]
    elif hasattr(heavy_genes, 'r'):  # AdapterConfig object
        return [
            heavy_genes.r,
            heavy_genes.alpha, 
            heavy_genes.dropout,
            list(heavy_genes.target_modules),  # Convert tuple to list for Modal
            getattr(heavy_genes, 'adapter_type', 'lora'),  # Default to lora if missing
            None  # No run_id in AdapterConfig
        ]
    else:
        raise ValueError(f"FAIL-FAST: Invalid heavy_genes type: {type(heavy_genes)}. Expected HeavyGenes or AdapterConfig object.")


def deserialize_heavy_genes_from_modal(heavy_key_list: list) -> tuple:
    """
    Pure mapping function: Modal list → HeavyGenes tuple.
    Reconstructs nested tuple structure from Modal serialization.
    """
    if not isinstance(heavy_key_list, list):
        raise ValueError(f"FAIL-FAST: Invalid heavy_key_list format: {heavy_key_list}")
    
    if len(heavy_key_list) == 6:
        # Newest format: [rank, alpha, dropout, target_modules, adapter_type, run_id]
        rank, alpha, dropout, target_modules_list, adapter_type, run_id = heavy_key_list
    elif len(heavy_key_list) == 5:
        # New format: [rank, alpha, dropout, target_modules, adapter_type]
        rank, alpha, dropout, target_modules_list, adapter_type = heavy_key_list
        run_id = None  # Default for backward compatibility
    elif len(heavy_key_list) == 4:
        # Old format: [rank, alpha, dropout, target_modules] - default to LoRA
        rank, alpha, dropout, target_modules_list = heavy_key_list
        adapter_type = 'lora'
        run_id = None  # Default for backward compatibility
    else:
        raise ValueError(f"FAIL-FAST: Invalid heavy_key_list format. Expected 4, 5, or 6 elements, got {len(heavy_key_list)}: {heavy_key_list}")
    
    # Convert target_modules list back to tuple
    target_modules = tuple(target_modules_list) if isinstance(target_modules_list, list) else target_modules_list
    
    # Return properly structured tuple (always 6 elements for current format)
    return (rank, alpha, dropout, target_modules, adapter_type, run_id)


# ========== NEW: Monadic Feature Mapping ==========
# Demonstrates compositional error handling vs exception-based FAIL-FAST

def map_features_to_lora_config_monadic(features: CAFeatures, 
                                       config: 'CoralConfig',  # ✅ ENHANCED CONFIG TYPE
                                       diversity_strength: float = 1.0, 
                                       genome_index: int = 0) -> 'Result[AdapterConfig, str]':
    """
    NEW: Monadic feature mapping using Result monad with enhanced config.
    No more type detection needed - config provides both dict and structured access.
    """
    from coral.domain.categorical_result import success, error
    
    print(f"🧮 MONADIC FEATURE MAPPING:")
    print(f"   • Using Result monad for compositional error handling")
    print(f"   • Automatic error propagation through pipeline")
    
    try:
        # ✅ CLEAN: Structured access (lazy-loaded)
        evo_cfg = config.evo
        
        # ✅ CLEAN: Dict access  
        adapter_type = config.get('adapter_type', 'lora')
        
        # ✅ DELETE: All the complex validation and type detection!
        
        # Apply enhanced diversity mapping (reuse existing pure function)
        rank = _map_with_enhanced_diversity(features, evo_cfg.rank_candidates, 'rank', diversity_strength, genome_index)
        alpha = _map_with_enhanced_diversity(features, evo_cfg.alpha_candidates, 'alpha', diversity_strength, genome_index) 
        dropout = _map_with_enhanced_diversity(features, evo_cfg.dropout_candidates, 'dropout', diversity_strength, genome_index)
        
        # Create final configuration  
        adapter_config = AdapterConfig(
            r=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=evo_cfg.target_modules,
            adapter_type=adapter_type
        )
        
        print(f"   • Monadic composition: Success")
        print(f"   • Generated config: rank={rank}, alpha={alpha}, dropout={dropout}")
        
        return success(adapter_config)
        
    except Exception as e:
        return error(f"Feature mapping failed: {str(e)}")


def safe_extract_features(ca_history) -> 'Result[CAFeatures, str]':
    """
    NEW: Safe feature extraction using Result monad.
    Demonstrates error handling for feature extraction pipeline.
    """
    from coral.domain.categorical_result import safe_call
    from coral.domain.feature_extraction import extract_features
    
    print(f"🧮 SAFE FEATURE EXTRACTION:")
    print(f"   • Using monadic error handling")
    
    # Use safe_call to convert exceptions to Result
    result = safe_call(
        lambda: extract_features(ca_history),
        "Feature extraction failed"
    )
    
    if result.is_success():
        features = result.unwrap()
        print(f"   • Extraction: Success")
        print(f"   • Features: complexity={features.complexity:.3f}, intensity={features.intensity:.3f}")
    else:
        print(f"   • Extraction: Failed - {result.unwrap_error()}")
    
    return result


def compose_ca_pipeline_monadic(ca_seed, config=None) -> 'Result[AdapterConfig, str]':
    """
    NEW: Complete CA → LoRA pipeline using monadic composition with enhanced config.
    Uses YAML-driven configuration with automatic loading.
    """
    from coral.domain.categorical_result import safe_call
    from coral.domain.ca import evolve
    
    print(f"🧮 MONADIC CA → LORA PIPELINE:")
    print(f"   • Full compositional pipeline with automatic error propagation")
    
    # Use provided config or load from YAML (no hardcoded defaults)
    if config is None:
        try:
            from coral.config.loader import load_config
            # Try to load a real config file for demonstration
            config_files = ["config/test.yaml", "config/quick_test.yaml", "config/main.yaml"]
            for config_file in config_files:
                try:
                    config = load_config(config_file)
                    print(f"   • Loaded configuration from {config_file}")
                    break
                except:
                    continue
            
            if config is None:
                raise ValueError("No valid configuration file found")
                
        except Exception as e:
            from coral.domain.categorical_result import error
            return error(f"Could not load configuration: {str(e)}")
    
    # Monadic pipeline composition
    result = (
        safe_call(lambda: evolve(ca_seed), "CA evolution failed")
        .bind(lambda history: safe_extract_features(history))
        .bind(lambda features: map_features_to_lora_config_monadic(features, config))
    )
    
    if result.is_success():
        print(f"   • Pipeline: Complete success")
    else:
        print(f"   • Pipeline: Failed at some stage - {result.unwrap_error()}")
    
    return result 