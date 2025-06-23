###############################################################################
# Cheap Knobs - Runtime Generation Parameters from CA Features
# LOOP 2: Controls HOW the model generates (inference-time parameters)
###############################################################################
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from .feature_extraction import CAFeatures


@dataclass(frozen=True)
class CheapKnobs:
    """
    Runtime generation parameters derived from CA features.
    These can be adjusted at inference time without retraining adapters.
    
    LOOP 2: Controls HOW the model generates
    - temperature: Randomness/creativity in generation
    - top_p: Nucleus sampling threshold  
    - top_k: Top-k sampling limit
    - repetition_penalty: Penalty for repetitive text
    - max_new_tokens: Maximum tokens to generate
    - do_sample: Whether to use sampling vs greedy
    """
    temperature: float          # 0.1-1.5: Higher = more creative/random
    top_p: float               # 0.7-0.95: Nucleus sampling threshold
    top_k: int                 # 10-100: Top-k sampling limit
    repetition_penalty: float  # 1.0-1.3: Penalty for repetition
    max_new_tokens: int        # 50-500: Token generation limit
    do_sample: bool           # True for sampling, False for greedy


def map_ca_features_to_cheap_knobs(features: CAFeatures, 
                                  base_config: Dict[str, Any] = None) -> CheapKnobs:
    """
    Map CA features to runtime generation parameters.
    Pure function: CAFeatures → CheapKnobs
    
    CORE TWO-LOOP PRINCIPLE:
    - Heavy genes (LoRA rank/alpha/dropout) → WHAT the model learns
    - Cheap knobs (temperature/top_p/etc) → HOW the model generates
    
    Args:
        features: CA features from cellular automata evolution
        base_config: Configuration with parameter ranges (REQUIRED)
        
    Returns:
        CheapKnobs with runtime generation parameters
    """
    # FAIL-FAST: No default fallbacks - config must be provided
    if base_config is None:
        raise ValueError("FAIL-FAST: base_config required for cheap knobs mapping - no defaults provided")
    
    # FAIL-FAST: All parameter ranges must be specified  
    required_ranges = ['temperature_range', 'top_p_range', 'top_k_range', 'repetition_penalty_range', 'max_tokens_range']
    for range_key in required_ranges:
        if range_key not in base_config:
            raise ValueError(f"FAIL-FAST: {range_key} missing from cheap knobs config - two-loop architecture requires explicit ranges")
    
    temp_range = base_config['temperature_range']
    top_p_range = base_config['top_p_range'] 
    top_k_range = base_config['top_k_range']
    rep_penalty_range = base_config['repetition_penalty_range']
    token_range = base_config['max_tokens_range']
    
    # Map complexity to temperature (more complex CA → higher temperature)
    # Complex patterns suggest need for creative/diverse generation
    temperature = _map_complexity_to_temperature(features.complexity, temp_range)
    
    # Map intensity to top_p (more active CA → higher top_p)  
    # Active patterns suggest broader token sampling
    top_p = _map_intensity_to_top_p(features.intensity, top_p_range)
    
    # Map periodicity to repetition penalty (more periodic → higher penalty)
    # Periodic patterns suggest need to avoid repetitive generation
    repetition_penalty = _map_periodicity_to_repetition_penalty(features.periodicity, rep_penalty_range)
    
    # Map convergence to top_k (more convergent → lower top_k)
    # Convergent patterns suggest focused/precise generation
    top_k = _map_convergence_to_top_k(features.convergence, top_k_range)
    
    # Map combined features to token limit
    max_new_tokens = _map_combined_to_token_limit(features, token_range)
    
    # Determine sampling strategy based on overall feature profile
    do_sample = _should_use_sampling(features)
    
    return CheapKnobs(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample
    )


def _map_complexity_to_temperature(complexity: float, temp_range: tuple) -> float:
    """
    Map CA complexity to generation temperature.
    
    Logic: Complex CA patterns suggest need for creative/diverse code generation
    - Low complexity (0.0-0.3) → Low temperature (conservative generation)
    - Medium complexity (0.3-0.7) → Medium temperature (balanced)  
    - High complexity (0.7-1.0) → High temperature (creative generation)
    """
    min_temp, max_temp = temp_range
    
    # Apply non-linear mapping to emphasize high complexity
    enhanced_complexity = complexity ** 0.8
    
    # Map to temperature range with minimum threshold
    temperature = min_temp + enhanced_complexity * (max_temp - min_temp)
    
    return round(temperature, 2)


def _map_intensity_to_top_p(intensity: float, top_p_range: tuple) -> float:
    """
    Map CA intensity to nucleus sampling threshold.
    
    Logic: High intensity CA suggests dynamic/active code generation
    - Low intensity (0.0-0.3) → Lower top_p (focused sampling)
    - Medium intensity (0.3-0.7) → Medium top_p (balanced)
    - High intensity (0.7-1.0) → Higher top_p (broad sampling)
    """
    min_top_p, max_top_p = top_p_range
    
    # Apply square root to spread out low intensity values
    enhanced_intensity = np.sqrt(intensity)
    
    # Map to top_p range
    top_p = min_top_p + enhanced_intensity * (max_top_p - min_top_p)
    
    return round(top_p, 2)


def _map_periodicity_to_repetition_penalty(periodicity: float, penalty_range: tuple) -> float:
    """
    Map CA periodicity to repetition penalty.
    
    Logic: Periodic CA patterns suggest tendency toward repetition
    - Low periodicity (0.0-0.3) → Low penalty (allow some repetition)
    - Medium periodicity (0.3-0.7) → Medium penalty (balanced)
    - High periodicity (0.7-1.0) → High penalty (strong anti-repetition)
    """
    min_penalty, max_penalty = penalty_range
    
    # Apply power transformation to emphasize high periodicity
    enhanced_periodicity = periodicity ** 1.2
    
    # Map to penalty range
    penalty = min_penalty + enhanced_periodicity * (max_penalty - min_penalty)
    
    return round(penalty, 2)


def _map_convergence_to_top_k(convergence: float, top_k_range: tuple) -> int:
    """
    Map CA convergence to top-k sampling limit.
    
    Logic: Convergent CA suggests focused/precise generation
    - Low convergence (0.0-0.3) → High top_k (broad sampling)
    - Medium convergence (0.3-0.7) → Medium top_k (balanced)
    - High convergence (0.7-1.0) → Low top_k (focused sampling)
    """
    min_top_k, max_top_k = top_k_range
    
    # Invert convergence (high convergence → low top_k)
    inverted_convergence = 1.0 - convergence
    
    # Apply square root for smoother distribution
    enhanced_convergence = np.sqrt(inverted_convergence)
    
    # Map to top_k range
    top_k = min_top_k + enhanced_convergence * (max_top_k - min_top_k)
    
    return int(round(top_k))


def _map_combined_to_token_limit(features: CAFeatures, token_range: tuple) -> int:
    """
    Map combined CA features to maximum token generation limit.
    
    Logic: Complex, active patterns suggest need for longer generations
    """
    min_tokens, max_tokens = token_range
    
    # Combine features that suggest need for longer generation
    generation_need = (
        0.4 * features.complexity +      # Complex problems need more tokens
        0.3 * features.intensity +       # Active patterns need more exploration
        0.2 * (1 - features.convergence) + # Non-convergent needs more attempts
        0.1 * (1 - features.periodicity)   # Non-periodic needs more diversity
    )
    
    # Apply non-linear scaling
    enhanced_need = generation_need ** 0.9
    
    # Map to token range
    max_tokens_mapped = min_tokens + enhanced_need * (max_tokens - min_tokens)
    
    return int(round(max_tokens_mapped))


def _should_use_sampling(features: CAFeatures) -> bool:
    """
    Determine whether to use sampling vs greedy decoding.
    
    Logic: Dynamic, complex CA patterns benefit from sampling diversity
    """
    # Calculate "creativity score" from features
    creativity_score = (
        0.3 * features.complexity +      # Complex patterns need creativity
        0.3 * features.intensity +       # Active patterns need exploration  
        0.2 * (1 - features.convergence) + # Non-convergent needs diversity
        0.2 * (1 - features.periodicity)   # Non-periodic needs variation
    )
    
    # Use sampling if creativity score > 0.4 (moderate threshold)
    return creativity_score > 0.4


# NOTE: Default cheap knobs configuration removed - use YAML config only
# All cheap knobs parameters MUST come from experiment config files
# This enforces explicit configuration and prevents hidden defaults


def cheap_knobs_to_generation_kwargs(knobs: CheapKnobs) -> Dict[str, Any]:
    """
    Convert CheapKnobs to generation keyword arguments for transformers.
    Pure function: CheapKnobs → Dict[str, Any]
    """
    return {
        'temperature': knobs.temperature,
        'top_p': knobs.top_p,
        'top_k': knobs.top_k,
        'repetition_penalty': knobs.repetition_penalty,
        'max_new_tokens': knobs.max_new_tokens,
        'do_sample': knobs.do_sample,
        'pad_token_id': 2,  # EOS token for CodeLlama
        'eos_token_id': 2
    }


def analyze_cheap_knobs_diversity(knobs_list: list) -> Dict[str, Any]:
    """
    Analyze diversity of cheap knobs across population.
    Useful for understanding parameter space exploration.
    """
    if not knobs_list:
        return {'error': 'No knobs provided'}
    
    temperatures = [k.temperature for k in knobs_list]
    top_ps = [k.top_p for k in knobs_list]
    top_ks = [k.top_k for k in knobs_list]
    penalties = [k.repetition_penalty for k in knobs_list]
    tokens = [k.max_new_tokens for k in knobs_list]
    
    return {
        'count': len(knobs_list),
        'temperature': {
            'min': min(temperatures), 'max': max(temperatures),
            'mean': sum(temperatures) / len(temperatures),
            'std': np.std(temperatures)
        },
        'top_p': {
            'min': min(top_ps), 'max': max(top_ps),
            'mean': sum(top_ps) / len(top_ps),
            'std': np.std(top_ps)
        },
        'top_k': {
            'min': min(top_ks), 'max': max(top_ks),
            'mean': sum(top_ks) / len(top_ks),
            'std': np.std(top_ks)
        },
        'sampling_ratio': sum(1 for k in knobs_list if k.do_sample) / len(knobs_list)
    } 