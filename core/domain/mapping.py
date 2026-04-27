"""
Feature mapping for cellular automata to LoRA parameters.

This module implements the core mapping functions that transform cellular
automata features into LoRA adapter configurations. Uses dynamic diversity
injection and feature fingerprinting for optimal parameter selection.
"""

from dataclasses import dataclass
from typing import Any

from .feature_extraction import CAFeatures
from .stable_hash import stable_int


@dataclass(frozen=True)
class AdapterConfig:
    """Unified adapter configuration supporting both LoRA and DoRA."""

    r: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    adapter_type: str = "lora"  # "lora" or "dora"


# Backward compatibility alias
LoRAConfig = AdapterConfig


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration ranges for evolution parameters - NO DEFAULTS."""

    rank_candidates: tuple[int, ...]
    alpha_candidates: tuple[float, ...]  # Changed from range to discrete candidates
    dropout_candidates: tuple[float, ...]  # Changed from range to discrete candidates
    target_modules: tuple[str, ...]  # Target modules for adapter training


def map_features_to_lora_config(
    features: CAFeatures,
    config: dict[str, Any],
    diversity_strength: float = 1.0,
    genome_index: int = 0,
) -> AdapterConfig:
    """
    Map CA features to LoRA config with dynamic diversity injection.
    Uses feature fingerprinting with adjustable diversity strength.

    Args:
        features: CA features extracted from cellular automata evolution
        config: Configuration dictionary with evolution parameters
        diversity_strength: Multiplier for diversity injection (1.0 = baseline, 2.0 = max diversity)
        genome_index: Index of genome in population used as extra entropy
    """
    # Extract evolution configuration
    if "evo" not in config:
        raise ValueError("  'evo' section missing from configuration")

    evo_raw = config["evo"]

    # Validate required fields
    required_fields = [
        "rank_candidates",
        "alpha_candidates",
        "dropout_candidates",
        "target_modules",
    ]
    for field in required_fields:
        if field not in evo_raw:
            raise ValueError(f"  '{field}' missing from evolution configuration")

    # Create evolution config
    evo_cfg = EvolutionConfig(
        rank_candidates=tuple(evo_raw["rank_candidates"]),
        alpha_candidates=tuple(evo_raw["alpha_candidates"]),
        dropout_candidates=tuple(evo_raw["dropout_candidates"]),
        target_modules=tuple(evo_raw["target_modules"]),
    )

    # Use genome index as additional entropy for diversity
    rank = _map_with_enhanced_diversity(
        features, evo_cfg.rank_candidates, "rank", diversity_strength, genome_index
    )
    alpha = _map_with_enhanced_diversity(
        features, evo_cfg.alpha_candidates, "alpha", diversity_strength, genome_index
    )
    dropout = _map_with_enhanced_diversity(
        features,
        evo_cfg.dropout_candidates,
        "dropout",
        diversity_strength,
        genome_index,
    )

    # Get adapter type from config (default to LoRA for backward compatibility)
    adapter_type = config.get("adapter_type", "lora")

    return AdapterConfig(
        r=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=tuple(evo_raw["target_modules"]),
        adapter_type=adapter_type,
    )


def _map_with_enhanced_diversity(
    features: CAFeatures,
    candidates: tuple,
    param_type: str,
    diversity_strength: float,
    genome_index: int = 0,
):
    """Map features with additional diversity from feature fingerprints and genome index."""
    if len(candidates) == 0:
        raise ValueError(f"  No candidates provided for {param_type} mapping")

    # Create genome-specific entropy from CA features themselves
    # Use feature combinations to ensure each genome gets unique mappings
    genome_entropy = stable_int(
        {
            "complexity": f"{features.complexity * 1000:.0f}",
            "intensity": f"{features.intensity * 1000:.0f}",
            "periodicity": f"{features.periodicity * 1000:.0f}",
            "convergence": f"{features.convergence * 1000:.0f}",
            "param_type": param_type,
            "genome_index": genome_index * 7919,
        },
        10000,
    )

    # Create an enhanced feature fingerprint using genome-specific entropy.
    enhanced_fingerprint = stable_int(
        {
            "complexity": f"{features.complexity:.8f}",
            "intensity": f"{features.intensity:.8f}",
            "periodicity": f"{features.periodicity:.8f}",
            "convergence": f"{features.convergence:.8f}",
            "param_type": param_type,
            "genome_entropy": genome_entropy,
            "genome_index": genome_index + 1,
        }
    )

    # Apply diversity strength adjustment with genome-index entropy.
    if diversity_strength <= 0.5:
        # LOW DIVERSITY: Use feature blending for cache efficiency
        # Similar features map to similar configs, but genome index ensures uniqueness
        feature_blend = (
            features.complexity * 0.4
            + features.intensity * 0.3
            + features.periodicity * 0.2
            + features.convergence * 0.1
        )
        # Add genome entropy and index to spread low-diversity mappings.
        feature_blend = (
            feature_blend + genome_entropy * 0.001 + genome_index * 0.01
        ) % 1.0

        # Quantize to create cache groups with minimum diversity
        num_groups = max(2, int(len(candidates) * diversity_strength))
        group_index = int(feature_blend * num_groups) % num_groups
        candidate_index = int(group_index / num_groups * (len(candidates) - 1))

    elif diversity_strength >= 1.5:
        # HIGH DIVERSITY: Use enhanced fingerprinting with maximum spread
        super_enhanced_fingerprint = stable_int(
            {
                "fingerprint": enhanced_fingerprint,
                "complexity_intensity": f"{features.complexity * features.intensity:.12f}",
                "periodicity_convergence": f"{features.periodicity / (features.convergence + 1e-8):.12f}",
                "diversity_strength": int(diversity_strength * 1000),
                "genome_index": genome_index * 997,
            }
        )
        candidate_index = abs(super_enhanced_fingerprint) % len(candidates)

    else:
        # BALANCED DIVERSITY: Enhanced fingerprinting with genome index
        candidate_index = abs(enhanced_fingerprint) % len(candidates)

    return candidates[candidate_index]


def calculate_dynamic_diversity_strength(
    cache_hit_rate: float, recent_improvements: list, config: dict[str, Any]
) -> float:
    """
    Calculate dynamic diversity strength based on evolution state.

    Args:
        cache_hit_rate: Fraction of genomes using cached adapters (0.0-1.0)
        recent_improvements: List of fitness improvements over recent generations
        config: Configuration with diversity parameters

    Returns:
        diversity_strength: Multiplier for diversity injection
    """
    diversity_config = config.get("evo", {}).get("diversity", {})

    # Get configuration parameters with defaults
    mode = diversity_config.get("mode", "adaptive")
    base_strength = diversity_config.get("base_strength", 1.0)
    max_strength = diversity_config.get("max_strength", 2.0)
    min_strength = diversity_config.get("min_strength", 0.3)
    cache_threshold = diversity_config.get("cache_threshold", 0.8)
    plateau_threshold = diversity_config.get("plateau_threshold", 0.05)
    plateau_window = diversity_config.get("plateau_window", 3)

    if mode == "fixed":
        return base_strength
    elif mode == "aggressive":
        return max_strength

    # ADAPTIVE MODE: Adjust based on cache rate and performance
    strength = base_strength

    # Increase diversity if cache hit rate is too high
    if cache_hit_rate > cache_threshold:
        cache_penalty = (cache_hit_rate - cache_threshold) / (1.0 - cache_threshold)
        strength += cache_penalty * (max_strength - base_strength)
        print(
            f"   🔄 High cache rate ({cache_hit_rate:.2f}) → diversity boost: {strength:.2f}"
        )

    # Increase diversity if performance has plateaued
    if len(recent_improvements) >= plateau_window:
        recent_window = recent_improvements[-plateau_window:]
        avg_improvement = sum(recent_window) / len(recent_window)

        if avg_improvement < plateau_threshold:
            plateau_penalty = (plateau_threshold - avg_improvement) / plateau_threshold
            strength += (
                plateau_penalty * (max_strength - base_strength) * 0.5
            )  # Moderate boost
            print(
                f"   📈 Performance plateau ({avg_improvement:.3f}) → diversity boost: {strength:.2f}"
            )

    # Decrease diversity if we have good exploration (low cache rate + good improvement)
    if (
        cache_hit_rate < 0.3
        and len(recent_improvements) > 0
        and recent_improvements[-1] > plateau_threshold
    ):
        exploration_bonus = (0.3 - cache_hit_rate) / 0.3
        strength = max(
            min_strength,
            strength - exploration_bonus * (base_strength - min_strength) * 0.3,
        )
        print(f"   🎯 Good exploration → diversity reduction: {strength:.2f}")

    # Clamp to valid range
    return max(min_strength, min(max_strength, strength))
