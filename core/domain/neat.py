"""
NEAT-style evolutionary operations for CORAL-X.

This module implements core evolutionary operations including selection,
mutation, and crossover for genome populations. The implementation follows
NEAT principles while being optimized for LoRA adapter evolution.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from random import Random
from .genome import Genome
from .ca import CASeed
from .mapping import LoRAConfig, EvolutionConfig


@dataclass(frozen=True)
class Population:
    genomes: Tuple[Genome, ...]

    def size(self) -> int:
        """Get population size."""
        return len(self.genomes)

    def best(self) -> Genome:
        """Get the best genome by fitness."""
        if not self.genomes:
            raise ValueError("Cannot get best from empty population")

        evaluated = [g for g in self.genomes if g.is_evaluated()]
        if not evaluated:
            raise RuntimeError(
                f"No evaluated genomes in population of {len(self.genomes)} genomes. "
                f"This indicates evaluation failures - check adapter training/evaluation logs. "
                f"Cannot continue evolution without evaluated genomes."
            )

        return max(evaluated, key=lambda g: g.fitness)

    def sorted_by_fitness(self) -> 'Population':
        """Return population sorted by fitness (best first)."""
        evaluated = [g for g in self.genomes if g.is_evaluated()]
        if not evaluated:
            raise RuntimeError(
                f"No evaluated genomes to sort in population of {len(self.genomes)} genomes. "
                f"Cannot sort population without fitness scores."
            )

        sorted_genomes = sorted(evaluated, key=lambda g: g.fitness, reverse=True)
        return Population(tuple(sorted_genomes))

    def with_default_fitness(self, default_fitness: float = 0.0) -> 'Population':
        """No default fitness assignment allowed."""
        unevaluated_genomes = [g for g in self.genomes if not g.is_evaluated()]
        if unevaluated_genomes:
            unevaluated_ids = [g.id for g in unevaluated_genomes]
            raise RuntimeError(
                f"Cannot assign default fitness to unevaluated genomes. "
                f"All genomes must have real fitness scores. "
                f"Unevaluated genomes: {unevaluated_ids}"
            )
        return Population(self.genomes)


def select(pop: Population, k: int) -> Population:
    """Deterministic selection based on fitness."""
    if k <= 0:
        return Population(())

    # Check if we have any evaluated genomes
    evaluated = [g for g in pop.genomes if g.is_evaluated()]
    if not evaluated:
        raise RuntimeError(
            f"No evaluated genomes for selection. "
            f"Cannot perform selection without fitness scores. "
            f"All {len(pop.genomes)} genomes lack evaluation."
        )

    sorted_pop = pop.sorted_by_fitness()
    survivors = sorted_pop.genomes[:min(k, len(sorted_pop.genomes))]
    return Population(survivors)


def tournament_select(pop: Population, k: int, tournament_size: int = 3, rng: Random = None) -> Population:
    """Tournament selection for M1 deterministic evolution."""
    if k <= 0:
        return Population(())

    # Check if we have any evaluated genomes
    evaluated = [g for g in pop.genomes if g.is_evaluated()]
    if not evaluated:
        raise RuntimeError(
            f"No evaluated genomes for tournament selection. "
            f"Cannot perform selection without fitness scores. "
            f"All {len(pop.genomes)} genomes lack evaluation."
        )

    if rng is None:
        rng = Random(42)  # Deterministic default for M1

    survivors = []
    for _ in range(k):
        # Select tournament participants
        tournament = rng.sample(evaluated, min(tournament_size, len(evaluated)))

        # Select winner (highest fitness)
        winner = max(tournament, key=lambda g: g.fitness)
        survivors.append(winner)

    return Population(tuple(survivors))


def mutate(genome: Genome, evo_cfg: EvolutionConfig, rng: Random, generation: int = 0,
           diversity_strength: float = 1.0, config_dict: dict = None, run_id: str = None) -> Genome:
    """Return a new genome with small CA or LoRA perturbation using dynamic diversity."""
    # Create unique mutant ID with generation tracking
    mutant_number = rng.randint(1000, 9999)
    mutant_id = f"gen{generation + 1}_mut_{genome.id.split('_')[-1]}_{mutant_number}"



    # Decide whether to mutate CA or LoRA (70% CA, 30% LoRA for more CA exploration)
    if rng.random() < 0.7:
        # Mutate CA seed and regenerate LoRA config with dynamic diversity
        new_seed = _mutate_ca_seed(genome.seed, rng)

        # config_dict is required for proper CA to LoRA mapping
        if not config_dict:
            raise ValueError(
                "config_dict required for CA mutation. "
                "Cannot perform proper CA → LoRA mapping without configuration. "
                "No fallback to original LoRA config allowed."
            )

        from .ca import evolve
        from .feature_extraction import extract_features
        from .mapping import map_features_to_lora_config

        # Re-evolve CA and extract features
        history = evolve(new_seed, genome_id=mutant_id)
        features = extract_features(history)

        # Apply dynamic diversity to LoRA mapping with genome-derived index
        genome_index = abs(hash(genome.id)) % 1000  # Generate unique index from genome ID
        new_lora = map_features_to_lora_config(features, config_dict, diversity_strength, genome_index)

        # Store CA features for consistency
        return Genome(seed=new_seed, lora_cfg=new_lora, id=mutant_id, ca_features=features, run_id=run_id)
    else:
        # Mutate LoRA config directly (preserve CA features since CA didn't change)
        new_lora = _mutate_lora_config(genome.lora_cfg, evo_cfg, rng)
        return Genome(seed=genome.seed, lora_cfg=new_lora, id=mutant_id, ca_features=genome.ca_features, run_id=run_id)


def crossover(p1: Genome, p2: Genome, evo_cfg: EvolutionConfig, rng: Random, generation: int = 0,
              diversity_strength: float = 1.0, config_dict: dict = None, run_id: str = None) -> Genome:
    """Breed new genome from two parents using dynamic diversity."""
    # Create unique child ID with generation tracking
    child_number = rng.randint(1000, 9999)
    p1_num = p1.id.split('_')[-1] if '_' in p1.id else p1.id[-4:]
    p2_num = p2.id.split('_')[-1] if '_' in p2.id else p2.id[-4:]
    child_id = f"gen{generation + 1}_cross_{p1_num}x{p2_num}_{child_number}"



    # Create hybrid CA seed
    hybrid_seed = _crossover_ca_seeds(p1.seed, p2.seed, rng)

    # config_dict is required for proper CA to LoRA mapping
    if not config_dict:
        raise ValueError(
            "config_dict required for crossover. "
            "Cannot perform proper CA → LoRA mapping without configuration. "
            "No fallback to direct LoRA crossover allowed."
        )

    from .ca import evolve
    from .feature_extraction import extract_features
    from .mapping import map_features_to_lora_config

    # Evolve hybrid CA and extract features
    history = evolve(hybrid_seed, genome_id=child_id)
    features = extract_features(history)

    # Apply dynamic diversity to LoRA mapping with parent-derived index
    parent_hash = abs(hash(p1.id + p2.id)) % 1000  # Generate unique index from parent IDs
    hybrid_lora = map_features_to_lora_config(features, config_dict, diversity_strength, parent_hash)

    # 🔥 FIX: Store CA features for consistency
    return Genome(seed=hybrid_seed, lora_cfg=hybrid_lora, id=child_id, ca_features=features, run_id=run_id)


def _mutate_ca_seed(seed: CASeed, rng: Random) -> CASeed:
    """Mutate CA seed parameters."""
    new_grid = seed.grid.copy()

    # Flip a small percentage of cells
    mutation_rate = 0.05
    mask = rng.random() < mutation_rate
    if isinstance(mask, bool):
        # For scalar case
        flip_positions = [(rng.randint(0, seed.grid.shape[0]-1),
                          rng.randint(0, seed.grid.shape[1]-1))]
    else:
        # Would need proper vectorized approach for array mask
        flip_positions = [(rng.randint(0, seed.grid.shape[0]-1),
                          rng.randint(0, seed.grid.shape[1]-1))]

    for i, j in flip_positions:
        max_state = int(np.max(new_grid)) + 1
        new_grid[i, j] = rng.randint(0, max_state - 1)

    # Possibly mutate rule
    new_rule = seed.rule
    if rng.random() < 0.1:  # 10% chance to mutate rule
        new_rule = rng.randint(max(0, seed.rule - 5), seed.rule + 5)

    # Possibly mutate steps
    new_steps = seed.steps
    if rng.random() < 0.1:  # 10% chance to mutate steps
        new_steps = max(1, seed.steps + rng.randint(-2, 2))

    return CASeed(grid=new_grid, rule=new_rule, steps=new_steps)


def _mutate_lora_config(lora_cfg: LoRAConfig, evo_cfg: EvolutionConfig, rng: Random) -> LoRAConfig:
    """Mutate LoRA configuration parameters."""
    # Mutate rank
    new_r = lora_cfg.r
    if rng.random() < 0.3:  # 30% chance to mutate rank
        new_r = rng.choice(evo_cfg.rank_candidates)

    # Mutate alpha using discrete candidates
    new_alpha = lora_cfg.alpha
    if rng.random() < 0.3:  # 30% chance to mutate alpha
        new_alpha = rng.choice(evo_cfg.alpha_candidates)

    # Mutate dropout using discrete candidates
    new_dropout = lora_cfg.dropout
    if rng.random() < 0.3:  # 30% chance to mutate dropout
        new_dropout = rng.choice(evo_cfg.dropout_candidates)

    return LoRAConfig(
        r=new_r,
        alpha=new_alpha,
        dropout=new_dropout,
        target_modules=lora_cfg.target_modules
    )


def _crossover_ca_seeds(seed1: CASeed, seed2: CASeed, rng: Random) -> CASeed:
    """Crossover two CA seeds."""
    # Take grid from one parent and rule/steps from other
    if rng.random() < 0.5:
        grid = seed1.grid.copy()
        rule = seed2.rule
        steps = seed2.steps
    else:
        grid = seed2.grid.copy()
        rule = seed1.rule
        steps = seed1.steps

    return CASeed(grid=grid, rule=rule, steps=steps)


def _crossover_lora_configs(cfg1: LoRAConfig, cfg2: LoRAConfig,
                           evo_cfg: EvolutionConfig, rng: Random) -> LoRAConfig:
    """Crossover two LoRA configurations."""
    # Take each parameter from one parent or the other
    r = cfg1.r if rng.random() < 0.5 else cfg2.r
    alpha = cfg1.alpha if rng.random() < 0.5 else cfg2.alpha
    dropout = cfg1.dropout if rng.random() < 0.5 else cfg2.dropout

    return LoRAConfig(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=cfg1.target_modules  # Keep modules consistent
    )
