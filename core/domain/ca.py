"""
Cellular Automata implementation for CORAL-X.

This module provides pure functional implementations of cellular automata
evolution rules used to generate diverse initial configurations for the
evolutionary algorithm. All functions are side-effect free.
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CASeed:
    """Initial bit‑pattern or random seed."""
    grid: NDArray[np.int_]   # shape (H, W), values 0..S‑1
    rule: int                # e.g. rule‑30 for 2‑state CA
    steps: int               # evolution length


@dataclass(frozen=True)
class CAStateHistory:
    """All intermediate grids (0 … t)."""
    history: List[NDArray[np.int_]]


def evolve(seed: CASeed, genome_id: str = None) -> CAStateHistory:
    """Pure arrow: Seed ──▶ History."""
    state = seed.grid.copy()
    hist = [state.copy()]

    for _ in range(seed.steps):
        state = next_step(state, seed.rule)
        hist.append(state.copy())

    return CAStateHistory(hist)


def next_step(grid: NDArray[np.int_], rule: int) -> NDArray[np.int_]:
    """Apply rule to whole grid without side‑effects."""
    height, width = grid.shape
    new_grid = np.zeros_like(grid)

    # Apply elementary CA rule with Moore neighborhood
    for i in range(height):
        for j in range(width):
            # Calculate neighbor configuration with periodic boundary conditions
            neighbor_config = 0
            bit_pos = 0

            # Moore neighborhood: 8 neighbors + center
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = (i + di) % height, (j + dj) % width
                    if grid[ni, nj] == 1:
                        neighbor_config |= (1 << bit_pos)
                    bit_pos += 1

            # Apply rule based on neighbor configuration
            new_grid[i, j] = _apply_rule_fixed(neighbor_config, rule)

    return new_grid


def _apply_rule_fixed(neighbor_config: int, rule: int) -> int:
    """
    Apply CA rule using pure mathematical approach.
    
    Uses rule number directly in bit-based lookup for elementary CA behavior.
    No hardcoded mappings, no fallbacks - pure functional approach.
    """
    # Count live neighbors (exclude center cell)
    live_neighbors = bin(neighbor_config).count('1')
    center_alive = (neighbor_config >> 4) & 1

    # Use rule number as bit pattern for elementary CA
    # Rule number encodes all possible neighborhood states
    # This ensures different rules produce different behaviors

    # Create configuration index from neighborhood
    # Combine center state and neighbor count for rule lookup
    config_index = (center_alive << 3) | min(live_neighbors, 7)

    # Apply rule by checking corresponding bit
    # Each rule number is treated as 8-bit pattern
    return (rule >> config_index) & 1
