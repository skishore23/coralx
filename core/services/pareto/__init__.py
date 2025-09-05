"""
Pareto selection services for multi-objective optimization.

This module provides NSGA-II based Pareto selection for maintaining
diversity in multi-objective evolutionary optimization.
"""

from .selection import (
    ParetoRank,
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    nsga2_select
)

__all__ = [
    "ParetoRank",
    "dominates",
    "fast_non_dominated_sort",
    "calculate_crowding_distance",
    "nsga2_select"
]
