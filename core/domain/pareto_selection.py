"""
NSGA-II multi-objective selection for CORAL-X.

This module implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II)
for Pareto-optimal selection in multi-objective optimization. Used to maintain
diversity in genome populations while optimizing multiple fitness objectives.
"""
from dataclasses import dataclass
from typing import List, Tuple, Set
import numpy as np
from .genome import Genome, MultiObjectiveScores
from .neat import Population


@dataclass(frozen=True)
class ParetoRank:
    """Pareto ranking information for a genome."""
    rank: int
    crowding_distance: float


def dominates(scores1: MultiObjectiveScores, scores2: MultiObjectiveScores) -> bool:
    """Check if scores1 dominates scores2 in Pareto sense."""
    s1_dict = scores1.to_dict()
    s2_dict = scores2.to_dict()
    
    # At least one objective is strictly better
    at_least_one_better = False
    # All objectives are at least as good
    all_at_least_as_good = True
    
    for objective in s1_dict:
        if s1_dict[objective] > s2_dict[objective]:
            at_least_one_better = True
        elif s1_dict[objective] < s2_dict[objective]:
            all_at_least_as_good = False
            break
    
    return at_least_one_better and all_at_least_as_good


def fast_non_dominated_sort(genomes: List[Genome]) -> List[List[Genome]]:
    """Fast non-dominated sorting algorithm from NSGA-II."""
    if not genomes:
        return []
    
    # Ensure all genomes have multi-objective scores
    evaluated_genomes = [g for g in genomes if g.has_multi_scores()]
    if not evaluated_genomes:
        raise ValueError("  No genomes with multi-objective scores for Pareto selection")
    
    fronts = []
    domination_count = {}  # How many solutions dominate this one
    dominated_solutions = {}  # Which solutions this one dominates
    
    # Initialize
    for genome in evaluated_genomes:
        domination_count[genome.id] = 0
        dominated_solutions[genome.id] = []
    
    # Calculate domination relationships
    for i, genome1 in enumerate(evaluated_genomes):
        for j, genome2 in enumerate(evaluated_genomes):
            if i != j:
                if dominates(genome1.multi_scores, genome2.multi_scores):
                    dominated_solutions[genome1.id].append(genome2)
                elif dominates(genome2.multi_scores, genome1.multi_scores):
                    domination_count[genome1.id] += 1
    
    # First front (non-dominated solutions)
    first_front = []
    for genome in evaluated_genomes:
        if domination_count[genome.id] == 0:
            first_front.append(genome)
    
    fronts.append(first_front)
    
    # Build subsequent fronts
    current_front_idx = 0
    while current_front_idx < len(fronts) and fronts[current_front_idx]:
        next_front = []
        for genome in fronts[current_front_idx]:
            for dominated_genome in dominated_solutions[genome.id]:
                domination_count[dominated_genome.id] -= 1
                if domination_count[dominated_genome.id] == 0:
                    next_front.append(dominated_genome)
        
        if next_front:
            fronts.append(next_front)
        else:
            break
        current_front_idx += 1
    
    return fronts


def calculate_crowding_distance(front: List[Genome]) -> List[Tuple[Genome, float]]:
    """Calculate crowding distance for genomes in a front."""
    if len(front) <= 2:
        # Boundary solutions get infinite distance
        return [(genome, float('inf')) for genome in front]
    
    # Initialize distances
    distances = {genome.id: 0.0 for genome in front}
    
    # Get all objective names
    objectives = list(front[0].multi_scores.to_dict().keys())
    
    # Calculate crowding distance for each objective
    for objective in objectives:
        # Sort by this objective
        sorted_front = sorted(front, key=lambda g: g.multi_scores.to_dict()[objective])
        
        # Boundary solutions get infinite distance
        distances[sorted_front[0].id] = float('inf')
        distances[sorted_front[-1].id] = float('inf')
        
        # Get objective range
        obj_values = [g.multi_scores.to_dict()[objective] for g in sorted_front]
        obj_range = max(obj_values) - min(obj_values)
        
        if obj_range > 0:  # Avoid division by zero
            # Calculate crowding distance for intermediate solutions
            for i in range(1, len(sorted_front) - 1):
                distance_contrib = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                distances[sorted_front[i].id] += distance_contrib
    
    return [(genome, distances[genome.id]) for genome in front]


def nsga2_select(population: Population, target_size: int) -> Population:
    """Select population using NSGA-II algorithm."""
    if target_size <= 0:
        return Population(())
    
    if population.size() <= target_size:
        return population
    
    # Get non-dominated fronts
    fronts = fast_non_dominated_sort(list(population.genomes))
    
    # Select genomes from fronts
    selected = []
    front_idx = 0
    
    # Add complete fronts while possible
    while front_idx < len(fronts) and len(selected) + len(fronts[front_idx]) <= target_size:
        selected.extend(fronts[front_idx])
        front_idx += 1
    
    # If we need to partially fill from the next front
    if front_idx < len(fronts) and len(selected) < target_size:
        remaining_slots = target_size - len(selected)
        current_front = fronts[front_idx]
        
        # Calculate crowding distances and sort by them
        front_with_distances = calculate_crowding_distance(current_front)
        front_with_distances.sort(key=lambda x: x[1], reverse=True)  # Higher crowding distance first
        
        # Take the most diverse solutions
        for i in range(remaining_slots):
            selected.append(front_with_distances[i][0])
    
    return Population(tuple(selected)) 