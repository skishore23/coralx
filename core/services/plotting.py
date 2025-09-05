"""
Plotting and visualization services for CORAL-X evolution results.

This module provides Pareto front visualization, hypervolume calculation,
and other plotting utilities for multi-objective optimization results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ..domain.genome import Genome, MultiObjectiveScores
from ..domain.neat import Population
from .pareto.selection import fast_non_dominated_sort, calculate_crowding_distance


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for plotting parameters."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    colors: List[str] = None
    markers: List[str] = None

    def __post_init__(self):
        if self.colors is None:
            object.__setattr__(self, 'colors', [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ])
        if self.markers is None:
            object.__setattr__(self, 'markers', ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'])


class ParetoPlotter:
    """Pareto front visualization and analysis."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        plt.style.use(self.config.style)

    def plot_pareto_fronts_2d(self,
                             population: Population,
                             obj1: str,
                             obj2: str,
                             output_path: Path,
                             title: str = "Pareto Fronts",
                             show_crowding: bool = True) -> None:
        """Plot 2D Pareto fronts for two objectives."""

        if population.size() == 0:
            raise ValueError("Cannot plot Pareto fronts for empty population")

        # Get evaluated genomes with multi-objective scores
        evaluated_genomes = [g for g in population.genomes if g.has_multi_scores()]
        if not evaluated_genomes:
            raise ValueError("No genomes with multi-objective scores for Pareto plotting")

        # Extract objective values
        obj1_values = [getattr(g.multi_scores, obj1) for g in evaluated_genomes]
        obj2_values = [getattr(g.multi_scores, obj2) for g in evaluated_genomes]

        # Calculate Pareto fronts
        fronts = fast_non_dominated_sort(evaluated_genomes)

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Plot each front
        for i, front in enumerate(fronts):
            if not front:
                continue

            front_obj1 = [getattr(g.multi_scores, obj1) for g in front]
            front_obj2 = [getattr(g.multi_scores, obj2) for g in front]

            # Sort by first objective for line plotting
            sorted_indices = np.argsort(front_obj1)
            front_obj1_sorted = [front_obj1[i] for i in sorted_indices]
            front_obj2_sorted = [front_obj2[i] for i in sorted_indices]

            color = self.config.colors[i % len(self.config.colors)]
            marker = self.config.markers[i % len(self.config.markers)]

            # Plot points
            ax.scatter(front_obj1, front_obj2,
                      c=color, marker=marker, s=100,
                      label=f'Front {i+1} ({len(front)} solutions)',
                      alpha=0.7, edgecolors='black', linewidth=0.5)

            # Plot front line
            if len(front_obj1_sorted) > 1:
                try:
                    ax.plot(front_obj1_sorted, front_obj2_sorted,
                           color=color, linestyle='--', alpha=0.5, linewidth=1)
                except ValueError:
                    # Skip line plotting if there are issues with the data
                    pass

        # Add crowding distance visualization if requested
        if show_crowding and fronts:
            self._add_crowding_visualization(ax, fronts[0], obj1, obj2)

        # Formatting
        ax.set_xlabel(f'{obj1.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel(f'{obj2.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

    def plot_hypervolume_progress(self,
                                 hypervolume_history: List[float],
                                 output_path: Path,
                                 title: str = "Hypervolume Progress") -> None:
        """Plot hypervolume progress over generations."""

        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        generations = list(range(len(hypervolume_history)))
        ax.plot(generations, hypervolume_history,
               marker='o', linewidth=2, markersize=6,
               color=self.config.colors[0])

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Hypervolume', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

    def _add_crowding_visualization(self, ax, front: List[Genome], obj1: str, obj2: str) -> None:
        """Add crowding distance visualization to the plot."""
        if len(front) < 3:
            return

        try:
            # Calculate crowding distances
            crowding_data = calculate_crowding_distance(front)

            # Normalize crowding distances for visualization
            distances = [d for _, d in crowding_data]
            if not distances or max(distances) == 0:
                return

            max_distance = max(distances)
            normalized_distances = [d / max_distance for d in distances]

            # Plot crowding distance as point size
            for (genome, distance), norm_dist in zip(crowding_data, normalized_distances):
                obj1_val = getattr(genome.multi_scores, obj1)
                obj2_val = getattr(genome.multi_scores, obj2)

                # Size based on crowding distance (more diverse = larger)
                size = 50 + norm_dist * 100
                ax.scatter(obj1_val, obj2_val, s=size,
                          alpha=0.3, color='red', edgecolors='darkred')
        except Exception:
            # Skip crowding visualization if there are issues
            pass


class ResultsExporter:
    """Export evolution results to various formats."""

    def export_to_csv(self,
                     population: Population,
                     generation: int,
                     output_path: Path,
                     include_ca_features: bool = True) -> None:
        """Export population results to CSV format."""

        if population.size() == 0:
            return

        rows = []
        for genome in population.genomes:
            row = {
                'generation': generation,
                'genome_id': genome.id,
                'fitness': genome.fitness if genome.fitness is not None else 0.0,
            }

            # Add multi-objective scores
            if genome.has_multi_scores():
                scores_dict = genome.multi_scores.to_dict()
                row.update({f'score_{k}': v for k, v in scores_dict.items()})

            # Add LoRA configuration
            if genome.lora_cfg:
                row.update({
                    'lora_rank': genome.lora_cfg.r,
                    'lora_alpha': genome.lora_cfg.alpha,
                    'lora_dropout': genome.lora_cfg.dropout,
                    'lora_target_modules': ','.join(genome.lora_cfg.target_modules),
                })

            # Add CA seed information
            if genome.seed:
                row.update({
                    'ca_rule': genome.seed.rule,
                    'ca_steps': genome.seed.steps,
                    'ca_grid_size': f"{genome.seed.grid.shape[0]}x{genome.seed.grid.shape[1]}",
                })

            # Add CA features if available
            if include_ca_features and hasattr(genome, 'ca_features') and genome.ca_features:
                features_dict = genome.ca_features.to_dict()
                row.update({f'ca_{k}': v for k, v in features_dict.items()})

            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def export_generation_summary(self,
                                generation_data: List[Dict[str, Any]],
                                output_path: Path) -> None:
        """Export generation summary statistics to CSV."""

        df = pd.DataFrame(generation_data)
        df.to_csv(output_path, index=False)


def calculate_hypervolume(population: Population,
                         reference_point: Optional[MultiObjectiveScores] = None) -> float:
    """Calculate hypervolume indicator for a population."""

    if population.size() == 0:
        return 0.0

    evaluated_genomes = [g for g in population.genomes if g.has_multi_scores()]
    if not evaluated_genomes:
        return 0.0

    # Extract objective values
    objectives = ['bugfix', 'style', 'security', 'runtime', 'syntax']
    points = np.array([[getattr(g.multi_scores, obj) for obj in objectives]
                      for g in evaluated_genomes])

    # Use reference point if provided, otherwise use worst point
    if reference_point is None:
        ref_point = np.min(points, axis=0) - 0.1  # Slightly worse than worst
    else:
        ref_point = np.array([getattr(reference_point, obj) for obj in objectives])

    # Calculate hypervolume (simplified 2D case for now)
    if points.shape[1] == 2:
        return _calculate_hypervolume_2d(points, ref_point)
    else:
        # For higher dimensions, use approximation
        return _calculate_hypervolume_approx(points, ref_point)


def _calculate_hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Calculate hypervolume for 2D case."""
    if len(points) == 0:
        return 0.0

    # Sort points by first objective
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]

    # Calculate hypervolume
    hv = 0.0
    prev_x = ref_point[0]

    for point in sorted_points:
        if point[1] > ref_point[1]:  # Only count if better than reference
            hv += (point[0] - prev_x) * (point[1] - ref_point[1])
            prev_x = point[0]

    return hv


def _calculate_hypervolume_approx(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Approximate hypervolume calculation for higher dimensions."""
    # Simple approximation: count points that dominate reference
    dominated_count = np.sum(np.all(points > ref_point, axis=1))
    return float(dominated_count) / len(points) if len(points) > 0 else 0.0
