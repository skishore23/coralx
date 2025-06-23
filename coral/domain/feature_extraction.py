###############################################################################
# Feature extractor — still pure
###############################################################################
from dataclasses import dataclass
import numpy as np
from scipy import stats
from .ca import CAStateHistory


@dataclass(frozen=True)
class CAFeatures:
    complexity: float
    intensity: float
    periodicity: float
    convergence: float


def extract_features(hist: CAStateHistory) -> CAFeatures:
    """History ──▶ Features.  Pure & vectorisable."""
    grids = hist.history
    
    complexity = _calculate_complexity(grids)
    intensity = _calculate_intensity(grids)
    periodicity = _calculate_periodicity(grids)
    convergence = _calculate_convergence(grids)
    
    return CAFeatures(
        complexity=complexity,
        intensity=intensity,
        periodicity=periodicity,
        convergence=convergence
    )


def _calculate_complexity(grids) -> float:
    """Calculate entropy-based complexity measure with improved handling of uniform states."""
    if not grids:
        return 0.0
    
    # Use the final grid for complexity calculation
    final_grid = grids[-1]
    flat = final_grid.flatten()
    
    # Calculate Shannon entropy
    unique_vals, counts = np.unique(flat, return_counts=True)
    probabilities = counts / len(flat)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # Improved normalization: handle uniform states gracefully
    num_unique = len(unique_vals)
    if num_unique <= 1:
        # Uniform state: use spatial patterns for complexity
        return _calculate_spatial_complexity(final_grid)
    else:
        # Normal entropy calculation
        max_entropy = np.log2(num_unique)
        base_complexity = entropy / max_entropy
        
        # Add spatial complexity component to increase diversity
        spatial_complexity = _calculate_spatial_complexity(final_grid)
        
        # Combine entropy and spatial complexity (60% entropy, 40% spatial)
        return 0.6 * base_complexity + 0.4 * spatial_complexity


def _calculate_spatial_complexity(grid) -> float:
    """Calculate spatial complexity based on local patterns and gradients."""
    if grid.size == 0:
        return 0.0
    
    # Method 1: Edge density (transitions between different values)
    edge_count = 0
    total_edges = 0
    
    height, width = grid.shape
    for i in range(height):
        for j in range(width):
            # Check 4-connected neighbors
            neighbors = [
                (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            ]
            
            for ni, nj in neighbors:
                if 0 <= ni < height and 0 <= nj < width:
                    if grid[i, j] != grid[ni, nj]:
                        edge_count += 1
                    total_edges += 1
    
    edge_density = edge_count / total_edges if total_edges > 0 else 0.0
    
    # Method 2: Pattern diversity (using local 2x2 windows)
    if height >= 2 and width >= 2:
        patterns = set()
        for i in range(height - 1):
            for j in range(width - 1):
                # Extract 2x2 window as tuple
                window = (
                    grid[i, j], grid[i, j+1],
                    grid[i+1, j], grid[i+1, j+1]
                )
                patterns.add(window)
        
        # Normalize by maximum possible 2x2 patterns for binary CA
        max_patterns = 16  # 2^4 possible binary patterns
        pattern_diversity = len(patterns) / max_patterns
    else:
        pattern_diversity = 0.0
    
    # Method 3: Variance-based complexity
    variance_complexity = np.var(grid.astype(float))
    
    # Combine multiple spatial measures
    spatial_score = (
        0.4 * edge_density +
        0.4 * pattern_diversity + 
        0.2 * min(1.0, variance_complexity * 4)  # Scale variance to [0,1]
    )
    
    return min(1.0, spatial_score)


def _calculate_intensity(grids) -> float:
    """Calculate activity/change intensity across time."""
    if len(grids) < 2:
        return 0.0
    
    total_changes = 0
    total_cells = 0
    
    for i in range(1, len(grids)):
        changes = np.sum(grids[i] != grids[i-1])
        total_changes += changes
        total_cells += grids[i].size
    
    return total_changes / total_cells if total_cells > 0 else 0.0


def _calculate_periodicity(grids) -> float:
    """Detect periodic patterns in the evolution."""
    if len(grids) < 3:
        return 0.0
    
    # Calculate hash for each grid to detect cycles
    grid_hashes = [hash(grid.tobytes()) for grid in grids]
    
    # Look for repeating patterns
    max_period_score = 0.0
    for period in range(1, min(len(grid_hashes) // 2, 10)):
        matches = 0
        comparisons = 0
        
        for i in range(period, len(grid_hashes)):
            if grid_hashes[i] == grid_hashes[i - period]:
                matches += 1
            comparisons += 1
        
        if comparisons > 0:
            period_score = matches / comparisons
            max_period_score = max(max_period_score, period_score)
    
    return max_period_score


def _calculate_convergence(grids) -> float:
    """Measure how quickly the system converges to stability."""
    if len(grids) < 2:
        return 1.0
    
    # Calculate the rate of change decay
    changes = []
    for i in range(1, len(grids)):
        change_rate = np.sum(grids[i] != grids[i-1]) / grids[i].size
        changes.append(change_rate)
    
    if not changes:
        return 1.0
    
    # Measure convergence as the negative slope of change rate
    time_steps = np.arange(len(changes))
    if len(changes) > 1:
        slope, _, _, _, _ = stats.linregress(time_steps, changes)
        # Convert slope to convergence score (higher = more convergent)
        convergence_score = max(0.0, -slope)
        return min(1.0, convergence_score)
    
    return changes[0] 