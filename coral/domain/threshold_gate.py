###############################################################################
# Threshold Gate — Multi-objective filtering with dynamic σ-wave strictness
###############################################################################
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
from .genome import Genome


@dataclass(frozen=True)
class ObjectiveThresholds:
    """Multi-objective threshold configuration."""
    bugfix: float
    style: float
    security: float
    runtime: float
    syntax: float  # NEW: Syntax correctness objective
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easier iteration."""
        return {
            'bugfix': self.bugfix,
            'style': self.style,
            'security': self.security,
            'runtime': self.runtime,
            'syntax': self.syntax
        }


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for dynamic threshold evolution - NO DEFAULTS."""
    base_thresholds: ObjectiveThresholds
    max_thresholds: ObjectiveThresholds
    schedule: str  # linear | sqrt | sigmoid


@dataclass(frozen=True)
class MultiObjectiveScores:
    """Multi-objective evaluation scores."""
    bugfix: float
    style: float
    security: float
    runtime: float
    syntax: float  # NEW: Syntax correctness score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easier iteration."""
        return {
            'bugfix': self.bugfix,
            'style': self.style,
            'security': self.security,
            'runtime': self.runtime,
            'syntax': self.syntax
        }


def calculate_sigma(gen: int, max_gen: int, mode: str = "sigmoid") -> float:
    """
    Calculate σ-wave progression factor [0,1] - CORAL-X Architecture.
    
    From architecture: sigma(gen) = 1/(1+exp(-12*(x-0.5))) where x = gen/max_gen
    This creates the dynamic threshold progression: loose early → strict at gen 40
    """
    if max_gen <= 0:
        return 1.0
    
    x = gen / max_gen
    
    if mode == "sigmoid":
        # CORAL-X specified formula: σ-wave with 12 coefficient
        return 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
    elif mode == "sqrt":
        return math.sqrt(x)
    elif mode == "linear":
        return x
    else:
        raise ValueError(f"FAIL-FAST: Unknown threshold schedule mode: {mode}")
    
    
def get_sla_targets() -> Dict[str, float]:
    """Get SLA targets from CORAL-X architecture specification."""
    return {
        'bugfix': 0.90,    # Architecture: ≥ 0.90 BugFix rate at MAX_GEN
        'style': 0.97,     # Architecture: ≥ 0.97 Style score 
        'security': 1.0,   # Architecture: 1.0 Security flag (no security issues)
        'runtime': 0.90    # Architecture: ≥ 0.90 Runtime speed‑up
    }


def calculate_dynamic_thresholds(gen: int, max_gen: int, config: ThresholdConfig) -> ObjectiveThresholds:
    """Calculate current thresholds based on generation and σ-wave."""
    sigma = calculate_sigma(gen, max_gen, config.schedule)
    
    base = config.base_thresholds.to_dict()
    max_vals = config.max_thresholds.to_dict()
    
    current = {}
    for key in base:
        current[key] = base[key] + sigma * (max_vals[key] - base[key])
    
    return ObjectiveThresholds(**current)


def apply_threshold_gate(scores: MultiObjectiveScores, 
                        thresholds: ObjectiveThresholds) -> bool:
    """Apply threshold gate - returns True if genome passes all thresholds."""
    score_dict = scores.to_dict()
    threshold_dict = thresholds.to_dict()
    
    for objective, score in score_dict.items():
        if score < threshold_dict[objective]:
            return False
    
    return True


def filter_population_by_thresholds(genomes: List[Genome], 
                                   score_extractor, 
                                   thresholds: ObjectiveThresholds) -> List[Genome]:
    """Filter population by threshold gate."""
    survivors = []
    
    for genome in genomes:
        if not genome.is_evaluated():
            continue
            
        # Extract multi-objective scores from genome
        scores = score_extractor(genome)
        
        # Apply threshold gate
        if apply_threshold_gate(scores, thresholds):
            survivors.append(genome)
    
    return survivors 