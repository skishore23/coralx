###############################################################################
# Genome definition - immutable combination of CA and LoRA parameters
###############################################################################
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .ca import CASeed
from .mapping import LoRAConfig


@dataclass(frozen=True)
class MultiObjectiveScores:
    """Multi-objective evaluation scores for CORAL-X."""
    bugfix: float
    style: float
    security: float
    runtime: float
    syntax: float  # NEW: Syntax correctness score
    
    def overall_fitness(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall fitness as weighted average."""
        # Use provided weights or default CORAL-X priorities
        if weights is None:
            weights = {'bugfix': 0.3, 'style': 0.15, 'security': 0.25, 'runtime': 0.1, 'syntax': 0.2}
        
        return (
            self.bugfix * weights.get('bugfix', 0.3) +
            self.style * weights.get('style', 0.15) +
            self.security * weights.get('security', 0.25) +
            self.runtime * weights.get('runtime', 0.1) +
            self.syntax * weights.get('syntax', 0.2)
        )
    
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
class Genome:
    """CA seed + current LoRA params + multi-objective scores."""
    seed: CASeed
    lora_cfg: LoRAConfig
    id: str  # Unique genome identifier
    fitness: Optional[float] = None
    multi_scores: Optional[MultiObjectiveScores] = None
    metadata: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None  # Experiment-specific identifier
    
    def with_fitness(self, fitness: float) -> 'Genome':
        """Return new genome with updated fitness score."""
        return Genome(
            seed=self.seed,
            lora_cfg=self.lora_cfg,
            id=self.id,
            fitness=fitness,
            multi_scores=self.multi_scores,
            metadata=self.metadata,
            run_id=self.run_id
        )
    
    def with_multi_scores(self, scores: MultiObjectiveScores) -> 'Genome':
        """Return new genome with multi-objective scores."""
        # Update overall fitness based on multi-objective scores
        overall_fitness = scores.overall_fitness()
        return Genome(
            seed=self.seed,
            lora_cfg=self.lora_cfg,
            id=self.id,
            fitness=overall_fitness,
            multi_scores=scores,
            metadata=self.metadata,
            run_id=self.run_id
        )
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'Genome':
        """Return new genome with updated metadata."""
        return Genome(
            seed=self.seed,
            lora_cfg=self.lora_cfg,
            id=self.id,
            fitness=self.fitness,
            multi_scores=self.multi_scores,
            metadata=metadata,
            run_id=self.run_id
        )
    
    def is_evaluated(self) -> bool:
        """Check if genome has been evaluated."""
        return self.fitness is not None
    
    def has_multi_scores(self) -> bool:
        """Check if genome has multi-objective scores."""
        return self.multi_scores is not None
    
    def get_heavy_genes_key(self) -> tuple:
        """Extract heavy genes that require adapter training."""
        return (
            self.lora_cfg.r,             # 🔥 FIXED: Use PEFT convention (r)
            self.lora_cfg.alpha,
            self.lora_cfg.dropout,
            self.lora_cfg.target_modules,
            self.lora_cfg.adapter_type,  # 🔥 Include adapter_type in heavy genes
            self.run_id                  # 🔥 Include run_id for experiment isolation
        )
    
    def __lt__(self, other: 'Genome') -> bool:
        """Support sorting by fitness (higher is better)."""
        if self.fitness is None and other.fitness is None:
            return False
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness 