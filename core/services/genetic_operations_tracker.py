"""
Genetic Operations Tracker for CORAL-X Evolution System.
Tracks crossovers, mutations, and their evolutionary success patterns.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time
import numpy as np
from collections import defaultdict

from ..domain.genome import Genome
from ..domain.ca import CASeed  
from ..domain.mapping import LoRAConfig


@dataclass(frozen=True)
class CrossoverRecord:
    """Record of a crossover genetic operation."""
    child_id: str
    parent1_id: str
    parent2_id: str
    generation: int
    timestamp: float
    
    # Parent characteristics
    parent1_ca_features: Dict[str, float]
    parent2_ca_features: Dict[str, float]
    parent1_lora_config: Dict[str, Any]
    parent2_lora_config: Dict[str, Any]
    
    # Child characteristics
    child_ca_features: Dict[str, float]
    child_lora_config: Dict[str, Any]
    
    # Crossover details (with defaults)
    parent1_fitness: Optional[float] = None
    parent2_fitness: Optional[float] = None
    child_fitness: Optional[float] = None
    ca_crossover_details: Optional[Dict[str, Any]] = None  # Which parent contributed what
    diversity_strength: float = 1.0
    
    # Success metrics (filled in after evaluation)
    success_score: Optional[float] = None
    outperformed_parents: Optional[bool] = None


@dataclass(frozen=True)
class MutationRecord:
    """Record of a mutation genetic operation."""
    child_id: str
    parent_id: str
    generation: int
    timestamp: float
    
    # Parent characteristics
    parent_ca_features: Dict[str, float]
    parent_lora_config: Dict[str, Any]
    
    # Child characteristics
    child_ca_features: Dict[str, float]
    child_lora_config: Dict[str, Any]
    
    # Mutation details (with defaults)
    mutation_type: str = "ca_mutation"  # "ca_mutation" or "lora_mutation"
    parent_fitness: Optional[float] = None
    child_fitness: Optional[float] = None
    ca_mutations: Optional[Dict[str, Any]] = None  # Grid, rule, steps changes
    lora_mutations: Optional[Dict[str, Any]] = None  # Rank, alpha, dropout changes
    diversity_strength: float = 1.0
    
    # Success metrics (filled in after evaluation)
    success_score: Optional[float] = None
    improvement_over_parent: Optional[float] = None


@dataclass(frozen=True)
class GeneticPattern:
    """Detected pattern in genetic operations."""
    pattern_type: str  # "successful_crossover", "beneficial_mutation", etc.
    pattern_name: str
    confidence: float
    generation_detected: int
    evidence: Dict[str, Any]
    description: str


class GeneticOperationsTracker:
    """Tracks genetic operations and detects evolutionary patterns."""
    
    def __init__(self, output_dir: str = "results/genetic_tracking"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.crossover_records: List[CrossoverRecord] = []
        self.mutation_records: List[MutationRecord] = []
        self.detected_patterns: List[GeneticPattern] = []
        
        # Statistics
        self.generation_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "crossovers": 0,
            "mutations": 0,
            "ca_mutations": 0,
            "lora_mutations": 0,
            "successful_crossovers": 0,
            "successful_mutations": 0,
            "avg_crossover_success": 0.0,
            "avg_mutation_success": 0.0
        })
        
        print(f"🧬 Genetic Operations Tracker initialized")
        print(f"   📂 Output directory: {self.output_dir}")
    
    def track_crossover(self, child: Genome, parent1: Genome, parent2: Genome, 
                       generation: int, diversity_strength: float = 1.0,
                       ca_crossover_details: Dict[str, Any] = None) -> CrossoverRecord:
        """Track a crossover operation."""
        
        # Extract CA features for all genomes
        child_ca_features = self._extract_ca_features(child)
        parent1_ca_features = self._extract_ca_features(parent1)
        parent2_ca_features = self._extract_ca_features(parent2)
        
        # Extract LoRA configs
        child_lora = self._lora_config_to_dict(child.lora_cfg)
        parent1_lora = self._lora_config_to_dict(parent1.lora_cfg)
        parent2_lora = self._lora_config_to_dict(parent2.lora_cfg)
        
        record = CrossoverRecord(
            child_id=child.id,
            parent1_id=parent1.id,
            parent2_id=parent2.id,
            generation=generation,
            timestamp=time.time(),
            parent1_ca_features=parent1_ca_features,
            parent2_ca_features=parent2_ca_features,
            parent1_lora_config=parent1_lora,
            parent2_lora_config=parent2_lora,
            parent1_fitness=parent1.fitness,
            parent2_fitness=parent2.fitness,
            child_ca_features=child_ca_features,
            child_lora_config=child_lora,
            diversity_strength=diversity_strength,
            ca_crossover_details=ca_crossover_details or {}
        )
        
        self.crossover_records.append(record)
        self.generation_stats[generation]["crossovers"] += 1
        
        print(f"🔀 CROSSOVER TRACKED")
        print(f"   👨 Parent 1: {parent1.id} (fitness: {parent1.fitness})")
        print(f"   👩 Parent 2: {parent2.id} (fitness: {parent2.fitness})")
        print(f"   👶 Child: {child.id}")
        print(f"   🧬 CA Hybrid: complexity={child_ca_features.get('complexity', 0):.3f}")
        print(f"   🎛️  LoRA: r={child_lora['r']}, α={child_lora['alpha']}")
        
        return record
    
    def track_mutation(self, child: Genome, parent: Genome, generation: int,
                      mutation_type: str, mutation_details: Dict[str, Any] = None,
                      diversity_strength: float = 1.0) -> MutationRecord:
        """Track a mutation operation."""
        
        # Extract CA features
        child_ca_features = self._extract_ca_features(child)
        parent_ca_features = self._extract_ca_features(parent)
        
        # Extract LoRA configs
        child_lora = self._lora_config_to_dict(child.lora_cfg)
        parent_lora = self._lora_config_to_dict(parent.lora_cfg)
        
        # Determine mutation details
        ca_mutations = None
        lora_mutations = None
        
        if mutation_type == "ca_mutation":
            ca_mutations = self._analyze_ca_mutations(parent.seed, child.seed)
        elif mutation_type == "lora_mutation":
            lora_mutations = self._analyze_lora_mutations(parent.lora_cfg, child.lora_cfg)
        
        record = MutationRecord(
            child_id=child.id,
            parent_id=parent.id,
            generation=generation,
            timestamp=time.time(),
            parent_ca_features=parent_ca_features,
            parent_lora_config=parent_lora,
            parent_fitness=parent.fitness,
            child_ca_features=child_ca_features,
            child_lora_config=child_lora,
            mutation_type=mutation_type,
            ca_mutations=ca_mutations,
            lora_mutations=lora_mutations,
            diversity_strength=diversity_strength
        )
        
        self.mutation_records.append(record)
        self.generation_stats[generation]["mutations"] += 1
        self.generation_stats[generation][f"{mutation_type}s"] += 1
        
        print(f"🧬 MUTATION TRACKED")
        print(f"   👨 Parent: {parent.id} (fitness: {parent.fitness})")
        print(f"   👶 Child: {child.id}")
        print(f"   🔄 Type: {mutation_type}")
        if ca_mutations:
            print(f"   🌊 CA changes: {ca_mutations}")
        if lora_mutations:
            print(f"   🎛️  LoRA changes: {lora_mutations}")
        
        return record
    
    def update_fitness_outcomes(self, genome_id: str, fitness: float, 
                               multi_scores: Dict[str, float] = None):
        """Update fitness outcomes for tracked genetic operations."""
        
        # Update crossover records
        for i, record in enumerate(self.crossover_records):
            if record.child_id == genome_id and record.child_fitness is None:
                # Calculate success metrics
                parent_fitnesses = [f for f in [record.parent1_fitness, record.parent2_fitness] if f is not None]
                avg_parent_fitness = sum(parent_fitnesses) / len(parent_fitnesses) if parent_fitnesses else 0.0
                
                outperformed = fitness > avg_parent_fitness if parent_fitnesses else None
                success_score = fitness / avg_parent_fitness if avg_parent_fitness > 0 else 1.0
                
                # Create updated record
                updated_record = CrossoverRecord(
                    child_id=record.child_id,
                    parent1_id=record.parent1_id,
                    parent2_id=record.parent2_id,
                    generation=record.generation,
                    timestamp=record.timestamp,
                    parent1_ca_features=record.parent1_ca_features,
                    parent2_ca_features=record.parent2_ca_features,
                    parent1_lora_config=record.parent1_lora_config,
                    parent2_lora_config=record.parent2_lora_config,
                    parent1_fitness=record.parent1_fitness,
                    parent2_fitness=record.parent2_fitness,
                    child_ca_features=record.child_ca_features,
                    child_lora_config=record.child_lora_config,
                    child_fitness=fitness,
                    ca_crossover_details=record.ca_crossover_details,
                    diversity_strength=record.diversity_strength,
                    success_score=success_score,
                    outperformed_parents=outperformed
                )
                
                self.crossover_records[i] = updated_record
                
                if outperformed:
                    self.generation_stats[record.generation]["successful_crossovers"] += 1
                    print(f"✅ SUCCESSFUL CROSSOVER: {genome_id} (fitness: {fitness:.3f}, improvement: {success_score:.2f}x)")
                
                break
        
        # Update mutation records  
        for i, record in enumerate(self.mutation_records):
            if record.child_id == genome_id and record.child_fitness is None:
                # Calculate success metrics
                improvement = fitness - record.parent_fitness if record.parent_fitness is not None else 0.0
                success_score = fitness / record.parent_fitness if record.parent_fitness and record.parent_fitness > 0 else 1.0
                
                # Create updated record
                updated_record = MutationRecord(
                    child_id=record.child_id,
                    parent_id=record.parent_id,
                    generation=record.generation,
                    timestamp=record.timestamp,
                    parent_ca_features=record.parent_ca_features,
                    parent_lora_config=record.parent_lora_config,
                    parent_fitness=record.parent_fitness,
                    child_ca_features=record.child_ca_features,
                    child_lora_config=record.child_lora_config,
                    child_fitness=fitness,
                    mutation_type=record.mutation_type,
                    ca_mutations=record.ca_mutations,
                    lora_mutations=record.lora_mutations,
                    diversity_strength=record.diversity_strength,
                    success_score=success_score,
                    improvement_over_parent=improvement
                )
                
                self.mutation_records[i] = updated_record
                
                if improvement > 0:
                    self.generation_stats[record.generation]["successful_mutations"] += 1
                    print(f"✅ SUCCESSFUL MUTATION: {genome_id} (fitness: {fitness:.3f}, improvement: +{improvement:.3f})")
                
                break
    
    def detect_genetic_patterns(self, generation: int) -> List[GeneticPattern]:
        """Detect patterns in genetic operations."""
        patterns = []
        
        # Pattern 1: Successful crossover combinations
        successful_crossovers = [r for r in self.crossover_records 
                               if r.generation <= generation and r.outperformed_parents]
        
        if len(successful_crossovers) >= 3:
            # Look for CA feature combinations that work well
            complexity_combinations = [(r.parent1_ca_features.get('complexity', 0), 
                                     r.parent2_ca_features.get('complexity', 0))
                                     for r in successful_crossovers]
            
            # Check if high-complexity × low-complexity crossovers are successful
            high_low_crosses = sum(1 for c1, c2 in complexity_combinations 
                                 if abs(c1 - c2) > 0.4)
            
            if high_low_crosses / len(complexity_combinations) > 0.6:
                patterns.append(GeneticPattern(
                    pattern_type="genetic_diversity",
                    pattern_name="complexity_hybrid_vigor",
                    confidence=high_low_crosses / len(complexity_combinations),
                    generation_detected=generation,
                    evidence={
                        "successful_crossovers": len(successful_crossovers),
                        "high_low_crosses": high_low_crosses,
                        "avg_success_score": sum(r.success_score for r in successful_crossovers) / len(successful_crossovers)
                    },
                    description=f"High-complexity × Low-complexity crossovers show hybrid vigor ({high_low_crosses}/{len(complexity_combinations)} cases)"
                ))
        
        # Pattern 2: Beneficial mutation types  
        successful_mutations = [r for r in self.mutation_records
                              if r.generation <= generation and r.improvement_over_parent and r.improvement_over_parent > 0]
        
        if len(successful_mutations) >= 5:
            ca_success_rate = sum(1 for r in successful_mutations if r.mutation_type == "ca_mutation") / len(successful_mutations)
            
            if ca_success_rate > 0.7:
                patterns.append(GeneticPattern(
                    pattern_type="mutation_preference",
                    pattern_name="ca_mutation_advantage",
                    confidence=ca_success_rate,
                    generation_detected=generation,
                    evidence={
                        "total_successful_mutations": len(successful_mutations),
                        "ca_mutations": sum(1 for r in successful_mutations if r.mutation_type == "ca_mutation"),
                        "avg_improvement": sum(r.improvement_over_parent for r in successful_mutations) / len(successful_mutations)
                    },
                    description=f"CA mutations show higher success rate than LoRA mutations ({ca_success_rate:.1%})"
                ))
        
        # Pattern 3: Generation-based success trends
        recent_crossovers = [r for r in self.crossover_records if r.generation >= generation - 2]
        if len(recent_crossovers) >= 5:
            success_rate = sum(1 for r in recent_crossovers if r.outperformed_parents) / len(recent_crossovers)
            
            if success_rate > 0.8:
                patterns.append(GeneticPattern(
                    pattern_type="evolution_acceleration",
                    pattern_name="high_crossover_success",
                    confidence=success_rate,
                    generation_detected=generation,
                    evidence={
                        "recent_crossovers": len(recent_crossovers),
                        "success_rate": success_rate,
                        "generation_range": f"{generation-2}-{generation}"
                    },
                    description=f"Recent crossovers show high success rate ({success_rate:.1%}) - evolution accelerating"
                ))
        
        # Store detected patterns
        self.detected_patterns.extend(patterns)
        
        # Alert on significant patterns
        for pattern in patterns:
            if pattern.confidence > 0.7:
                print(f"\n🎯 GENETIC PATTERN DETECTED")
                print(f"   🧬 Pattern: {pattern.pattern_name}")  
                print(f"   📊 Confidence: {pattern.confidence:.1%}")
                print(f"   💡 {pattern.description}")
        
        return patterns
    
    def save_tracking_data(self, generation: int):
        """Save all tracking data to files."""
        timestamp = int(time.time())
        
        # Save crossover records
        crossover_file = self.output_dir / f"crossovers_gen{generation}_{timestamp}.json"
        crossover_data = [self._record_to_dict(r) for r in self.crossover_records]
        with open(crossover_file, 'w') as f:
            json.dump(crossover_data, f, indent=2)
        
        # Save mutation records
        mutation_file = self.output_dir / f"mutations_gen{generation}_{timestamp}.json"
        mutation_data = [self._record_to_dict(r) for r in self.mutation_records]
        with open(mutation_file, 'w') as f:
            json.dump(mutation_data, f, indent=2)
        
        # Save generation statistics
        stats_file = self.output_dir / f"genetic_stats_gen{generation}_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(dict(self.generation_stats), f, indent=2)
        
        # Save detected patterns
        patterns_file = self.output_dir / f"genetic_patterns_gen{generation}_{timestamp}.json"
        patterns_data = [self._pattern_to_dict(p) for p in self.detected_patterns]
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        print(f"💾 Genetic tracking data saved for generation {generation}")
    
    def get_generation_summary(self, generation: int) -> Dict[str, Any]:
        """Get summary of genetic operations for a generation."""
        stats = self.generation_stats[generation]
        
        # Calculate success rates
        crossover_success_rate = (stats["successful_crossovers"] / max(1, stats["crossovers"])) * 100
        mutation_success_rate = (stats["successful_mutations"] / max(1, stats["mutations"])) * 100
        
        return {
            "generation": generation,
            "total_operations": stats["crossovers"] + stats["mutations"],
            "crossovers": stats["crossovers"],
            "mutations": stats["mutations"],
            "ca_mutations": stats["ca_mutations"],
            "lora_mutations": stats["lora_mutations"],
            "crossover_success_rate": crossover_success_rate,
            "mutation_success_rate": mutation_success_rate,
            "patterns_detected": len([p for p in self.detected_patterns if p.generation_detected == generation])
        }
    
    def _extract_ca_features(self, genome: Genome) -> Dict[str, float]:
        """Extract CA features from a genome."""
        try:
            from .ca import evolve
            from .feature_extraction import extract_features
            
            history = evolve(genome.seed)
            features = extract_features(history)
            
            return {
                "complexity": features.complexity,
                "intensity": features.intensity,
                "periodicity": features.periodicity,
                "convergence": features.convergence
            }
        except Exception as e:
            print(f"⚠️ Failed to extract CA features for {genome.id}: {e}")
            return {"complexity": 0.0, "intensity": 0.0, "periodicity": 0.0, "convergence": 0.0}
    
    def _lora_config_to_dict(self, lora_cfg: LoRAConfig) -> Dict[str, Any]:
        """Convert LoRA config to dictionary."""
        return {
            "r": lora_cfg.r,
            "alpha": lora_cfg.alpha,
            "dropout": lora_cfg.dropout,
            "target_modules": list(lora_cfg.target_modules)
        }
    
    def _analyze_ca_mutations(self, parent_seed: CASeed, child_seed: CASeed) -> Dict[str, Any]:
        """Analyze what changed in a CA mutation."""
        changes = {}
        
        # Check grid changes
        if not np.array_equal(parent_seed.grid, child_seed.grid):
            diff_positions = np.where(parent_seed.grid != child_seed.grid)
            changes["grid_mutations"] = len(diff_positions[0]) if len(diff_positions) > 0 else 0
        
        # Check rule changes
        if parent_seed.rule != child_seed.rule:
            changes["rule_change"] = {
                "from": parent_seed.rule,
                "to": child_seed.rule,
                "delta": child_seed.rule - parent_seed.rule
            }
        
        # Check steps changes
        if parent_seed.steps != child_seed.steps:
            changes["steps_change"] = {
                "from": parent_seed.steps,
                "to": child_seed.steps,
                "delta": child_seed.steps - parent_seed.steps
            }
        
        return changes
    
    def _analyze_lora_mutations(self, parent_cfg: LoRAConfig, child_cfg: LoRAConfig) -> Dict[str, Any]:
        """Analyze what changed in a LoRA mutation."""
        changes = {}
        
        if parent_cfg.r != child_cfg.r:
            changes["rank_change"] = {"from": parent_cfg.r, "to": child_cfg.r}
        
        if parent_cfg.alpha != child_cfg.alpha:
            changes["alpha_change"] = {"from": parent_cfg.alpha, "to": child_cfg.alpha}
        
        if parent_cfg.dropout != child_cfg.dropout:
            changes["dropout_change"] = {"from": parent_cfg.dropout, "to": child_cfg.dropout}
        
        return changes
    
    def _record_to_dict(self, record) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        if isinstance(record, CrossoverRecord):
            return {
                "type": "crossover",
                "child_id": record.child_id,
                "parent1_id": record.parent1_id,
                "parent2_id": record.parent2_id,
                "generation": record.generation,
                "timestamp": record.timestamp,
                "parent1_fitness": record.parent1_fitness,
                "parent2_fitness": record.parent2_fitness,
                "child_fitness": record.child_fitness,
                "success_score": record.success_score,
                "outperformed_parents": record.outperformed_parents,
                "diversity_strength": record.diversity_strength,
                "child_lora_config": record.child_lora_config
            }
        elif isinstance(record, MutationRecord):
            return {
                "type": "mutation",
                "child_id": record.child_id,
                "parent_id": record.parent_id,
                "generation": record.generation,
                "timestamp": record.timestamp,
                "parent_fitness": record.parent_fitness,
                "child_fitness": record.child_fitness,
                "mutation_type": record.mutation_type,
                "success_score": record.success_score,
                "improvement_over_parent": record.improvement_over_parent,
                "ca_mutations": record.ca_mutations,
                "lora_mutations": record.lora_mutations,
                "diversity_strength": record.diversity_strength
            }
    
    def _pattern_to_dict(self, pattern: GeneticPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary for JSON serialization."""
        return {
            "pattern_type": pattern.pattern_type,
            "pattern_name": pattern.pattern_name,
            "confidence": pattern.confidence,
            "generation_detected": pattern.generation_detected,
            "evidence": pattern.evidence,
            "description": pattern.description
        } 