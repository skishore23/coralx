"""
Hyperband Multi-Fidelity Training for CoralX Multi-Modal AI Safety
Implements progressive training stages with successive halving
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import time
import yaml

from coral.domain.genome import Genome
from coral.domain.mapping import LoRAConfig


@dataclass(frozen=True)
class TrainingStage:
    """Define a training stage in the hyperband progression."""
    name: str
    epoch_budget: float      # Fraction of full epochs
    data_percentage: int     # Percentage of dataset to use
    survival_rate: float     # What fraction advances to next stage
    proxy_metrics: List[str] # Cheap metrics to evaluate
    

@dataclass(frozen=True)
class TrainingResult:
    """Result from a training stage."""
    genome_id: str
    stage: str
    metrics: Dict[str, float]
    training_time: float
    should_continue: bool
    checkpoint_path: Optional[str] = None


class HyperbandTrainer:
    """Multi-fidelity training with successive halving for LoRA optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        self._setup_stages()
    
    def _validate_config(self):
        """Fail-fast configuration validation."""
        required_fields = ['base_model', 'dataset_config', 'training_stages']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"FAIL-FAST: '{field}' missing from hyperband config")
    
    def _setup_stages(self):
        """Setup the multi-fidelity training stages."""
        # Default hyperband progression if not specified
        default_stages = [
            TrainingStage(
                name="S0_sanity",
                epoch_budget=0.05,
                data_percentage=5,
                survival_rate=1.0,  # All genomes pass sanity check
                proxy_metrics=["gradient_norm", "loss_slope"]
            ),
            TrainingStage(
                name="S1_shakeout", 
                epoch_budget=0.3,
                data_percentage=20,
                survival_rate=0.5,  # Top 50% advance
                proxy_metrics=["text_only_auroc", "val_loss"]
            ),
            TrainingStage(
                name="S2_serious",
                epoch_budget=1.0,
                data_percentage=100,
                survival_rate=0.25,  # Top 25% advance
                proxy_metrics=["full_auroc", "safety_score"]
            ),
            TrainingStage(
                name="S3_finisher",
                epoch_budget=2.0,
                data_percentage=100, 
                survival_rate=0.1,   # Top 10% complete full training
                proxy_metrics=["all_metrics"]
            )
        ]
        
        # Use config stages or defaults
        stage_configs = self.config.get('training_stages', default_stages)
        self.stages = stage_configs if isinstance(stage_configs[0], TrainingStage) else [
            TrainingStage(**stage) for stage in stage_configs
        ]
    
    def train_population(self, genomes: List[Genome]) -> List[TrainingResult]:
        """Train population using hyperband successive halving."""
        print(f"🏁 Starting Hyperband training with {len(genomes)} genomes")
        
        current_genomes = genomes.copy()
        all_results = []
        
        for stage_idx, stage in enumerate(self.stages):
            print(f"\n🎯 Stage {stage_idx}: {stage.name}")
            print(f"   Budget: {stage.epoch_budget} epochs on {stage.data_percentage}% data")
            print(f"   Training: {len(current_genomes)} genomes")
            
            # Train all surviving genomes at this stage
            stage_results = []
            for genome in current_genomes:
                result = self._train_single_genome(genome, stage)
                stage_results.append(result)
                all_results.append(result)
            
            # Early stopping based on proxy metrics
            stage_results = self._apply_early_stopping(stage_results, stage)
            
            # Select survivors for next stage
            if stage_idx < len(self.stages) - 1:  # Not the final stage
                current_genomes = self._select_survivors(stage_results, stage)
                print(f"   ✅ {len(current_genomes)} genomes advance to next stage")
            else:
                print(f"   🏆 Final stage complete: {len(stage_results)} genomes finished")
        
        return all_results
    
    def _train_single_genome(self, genome: Genome, stage: TrainingStage) -> TrainingResult:
        """Train a single genome for the given stage."""
        start_time = time.time()
        
        # Check for warm-start checkpoint
        checkpoint_path = self._get_checkpoint_path(genome, stage)
        parent_checkpoint = self._find_parent_checkpoint(genome, stage)
        
        print(f"      🔧 Training {genome.id} ({stage.name})")
        
        # Simulate training for now (replace with real training)
        metrics = self._simulate_stage_training(genome, stage, parent_checkpoint)
        
        training_time = time.time() - start_time
        
        # Save checkpoint for potential inheritance
        self._save_checkpoint(genome, stage, metrics, checkpoint_path)
        
        # Determine if genome should continue
        should_continue = self._evaluate_continuation(metrics, stage)
        
        return TrainingResult(
            genome_id=genome.id,
            stage=stage.name,
            metrics=metrics,
            training_time=training_time,
            should_continue=should_continue,
            checkpoint_path=checkpoint_path
        )
    
    def _simulate_stage_training(self, genome: Genome, stage: TrainingStage, 
                                parent_checkpoint: Optional[str]) -> Dict[str, float]:
        """Simulate training for a stage (replace with real training)."""
        import random
        
        # Use genome + stage for deterministic results
        random.seed(hash(f"{genome.id}_{stage.name}"))
        
        # Base performance scaling with stage progression
        base_multiplier = 0.3 + stage.epoch_budget * 0.7  # S0: 0.33, S3: 1.0
        
        # LoRA parameter effects
        rank_factor = min(1.0, genome.lora_cfg.r / 16.0)
        alpha_factor = min(1.0, genome.lora_cfg.alpha / 32.0)
        
        # Warm-start bonus if inheriting from parent
        inheritance_bonus = 0.05 if parent_checkpoint else 0.0
        
        # Progressive improvement with more data
        data_quality_factor = stage.data_percentage / 100.0
        
        # Calculate stage-specific metrics
        base_auroc = 0.6 + rank_factor * 0.2 + alpha_factor * 0.1
        stage_auroc = base_auroc * base_multiplier * data_quality_factor + inheritance_bonus
        
        base_safety = 0.7 + alpha_factor * 0.15
        stage_safety = base_safety * base_multiplier + inheritance_bonus
        
        # Add realistic noise
        noise = random.uniform(-0.03, 0.03)
        
        metrics = {
            'auroc': min(0.95, max(0.5, stage_auroc + noise)),
            'safety_score': min(0.98, max(0.6, stage_safety + noise)),
            'val_loss': 2.0 - stage_auroc + random.uniform(-0.2, 0.2),
            'gradient_norm': random.uniform(0.1, 2.0),
            'training_steps': int(stage.epoch_budget * 1000),  # Simulated steps
            'data_samples': stage.data_percentage * 10,  # Simulated sample count
        }
        
        # Stage-specific proxy metrics
        if stage.name == "S0_sanity":
            metrics['loss_slope'] = random.uniform(-0.5, 0.1)  # Should be negative
        elif stage.name == "S1_shakeout":
            metrics['text_only_auroc'] = metrics['auroc'] * 0.9  # Slightly lower
        
        return metrics
    
    def _apply_early_stopping(self, results: List[TrainingResult], 
                             stage: TrainingStage) -> List[TrainingResult]:
        """Apply early stopping based on proxy metrics."""
        filtered_results = []
        
        for result in results:
            should_stop = False
            
            # Check for training failures
            if stage.name == "S0_sanity":
                if result.metrics.get('gradient_norm', 0) > 5.0:
                    print(f"      ❌ {result.genome_id}: Gradient explosion (norm={result.metrics['gradient_norm']:.2f})")
                    should_stop = True
                elif result.metrics.get('loss_slope', 0) > 0:
                    print(f"      ❌ {result.genome_id}: Loss not decreasing (slope={result.metrics['loss_slope']:.3f})")
                    should_stop = True
            
            # Check for poor performance
            elif stage.name in ["S1_shakeout", "S2_serious"]:
                if result.metrics.get('auroc', 0) < 0.55:  # Below random
                    print(f"      ❌ {result.genome_id}: Poor AUROC ({result.metrics['auroc']:.3f})")
                    should_stop = True
            
            if not should_stop:
                filtered_results.append(result)
        
        return filtered_results
    
    def _select_survivors(self, results: List[TrainingResult], 
                         stage: TrainingStage) -> List[Genome]:
        """Select top performers to advance to next stage."""
        # Sort by primary metric (AUROC for detection tasks)
        sorted_results = sorted(results, 
                               key=lambda r: r.metrics.get('auroc', 0), 
                               reverse=True)
        
        # Select top fraction
        n_survivors = max(1, int(len(sorted_results) * stage.survival_rate))
        survivors = sorted_results[:n_survivors]
        
        # Convert back to genomes (would need to maintain genome mapping)
        # For now, simulate genome selection
        return [self._result_to_genome(result) for result in survivors]
    
    def _result_to_genome(self, result: TrainingResult) -> Genome:
        """Convert training result back to genome (placeholder)."""
        # In real implementation, maintain genome mapping
        # For now, create dummy genome
        from coral.domain.ca import CASeed
        
        return Genome(
            seed=CASeed(grid=np.zeros((3,3)), rule=30, steps=5),
            lora_cfg=LoRAConfig(r=8, alpha=16.0, dropout=0.1, target_modules=['q_proj']),
            id=result.genome_id
        )
    
    def _get_checkpoint_path(self, genome: Genome, stage: TrainingStage) -> str:
        """Get checkpoint path for genome at stage."""
        cache_dir = self.config.get('cache_dir', 'cache/hyperband')
        return f"{cache_dir}/{genome.id}_{stage.name}.pt"
    
    def _find_parent_checkpoint(self, genome: Genome, stage: TrainingStage) -> Optional[str]:
        """Find parent checkpoint for warm-start (Population-Based Training style)."""
        # In real PBT, this would find the best parent from previous generation
        # For now, simulate availability
        if stage.name != "S0_sanity":  # Can inherit from previous stage
            return f"cache/hyperband/{genome.id}_previous_stage.pt"
        return None
    
    def _save_checkpoint(self, genome: Genome, stage: TrainingStage, 
                        metrics: Dict[str, float], checkpoint_path: str):
        """Save training checkpoint and metadata."""
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata for tracking
        metadata = {
            'genome_id': genome.id,
            'stage': stage.name,
            'metrics': metrics,
            'lora_config': {
                'r': genome.lora_cfg.r,
                'alpha': genome.lora_cfg.alpha,
                'dropout': genome.lora_cfg.dropout
            }
        }
        
        metadata_path = checkpoint_path.replace('.pt', '_metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
    
    def _evaluate_continuation(self, metrics: Dict[str, float], 
                              stage: TrainingStage) -> bool:
        """Evaluate if genome should continue to next stage."""
        # Simple thresholding for now
        if stage.name == "S0_sanity":
            return metrics.get('gradient_norm', float('inf')) < 3.0
        elif stage.name == "S1_shakeout":
            return metrics.get('auroc', 0) > 0.6
        elif stage.name == "S2_serious":
            return metrics.get('auroc', 0) > 0.75
        else:
            return True  # Final stage
    
    def get_training_efficiency_report(self, results: List[TrainingResult]) -> Dict[str, Any]:
        """Generate efficiency report showing savings."""
        stage_costs = {}
        total_genomes = len(set(r.genome_id for r in results))
        
        for stage in self.stages:
            stage_results = [r for r in results if r.stage == stage.name]
            stage_costs[stage.name] = {
                'genomes_trained': len(stage_results),
                'avg_time': np.mean([r.training_time for r in stage_results]),
                'total_time': sum(r.training_time for r in stage_results),
                'epoch_budget': stage.epoch_budget,
                'survival_rate': stage.survival_rate
            }
        
        # Calculate savings vs full training
        total_actual_time = sum(sum(r.training_time for r in results if r.stage == s.name) 
                               for s in self.stages)
        full_training_time = total_genomes * stage_costs['S3_finisher']['avg_time'] * 2  # Estimate
        
        return {
            'total_genomes': total_genomes,
            'stage_breakdown': stage_costs,
            'total_training_time': total_actual_time,
            'estimated_full_training_time': full_training_time,
            'time_savings': full_training_time - total_actual_time,
            'efficiency_ratio': total_actual_time / full_training_time
        } 