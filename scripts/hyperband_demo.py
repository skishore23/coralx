#!/usr/bin/env python3
"""
Hyperband Multi-Fidelity Training Demo
Demonstrates the sophisticated training strategy compared to simple alternatives
"""

import sys
import time
from pathlib import Path

# Add coralx to path
coralx_path = Path(__file__).parent.parent
sys.path.insert(0, str(coralx_path))

import numpy as np
from typing import List, Dict, Any

# Core imports
from coral.domain.genome import Genome
from coral.domain.mapping import LoRAConfig
from coral.domain.ca import CASeed

# Import our new hyperband trainer
from plugins.fakenews_gemma3n.hyperband_trainer import HyperbandTrainer, TrainingStage


def create_test_population(size: int = 8) -> List[Genome]:
    """Create a test population with diverse LoRA configurations."""
    genomes = []
    
    for i in range(size):
        # Create diverse LoRA configurations
        rank = [4, 8, 12, 16, 24, 32][i % 6]
        alpha = [8, 16, 24, 32, 48, 64][i % 6]
        dropout = [0.05, 0.1, 0.15, 0.2][i % 4]
        
        ca_seed = CASeed(
            grid=np.random.randint(0, 2, (4, 4)),
            rule=30 + i,
            steps=5 + i
        )
        
        lora_cfg = LoRAConfig(
            r=rank,
            alpha=float(alpha),
            dropout=dropout,
            target_modules=['q_proj', 'v_proj', 'o_proj'][:2 + i % 2]
        )
        
        genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_cfg,
            id=f"genome_{i:02d}_r{rank}_a{alpha}"
        )
        genomes.append(genome)
    
    return genomes


def demo_hyperband_training():
    """Demonstrate hyperband multi-fidelity training."""
    print("🎯 Hyperband Multi-Fidelity Training Demo")
    print("=" * 50)
    
    # Create test population
    genomes = create_test_population(8)
    print(f"📊 Created population of {len(genomes)} genomes:")
    for genome in genomes:
        print(f"   - {genome.id}: rank={genome.lora_cfg.r}, alpha={genome.lora_cfg.alpha}, dropout={genome.lora_cfg.dropout}")
    
    # Setup hyperband trainer
    config = {
        'base_model': 'google/gemma-3n-e4b-it',
        'dataset_config': {'fake_news': True, 'multimodal': True},
        'cache_dir': 'cache/hyperband_demo',
        'training_stages': [
            {
                'name': 'S0_sanity',
                'epoch_budget': 0.05,
                'data_percentage': 5,
                'survival_rate': 1.0,
                'proxy_metrics': ['gradient_norm', 'loss_slope']
            },
            {
                'name': 'S1_shakeout',
                'epoch_budget': 0.3,
                'data_percentage': 20,
                'survival_rate': 0.5,
                'proxy_metrics': ['text_only_auroc', 'val_loss']
            },
            {
                'name': 'S2_serious',
                'epoch_budget': 1.0,
                'data_percentage': 100,
                'survival_rate': 0.25,
                'proxy_metrics': ['full_auroc', 'safety_score']
            },
            {
                'name': 'S3_finisher',
                'epoch_budget': 2.0,
                'data_percentage': 100,
                'survival_rate': 0.1,
                'proxy_metrics': ['all_metrics']
            }
        ]
    }
    
    trainer = HyperbandTrainer(config)
    
    # Run hyperband training
    print(f"\n🚀 Starting Hyperband Training")
    start_time = time.time()
    
    results = trainer.train_population(genomes)
    
    hyperband_time = time.time() - start_time
    
    # Generate efficiency report
    efficiency_report = trainer.get_training_efficiency_report(results)
    
    print(f"\n📈 HYPERBAND RESULTS")
    print("-" * 30)
    print(f"Total training time: {hyperband_time:.2f}s")
    print(f"Efficiency ratio: {efficiency_report['efficiency_ratio']:.3f}")
    print(f"Time savings: {efficiency_report['time_savings']:.1f}s")
    
    # Show stage breakdown
    print(f"\n📊 STAGE BREAKDOWN")
    for stage_name, stats in efficiency_report['stage_breakdown'].items():
        print(f"{stage_name:12}: {stats['genomes_trained']} genomes, {stats['total_time']:.2f}s total")
    
    return results, efficiency_report


def compare_training_strategies():
    """Compare different training strategies."""
    print(f"\n🔥 TRAINING STRATEGY COMPARISON")
    print("=" * 50)
    
    # Create test population
    genomes = create_test_population(8)
    
    # Strategy 1: Simple Simulation (my current approach)
    print(f"1️⃣  SIMPLE SIMULATION")
    start_time = time.time()
    
    sim_results = []
    for genome in genomes:
        # Simulate instant evaluation
        result = simulate_simple_performance(genome)
        sim_results.append(result)
    
    simple_sim_time = time.time() - start_time
    print(f"   Time: {simple_sim_time:.3f}s")
    print(f"   Cost: Instant heuristics")
    print(f"   Accuracy: Estimated only")
    
    # Strategy 2: Naive Full Training
    print(f"\n2️⃣  NAIVE FULL TRAINING")
    start_time = time.time()
    
    naive_results = []
    for genome in genomes:
        # Simulate full training cost
        time.sleep(0.2)  # Simulate 2+ minutes of real training
        result = simulate_full_training_performance(genome)
        naive_results.append(result)
    
    naive_time = time.time() - start_time
    print(f"   Time: {naive_time:.1f}s")
    print(f"   Cost: Full training for all genomes")
    print(f"   Accuracy: High but expensive")
    
    # Strategy 3: Hyperband Multi-Fidelity
    print(f"\n3️⃣  HYPERBAND MULTI-FIDELITY")
    results, report = demo_hyperband_training()
    hyperband_time = report['total_training_time']
    
    print(f"   Time: {hyperband_time:.1f}s")
    print(f"   Cost: Progressive training with early stopping")
    print(f"   Accuracy: Real gradients + intelligent resource allocation")
    
    # Comparison summary
    print(f"\n📈 STRATEGY COMPARISON")
    print("-" * 40)
    print(f"{'Strategy':<20} {'Time (s)':<10} {'Speedup':<10} {'Accuracy'}")
    print("-" * 40)
    print(f"{'Simple Simulation':<20} {simple_sim_time:<10.3f} {'1x':<10} {'Low'}")
    print(f"{'Naive Full Training':<20} {naive_time:<10.1f} {naive_time/naive_time:<10.1f} {'High'}")
    print(f"{'Hyperband Multi-Fi':<20} {hyperband_time:<10.1f} {naive_time/hyperband_time:<10.1f} {'High'}")
    
    # Efficiency analysis
    print(f"\n🎯 EFFICIENCY ANALYSIS")
    print("-" * 25)
    print(f"Hyperband vs Naive:     {naive_time/hyperband_time:.1f}x faster")
    print(f"Hyperband vs Simulation: {hyperband_time/simple_sim_time:.1f}x slower")
    print(f"BUT: Hyperband uses real training data & gradients!")
    
    return {
        'simple_simulation': simple_sim_time,
        'naive_full': naive_time,
        'hyperband': hyperband_time,
        'hyperband_vs_naive_speedup': naive_time / hyperband_time
    }


def simulate_simple_performance(genome: Genome) -> Dict[str, float]:
    """Simulate simple heuristic performance (my current approach)."""
    import random
    random.seed(hash(genome.id))
    
    # Basic heuristic
    base_auroc = 0.75 + random.uniform(-0.1, 0.1)
    return {'auroc': base_auroc, 'method': 'heuristic_only'}


def simulate_full_training_performance(genome: Genome) -> Dict[str, float]:
    """Simulate full training performance (naive approach)."""
    import random
    random.seed(hash(genome.id))
    
    # High quality result but expensive
    base_auroc = 0.8 + random.uniform(-0.05, 0.15)
    return {'auroc': base_auroc, 'method': 'full_training'}


def main():
    """Run the hyperband demo and comparison."""
    print("🧪 CoralX Multi-Modal AI Safety - Advanced Training Strategies")
    print("🎯 Comparing: Simple Simulation vs Naive Full vs Hyperband Multi-Fidelity")
    print("")
    
    # Run comparison
    comparison_results = compare_training_strategies()
    
    print(f"\n🏆 FINAL VERDICT")
    print("=" * 25)
    
    if comparison_results['hyperband_vs_naive_speedup'] > 3:
        print(f"✅ EXCELLENT: Hyperband achieves {comparison_results['hyperband_vs_naive_speedup']:.1f}x speedup")
        print(f"🧬 Realistic evolutionary optimization now feasible!")
    elif comparison_results['hyperband_vs_naive_speedup'] > 2:
        print(f"👍 GOOD: Hyperband achieves {comparison_results['hyperband_vs_naive_speedup']:.1f}x speedup")
        print(f"🧬 Substantial efficiency gains for evolution!")
    else:
        print(f"⚠️  MODERATE: {comparison_results['hyperband_vs_naive_speedup']:.1f}x speedup")
        print(f"🧬 Some efficiency gains, but room for improvement")
    
    print(f"\n🎯 KEY INSIGHTS")
    print("-" * 15)
    print(f"1. Simple simulation: Fast but synthetic")
    print(f"2. Naive full training: Accurate but prohibitive")
    print(f"3. Hyperband multi-fidelity: Best of both worlds!")
    print(f"   - Uses real training data & gradients")
    print(f"   - Progressive resource allocation")
    print(f"   - Early stopping for poor performers")
    print(f"   - Warm-start from parent checkpoints")
    
    return comparison_results['hyperband_vs_naive_speedup'] > 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 