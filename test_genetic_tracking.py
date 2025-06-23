#!/usr/bin/env python3
"""
Test script for genetic operations tracking in CORAL-X.
Quick validation that crossovers and mutations are being tracked properly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from coral.domain.genetic_operations_tracker import GeneticOperationsTracker
from coral.domain.genome import Genome, MultiObjectiveScores
from coral.domain.ca import CASeed
from coral.domain.mapping import LoRAConfig
import numpy as np
import time


def create_test_genome(genome_id: str, fitness: float = None) -> Genome:
    """Create a test genome for tracking."""
    # Simple test CA seed
    grid = np.random.randint(0, 2, (4, 4))
    ca_seed = CASeed(grid=grid, rule=30, steps=10)
    
    # Simple test LoRA config
    lora_config = LoRAConfig(
        r=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    genome = Genome(seed=ca_seed, lora_cfg=lora_config, id=genome_id)
    
    # Add fitness if provided
    if fitness is not None:
        scores = MultiObjectiveScores(
            bugfix=fitness * 0.8,
            style=fitness * 0.9,
            security=fitness * 0.85,
            runtime=fitness * 0.7,
            syntax=fitness * 0.95
        )
        genome = genome.with_multi_scores(scores)
    
    return genome


def test_genetic_tracking():
    """Test genetic operations tracking."""
    
    print("🧬 Testing Genetic Operations Tracking")
    print("=" * 50)
    
    # Initialize tracker
    tracker = GeneticOperationsTracker(output_dir="test_output/genetic_tracking")
    
    # Create test genomes
    parent1 = create_test_genome("parent1", fitness=0.6)
    parent2 = create_test_genome("parent2", fitness=0.7)
    child_cross = create_test_genome("child_cross")
    child_mut = create_test_genome("child_mut")
    
    print("✅ Test genomes created")
    
    # Test crossover tracking
    print("\n🔀 Testing crossover tracking...")
    crossover_record = tracker.track_crossover(
        child=child_cross,
        parent1=parent1,
        parent2=parent2,
        generation=1,
        diversity_strength=1.2
    )
    
    print(f"   • Crossover record created: {crossover_record.child_id}")
    
    # Test mutation tracking
    print("\n🧬 Testing mutation tracking...")
    mutation_record = tracker.track_mutation(
        child=child_mut,
        parent=parent1,
        generation=1,
        mutation_type="ca_mutation",
        diversity_strength=1.0
    )
    
    print(f"   • Mutation record created: {mutation_record.child_id}")
    
    # Update fitness outcomes
    print("\n📈 Testing fitness updates...")
    tracker.update_fitness_outcomes("child_cross", 0.85)  # Successful crossover
    tracker.update_fitness_outcomes("child_mut", 0.55)    # Unsuccessful mutation
    
    # Get generation summary
    print("\n📊 Testing generation summary...")
    summary = tracker.get_generation_summary(1)
    print(f"   • Crossovers: {summary['crossovers']}")
    print(f"   • Mutations: {summary['mutations']}")
    print(f"   • Crossover success rate: {summary['crossover_success_rate']:.1f}%")
    print(f"   • Mutation success rate: {summary['mutation_success_rate']:.1f}%")
    
    # Test pattern detection
    print("\n🎯 Testing pattern detection...")
    patterns = tracker.detect_genetic_patterns(1)
    print(f"   • Patterns detected: {len(patterns)}")
    for pattern in patterns:
        print(f"     - {pattern.pattern_name}: {pattern.confidence:.1%}")
    
    # Save tracking data
    print("\n💾 Testing data saving...")
    tracker.save_tracking_data(1)
    
    print("\n✅ All genetic tracking tests passed!")
    
    # Show what was tracked
    print(f"\n📋 Summary:")
    print(f"   • Crossover records: {len(tracker.crossover_records)}")
    print(f"   • Mutation records: {len(tracker.mutation_records)}")
    print(f"   • Patterns detected: {len(tracker.detected_patterns)}")
    print(f"   • Output directory: {tracker.output_dir}")


def test_complex_scenario():
    """Test a more complex multi-generation scenario."""
    
    print("\n\n🔬 Testing Complex Multi-Generation Scenario")
    print("=" * 50)
    
    tracker = GeneticOperationsTracker(output_dir="test_output/genetic_complex")
    
    # Simulate 3 generations of evolution
    for gen in range(3):
        print(f"\n📊 Generation {gen + 1}")
        
        # Create parent population
        parents = [create_test_genome(f"gen{gen}_parent{i}", 
                                    fitness=0.5 + i * 0.1) for i in range(4)]
        
        # Simulate crossovers
        for i in range(3):
            child = create_test_genome(f"gen{gen}_cross{i}")
            tracker.track_crossover(
                child=child,
                parent1=parents[i],
                parent2=parents[i+1],
                generation=gen,
                diversity_strength=1.0 + gen * 0.2
            )
            
            # Child fitness varies (some successful, some not)
            child_fitness = 0.4 + i * 0.2 + gen * 0.1
            tracker.update_fitness_outcomes(f"gen{gen}_cross{i}", child_fitness)
        
        # Simulate mutations
        for i in range(2):
            child = create_test_genome(f"gen{gen}_mut{i}")
            mutation_type = "ca_mutation" if i == 0 else "lora_mutation"
            tracker.track_mutation(
                child=child,
                parent=parents[i],
                generation=gen,
                mutation_type=mutation_type,
                diversity_strength=1.0
            )
            
            # Mutation fitness varies
            parent_fitness = parents[i].fitness if hasattr(parents[i], 'fitness') else 0.5
            child_fitness = parent_fitness + (-0.1 + i * 0.3)  # Some improve, some don't
            tracker.update_fitness_outcomes(f"gen{gen}_mut{i}", child_fitness)
        
        # Show generation summary
        summary = tracker.get_generation_summary(gen)
        print(f"   🔀 Crossovers: {summary['crossovers']} (success: {summary['crossover_success_rate']:.1f}%)")
        print(f"   🧬 Mutations: {summary['mutations']} (success: {summary['mutation_success_rate']:.1f}%)")
        
        # Detect patterns
        patterns = tracker.detect_genetic_patterns(gen)
        if patterns:
            print(f"   🎯 Patterns: {len(patterns)} detected")
            for pattern in patterns:
                print(f"      - {pattern.pattern_name} ({pattern.confidence:.1%})")
    
    # Final save and summary
    tracker.save_tracking_data(2)
    
    print(f"\n📈 Final Summary:")
    print(f"   • Total crossovers: {len(tracker.crossover_records)}")
    print(f"   • Total mutations: {len(tracker.mutation_records)}")
    print(f"   • Total patterns: {len(tracker.detected_patterns)}")
    
    # Show successful operations
    successful_crosses = [r for r in tracker.crossover_records if r.outperformed_parents]
    successful_muts = [r for r in tracker.mutation_records 
                      if r.improvement_over_parent and r.improvement_over_parent > 0]
    
    print(f"   • Successful crossovers: {len(successful_crosses)}")
    print(f"   • Successful mutations: {len(successful_muts)}")
    
    print("\n✅ Complex scenario test completed!")


if __name__ == "__main__":
    try:
        test_genetic_tracking()
        test_complex_scenario()
        
        print("\n🎉 All genetic tracking tests successful!")
        print("   📂 Check test_output/genetic_tracking/ for saved data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 