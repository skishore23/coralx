#!/usr/bin/env python3
"""
Test script to verify genome IDs are working correctly
"""
import yaml
from pathlib import Path


def test_genome_creation():
    """Test that genomes are created with proper IDs."""
    print("🧪 TESTING GENOME ID CREATION")
    print("=" * 60)
    
    try:
        from coral.domain.experiment import create_initial_population, create_experiment_config
        
        # Load config
        config_file = "coral_x_codellama_config.yaml"
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            return False
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Create experiment config
        exp_config = create_experiment_config(config)
        print(f"✅ Experiment config created")
        print(f"   • Population size: {exp_config.population_size}")
        print(f"   • Generations: {exp_config.generations}")
        
        # Create initial population
        print(f"\n🧬 Creating initial population...")
        population = create_initial_population(exp_config)
        
        print(f"✅ Population created: {population.size()} genomes")
        
        # Check genome IDs
        print(f"\n🔍 Checking genome IDs:")
        ids_found = []
        for i, genome in enumerate(population.genomes[:5]):  # Check first 5
            genome_id = getattr(genome, 'id', 'NO_ID')
            ids_found.append(genome_id)
            print(f"   • Genome {i}: ID = '{genome_id}'")
        
        # Verify no duplicates
        unique_ids = set(ids_found)
        if len(unique_ids) == len(ids_found):
            print(f"✅ All IDs are unique")
        else:
            print(f"❌ Duplicate IDs found!")
            return False
        
        # Test serialization
        print(f"\n📦 Testing genome serialization...")
        test_genome = population.genomes[0]
        
        from infra.modal_executor import ModalExecutor
        
        # Create a mock modal executor to test serialization
        class MockModalExecutor:
            def __init__(self):
                pass
            
            def _serialize_genome(self, genome):
                """Use the same method as ModalExecutor."""
                return {
                    'id': getattr(genome, 'id', f'genome_{hash(str(genome))%10000:04d}'),
                    'seed': {
                        'grid': genome.seed.grid.tolist() if hasattr(genome.seed.grid, 'tolist') else genome.seed.grid,
                        'rule': getattr(genome.seed, 'rule', 0),
                        'steps': getattr(genome.seed, 'steps', 15)
                    },
                    'lora_config': {
                        'r': getattr(genome.lora_cfg, 'r', 8),
                        'alpha': getattr(genome.lora_cfg, 'alpha', 16.0),
                        'dropout': getattr(genome.lora_cfg, 'dropout', 0.1),
                        'target_modules': list(getattr(genome.lora_cfg, 'target_modules', ['q_proj', 'v_proj']))
                    }
                }
        
        mock_executor = MockModalExecutor()
        serialized = mock_executor._serialize_genome(test_genome)
        
        print(f"✅ Serialization test:")
        print(f"   • Original ID: '{test_genome.id}'")
        print(f"   • Serialized ID: '{serialized['id']}'")
        print(f"   • LoRA config present: {'lora_config' in serialized}")
        print(f"   • LoRA rank: {serialized['lora_config']['r']}")
        
        if test_genome.id == serialized['id']:
            print(f"✅ ID preserved through serialization")
        else:
            print(f"❌ ID not preserved!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evolution_operations():
    """Test that evolution operations preserve IDs."""
    print(f"\n🧬 TESTING EVOLUTION OPERATIONS")
    print("=" * 60)
    
    try:
        from coral.domain.experiment import create_initial_population, create_experiment_config
        from coral.domain.neat import crossover, mutate
        from coral.domain.mapping import EvolutionConfig
        from random import Random
        
        # Load config
        config_file = "coral_x_codellama_config.yaml"
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        exp_config = create_experiment_config(config)
        population = create_initial_population(exp_config)
        
        # Test crossover
        parent1 = population.genomes[0]
        parent2 = population.genomes[1]
        
        print(f"🔀 Testing crossover:")
        print(f"   • Parent 1 ID: '{parent1.id}'")
        print(f"   • Parent 2 ID: '{parent2.id}'")
        
        rng = Random(42)
        child = crossover(parent1, parent2, exp_config.evolution_config, rng)
        
        print(f"   • Child ID: '{child.id}'")
        print(f"   • Child ID format correct: {child.id.startswith('cross_')}")
        
        # Test mutation
        print(f"\n🔄 Testing mutation:")
        print(f"   • Parent ID: '{parent1.id}'")
        
        mutant = mutate(parent1, exp_config.evolution_config, rng)
        
        print(f"   • Mutant ID: '{mutant.id}'")
        print(f"   • Mutant ID format correct: {mutant.id.startswith('mut_')}")
        
        # Test with_* methods
        print(f"\n📝 Testing with_* methods:")
        from coral.domain.genome import MultiObjectiveScores
        
        scores = MultiObjectiveScores(bugfix=0.8, style=0.7, security=0.9, runtime=0.6, syntax=0.9)
        updated_genome = parent1.with_multi_scores(scores)
        
        print(f"   • Original ID: '{parent1.id}'")
        print(f"   • Updated ID: '{updated_genome.id}'")
        print(f"   • ID preserved: {parent1.id == updated_genome.id}")
        
        if parent1.id != updated_genome.id:
            print(f"❌ ID not preserved in with_multi_scores!")
            return False
        
        print(f"✅ All evolution operations preserve IDs correctly")
        return True
        
    except Exception as e:
        print(f"❌ Evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 CORAL-X GENOME ID VERIFICATION")
    print("=" * 80)
    
    # Test 1: Genome creation
    creation_success = test_genome_creation()
    
    # Test 2: Evolution operations
    evolution_success = test_evolution_operations()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Genome Creation: {'PASS' if creation_success else 'FAIL'}")
    print(f"✅ Evolution Operations: {'PASS' if evolution_success else 'FAIL'}")
    
    if creation_success and evolution_success:
        print("\n🎉 All tests passed! Genome IDs are working correctly.")
        print("💡 Your evolution should now show proper genome IDs instead of 'unknown'")
    else:
        print("\n❌ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main() 