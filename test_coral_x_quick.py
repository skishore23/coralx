#!/usr/bin/env python3
"""
CORAL-X Quick Test Runner
Fast end-to-end validation with reduced parameters.
"""

import time
import sys
from pathlib import Path
from coral.config.loader import load_config
from coral.application.experiment_orchestrator import ExperimentOrchestrator
from infra.modal_executor import create_executor_from_config

def validate_architecture_compliance(config):
    """Quick architecture validation for test config."""
    print("🔍 Validating CORAL-X architecture compliance...")
    
    required_sections = ['evolution', 'ca', 'lora', 'thresholds', 'experiment', 'modal']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")
    
    # Validate reduced parameters are reasonable
    pop_size = config['evolution']['population_size']
    max_gen = config['evolution']['max_generations']
    
    if pop_size < 4:
        raise ValueError(f"Population size {pop_size} too small (minimum 4)")
    if max_gen < 1:
        raise ValueError(f"Max generations {max_gen} too small (minimum 1)")
    
    print("✅ Architecture validation passed")
    return True

def run_quick_test():
    """Run a quick CORAL-X evolution test."""
    print("🚀 CORAL-X Quick Test Runner")
    print("============================================================")
    
    config_file = "coral_x_test_config.yaml"
    
    if not Path(config_file).exists():
        print(f"❌ Test config file not found: {config_file}")
        print("Please ensure coral_x_test_config.yaml exists")
        return False
    
    try:
        # Load test configuration
        print(f"📋 Loading test config: {config_file}")
        config = load_config(config_file)
        
        print(f"📋 Test Configuration:")
        print(f"   • Population: {config['evolution']['population_size']} genomes")
        print(f"   • Generations: {config['evolution']['max_generations']}")
        print(f"   • Model: {config['experiment']['model']['name']}")
        print(f"   • Dataset: {config['experiment']['dataset']}")
        print(f"   • Executor: modal")
        print("============================================================")
        
        # Validate architecture
        validate_architecture_compliance(config)
        
        # Create executor
        print("🔧 Creating Modal executor...")
        executor = create_executor_from_config(config)
        
        # Create experiment orchestrator
        print("🎼 Creating experiment orchestrator...")
        orchestrator = ExperimentOrchestrator(config=config, executor=executor)
        
        # Run the experiment
        print("🚀 Starting quick test evolution...")
        start_time = time.time()
        
        result = orchestrator.run_experiment()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Report results
        print("📊 Test results:")
        
        if result.success:
            print(f"✅ Quick test completed successfully!")
        else:
            print(f"❌ Quick test failed: {result.error}")
        
        print("============================================================")
        print(f"🏆 CORAL-X Quick Test Complete!")
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🎯 Success: {result.success}")
        
        # Show key metrics
        if result.final_population:
            best_fitness = max(
                genome.fitness.overall_score if genome.fitness else 0.0
                for genome in result.final_population
            )
            print(f"🏆 Best Overall Score: {best_fitness:.3f}")
        
        if result.generation_count:
            print(f"📈 Generations Completed: {result.generation_count}")
        
        # Cache efficiency
        if hasattr(result, 'cache_stats'):
            print(f"⚡ Cache Efficiency: {result.cache_stats.get('efficiency', 1.0):.1f}x")
        
        print("✅ Quick test report generated successfully")
        return result.success
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def estimate_full_run_time():
    """Estimate full production run time based on test results."""
    print("\n📊 PRODUCTION RUN ESTIMATE")
    print("=" * 60)
    
    # Assumptions based on test config vs production config
    test_population = 8
    test_generations = 5
    prod_population = 32
    prod_generations = 40
    
    # Time multipliers
    population_multiplier = prod_population / test_population  # 4x
    generation_multiplier = prod_generations / test_generations  # 8x
    
    # Rough estimates (these would be updated after test run)
    estimated_test_time_minutes = 15  # Estimate for test run
    estimated_prod_time_minutes = estimated_test_time_minutes * population_multiplier * generation_multiplier
    
    print(f"Test Configuration:")
    print(f"   Population: {test_population}, Generations: {test_generations}")
    print(f"   Estimated time: ~{estimated_test_time_minutes} minutes")
    
    print(f"\nProduction Configuration:")
    print(f"   Population: {prod_population}, Generations: {prod_generations}")
    print(f"   Time multiplier: {population_multiplier:.1f}x population × {generation_multiplier:.1f}x generations = {population_multiplier * generation_multiplier:.1f}x")
    print(f"   Estimated time: ~{estimated_prod_time_minutes:.0f} minutes ({estimated_prod_time_minutes/60:.1f} hours)")
    
    print(f"\nCache Benefits:")
    print(f"   • Generation 1: Minimal cache benefit (most adapters new)")
    print(f"   • Generation 5+: 2-3x speedup from adapter reuse")
    print(f"   • Final estimate: {estimated_prod_time_minutes * 0.4:.0f}-{estimated_prod_time_minutes * 0.7:.0f} minutes with cache")

if __name__ == "__main__":
    try:
        # Run quick test
        success = run_quick_test()
        
        if success:
            # Show production estimates
            estimate_full_run_time()
            print(f"\n🎯 Ready for production run with coral_x_codellama_config.yaml")
        else:
            print(f"\n❌ Fix issues before running production configuration")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1) 