#!/usr/bin/env python3
"""
Test Parameter Loading from Evolution Results
=============================================

Verify that the benchmark can load actual evolution parameters.
"""
import sys
sys.path.append('.')

def test_parameter_loading():
    """Test loading evolved and baseline parameters."""
    
    print("🔧 TESTING PARAMETER LOADING")
    print("=" * 50)
    
    try:
        from real_inference_benchmark import RealInferenceBenchmark
        
        # Initialize benchmark
        print("📁 Initializing benchmark...")
        benchmark = RealInferenceBenchmark("coral_x_clean_config.yaml")
        
        # Test evolved parameter loading
        print(f"\n🧬 EVOLVED PARAMETERS:")
        evolved_params = benchmark.evolved_params
        for key, value in evolved_params.items():
            print(f"   • {key}: {value}")
        
        # Test baseline parameter loading  
        print(f"\n📊 BASELINE PARAMETERS:")
        baseline_params = benchmark._get_baseline_parameters()
        for key, value in baseline_params.items():
            print(f"   • {key}: {value}")
        
        # Compare key differences
        print(f"\n🆚 KEY DIFFERENCES:")
        print(f"   • Rank: {evolved_params['r']} vs {baseline_params['r']} (evolved vs baseline)")
        print(f"   • Alpha: {evolved_params['lora_alpha']} vs {baseline_params['lora_alpha']}")
        print(f"   • Dropout: {evolved_params['lora_dropout']} vs {baseline_params['lora_dropout']}")
        print(f"   • Target modules: {len(evolved_params['target_modules'])} vs {len(baseline_params['target_modules'])}")
        
        # Test evolution results loading
        print(f"\n📊 EVOLUTION RESULTS:")
        evolution_results = benchmark._load_latest_evolution_results()
        if evolution_results:
            print(f"   • Best fitness: {evolution_results['best_fitness']:.3f}")
            print(f"   • Experiment time: {evolution_results['experiment_time']:.1f}s")
            print(f"   • Generations: {evolution_results['generations']}")
            print(f"   • Population size: {evolution_results['population_size']}")
            
            if 'best_scores' in evolution_results:
                scores = evolution_results['best_scores']
                print(f"   • Best scores:")
                for metric, score in scores.items():
                    print(f"     - {metric}: {score:.3f}")
        else:
            print(f"   • No evolution results found")
        
        print(f"\n✅ PARAMETER LOADING TEST COMPLETE!")
        return True
        
    except Exception as e:
        print(f"❌ Parameter loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_variations():
    """Test with different config files."""
    
    print(f"\n🔧 TESTING CONFIG VARIATIONS")
    print("=" * 50)
    
    configs_to_test = [
        "coral_x_clean_config.yaml",
        "coral_x_real_config.yaml", 
        "coral_x_test_config.yaml"
    ]
    
    for config_file in configs_to_test:
        try:
            print(f"\n📁 Testing config: {config_file}")
            from real_inference_benchmark import RealInferenceBenchmark
            
            benchmark = RealInferenceBenchmark(config_file)
            evolved = benchmark.evolved_params
            baseline = benchmark._get_baseline_parameters()
            
            print(f"   ✅ Config loaded successfully")
            print(f"   • Evolved r={evolved['r']}, α={evolved['lora_alpha']}")
            print(f"   • Baseline r={baseline['r']}, α={baseline['lora_alpha']}")
            
        except FileNotFoundError:
            print(f"   ⚠️ Config file not found: {config_file}")
        except Exception as e:
            print(f"   ❌ Config test failed: {e}")


if __name__ == "__main__":
    success = test_parameter_loading()
    
    if success:
        test_config_variations()
        
        print(f"\n🎉 ALL TESTS COMPLETE!")
        print(f"Ready to run: python real_inference_benchmark.py")
    else:
        print(f"\n❌ Tests failed - check configuration") 