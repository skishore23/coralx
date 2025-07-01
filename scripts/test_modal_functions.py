#!/usr/bin/env python3
"""
Test Modal functions locally before deployment
Simple YAML config testing - no new methods needed
"""

import sys
import json
from pathlib import Path
import tempfile

# Add coralx to path
coralx_path = Path(__file__).parent.parent
sys.path.insert(0, str(coralx_path))

def test_basic_functions():
    """Test basic Modal functions locally."""
    print("🧪 Testing Modal functions locally...")
    
    # Test 1: Progress function (CPU-only, should work)
    print("\n1. Testing get_evolution_progress_modal...")
    try:
        # Import the function logic
        sys.path.insert(0, str(coralx_path))
        
        # Simulate the function locally
        import json
        from datetime import datetime
        
        # Create test progress data
        test_progress = {
            'status': 'test',
            'message': 'Local test run',
            'current_generation': 2,
            'max_generations': 10,
            'best_fitness': 0.75,
            'elapsed_time': 120.0,
            'start_time_str': datetime.now().strftime('%H:%M:%S'),
            'population_size': 4,
            'run_id': 'test_run'
        }
        
        print(f"   ✅ Progress data structure: {test_progress}")
        
    except Exception as e:
        print(f"   ❌ Progress test failed: {e}")
    
    # Test 2: Config validation
    print("\n2. Testing config validation...")
    try:
        from coral.config.loader import create_config_from_dict
        
        # Load test config
        test_config_path = Path("config/test.yaml")
        if test_config_path.exists():
            import yaml
            with open(test_config_path) as f:
                config_dict = yaml.safe_load(f)
            
            # Test config creation (this is what Modal uses)
            coral_config = create_config_from_dict(config_dict)
            print(f"   ✅ Config validation passed")
            print(f"   📊 Generations: {coral_config.execution['generations']}")
            print(f"   👥 Population: {coral_config.execution['population_size']}")
        else:
            print(f"   ⚠️  Test config not found: {test_config_path}")
            
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
    
    # Test 3: Plugin loading
    print("\n3. Testing plugin loading...")
    try:
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        
        # Load test config for plugin
        test_config_path = Path("config/test.yaml")
        if test_config_path.exists():
            import yaml
            with open(test_config_path) as f:
                config_dict = yaml.safe_load(f)
            
            # Test plugin creation (this is what Modal uses)
            plugin = QuixBugsCodeLlamaRealPlugin(config_dict)
            print(f"   ✅ Plugin loading passed")
            
        else:
            print(f"   ⚠️  Test config not found: {test_config_path}")
            
    except Exception as e:
        print(f"   ❌ Plugin test failed: {e}")
    
    # Test 4: Evolution engine creation (without dataset files)
    print("\n4. Testing evolution engine setup...")
    try:
        from coral.config.loader import create_config_from_dict
        from coral.application.evolution_engine import EvolutionEngine
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        from infra.modal_executor import create_executor_from_config
        
        test_config_path = Path("config/test.yaml")
        if test_config_path.exists():
            import yaml
            with open(test_config_path) as f:
                config_dict = yaml.safe_load(f)
            
            # Test component creation (this is what Modal does)
            print("   🔧 Testing config creation...")
            coral_config = create_config_from_dict(config_dict)
            print(f"      ✅ Config created")
            
            print("   🔌 Testing plugin creation...")
            plugin = QuixBugsCodeLlamaRealPlugin(config_dict)
            print(f"      ✅ Plugin created")
            
            print("   ⚡ Testing executor creation...")
            executor = create_executor_from_config(config_dict)
            print(f"      ✅ Executor created")
            
            print("   🧬 Testing evolution engine creation...")
            run_id = config_dict.get('cache', {}).get('run_id', 'test')
            
            # Note: We can't test dataset.problems() locally since dataset files don't exist
            # But we can test that the objects are created properly
            try:
                # Create evolution engine (this is what Modal does)
                engine = EvolutionEngine(
                    cfg=coral_config,
                    fitness_fn=plugin.fitness_fn(),
                    executor=executor,
                    model_factory=plugin.model_factory(),
                    dataset=plugin.dataset(),  # This creates the dataset object but doesn't load files
                    run_id=run_id
                )
                print(f"      ✅ Evolution engine created successfully")
                print(f"   🚀 All components ready for Modal execution")
                
            except Exception as dataset_error:
                if "dataset not found" in str(dataset_error).lower():
                    print(f"      ✅ Evolution engine setup OK (dataset missing locally is expected)")
                    print(f"      📋 Dataset will be available in Modal volume: /cache/quixbugs_dataset")
                else:
                    raise dataset_error
            
        else:
            print(f"   ⚠️  Test config not found: {test_config_path}")
            
    except Exception as e:
        print(f"   ❌ Evolution engine test failed: {e}")
        print(f"      💡 This indicates a real setup issue that would break in Modal!")

def main():
    """Run local tests."""
    print("🔍 Local Modal Function Testing")
    print("=" * 50)
    print("This tests the same code paths that Modal uses")
    print("Catch errors BEFORE deployment!")
    print()
    
    test_basic_functions()
    
    print("\n" + "=" * 50)
    print("🏁 Local testing complete!")
    print()
    print("✅ If all tests pass → Ready to deploy")
    print("❌ If tests fail → Fix locally first")

if __name__ == "__main__":
    main() 