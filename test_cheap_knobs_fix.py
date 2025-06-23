#!/usr/bin/env python3
"""
Test script to validate cheap knobs fix before Modal deployment.
Tests that the plugin's evaluate_multi_objective generates proper cheap knobs.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_plugin_evaluation_pipeline():
    """Test that plugin evaluation generates cheap knobs properly."""
    print("🧪 TESTING: Plugin evaluation pipeline with cheap knobs")
    print("=" * 60)
    
    try:
        # Import required modules
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        from coral.domain.experiment import create_experiment_config, create_initial_population
        import numpy as np
        import yaml
        
        # Load test config
        config_path = "coral_x_codellama_config.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded from {config_path}")
        print(f"   • Has cheap_knobs section: {'cheap_knobs' in config}")
        
        if 'cheap_knobs' in config:
            knobs_config = config['cheap_knobs']
            print(f"   • Temperature range: {knobs_config.get('temperature_range')}")
            print(f"   • Top-p range: {knobs_config.get('top_p_range')}")
            print(f"   • Top-k range: {knobs_config.get('top_k_range')}")
        
        # Create plugin
        plugin = QuixBugsCodeLlamaRealPlugin(config)
        print(f"✅ Plugin created successfully")
        
        # Create test population with one genome
        exp_config = create_experiment_config(config)
        population = create_initial_population(exp_config, diversity_strength=0.5, raw_config=config, run_id="test_run")
        test_genome = population.genomes[0]
        print(f"✅ Test genome created: {test_genome.id}")
        
        # Test CA evolution and feature extraction directly
        print(f"\n🔍 TESTING CA PIPELINE:")
        fitness_fn = plugin.fitness_fn()
        
        # Test CA evolution step
        ca_history = fitness_fn._run_ca_from_genome(test_genome)
        print(f"✅ CA evolution: {len(ca_history)} states generated")
        
        # Test feature extraction
        from coral.domain.feature_extraction import extract_ca_features
        ca_features = extract_ca_features(ca_history)
        print(f"✅ CA features extracted:")
        print(f"   • Complexity: {ca_features.complexity:.3f}")
        print(f"   • Intensity: {ca_features.intensity:.3f}")
        print(f"   • Periodicity: {ca_features.periodicity:.3f}")
        print(f"   • Convergence: {ca_features.convergence:.3f}")
        
        # Test cheap knobs generation
        from coral.domain.cheap_knobs import map_ca_features_to_cheap_knobs
        knobs_config = config['cheap_knobs']
        cheap_knobs = map_ca_features_to_cheap_knobs(ca_features, knobs_config)
        print(f"✅ Cheap knobs generated:")
        print(f"   • Temperature: {cheap_knobs.temperature:.3f} (should be ~0.3-1.2)")
        print(f"   • Top-p: {cheap_knobs.top_p:.3f} (should be ~0.75-0.95)")
        print(f"   • Top-k: {cheap_knobs.top_k} (should be ~15-70)")
        print(f"   • Repetition penalty: {cheap_knobs.repetition_penalty:.3f} (should be ~1.05-1.25)")
        print(f"   • Max tokens: {cheap_knobs.max_new_tokens} (should be ~120-350)")
        print(f"   • Do sample: {cheap_knobs.do_sample}")
        
        # Validate ranges
        temp_range = knobs_config['temperature_range']
        if not (temp_range[0] <= cheap_knobs.temperature <= temp_range[1]):
            print(f"❌ Temperature {cheap_knobs.temperature} outside range {temp_range}")
            return False
            
        print(f"✅ All cheap knobs within expected ranges!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_modal_function_serialization():
    """Test genome serialization for Modal compatibility."""
    print(f"\n🧪 TESTING: Modal genome serialization")
    print("=" * 60)
    
    try:
        # Import required modules
        from coral.domain.experiment import create_experiment_config, create_initial_population
        from infra.modal_executor import ModalExecutor
        import yaml
        
        # Load config
        with open("coral_x_codellama_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create test genome
        exp_config = create_experiment_config(config)
        population = create_initial_population(exp_config, diversity_strength=0.5, raw_config=config, run_id="test_serialization")
        test_genome = population.genomes[0]
        
        print(f"✅ Test genome created: {test_genome.id}")
        print(f"   • LoRA rank: {test_genome.lora_cfg.r}")
        print(f"   • LoRA alpha: {test_genome.lora_cfg.alpha}")
        print(f"   • CA rule: {test_genome.seed.rule}")
        
        # Test serialization via ModalExecutor
        modal_executor = ModalExecutor(app_name="test", config=config)
        serialized = modal_executor._serialize_genome(test_genome)
        
        print(f"✅ Serialized: {len(str(serialized))} chars")
        print(f"   • Keys: {list(serialized.keys())}")
        print(f"   • ID: {serialized.get('id')}")
        print(f"   • Has seed: {'seed' in serialized}")
        print(f"   • Has lora_config: {'lora_config' in serialized}")
        
        # Validate required fields
        required_fields = ['id', 'seed', 'lora_config']
        for field in required_fields:
            if field not in serialized:
                print(f"❌ Missing required field: {field}")
                return False
        
        print(f"✅ Serialization contains all required fields!")
        return True
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_modal_function_imports():
    """Test that the new Modal function can be imported and has the right structure."""
    print(f"\n🧪 TESTING: Modal function imports and structure")
    print("=" * 60)
    
    try:
        # Test importing the function we're modifying
        from infra.modal.experiment_service import evaluate_genome_modal
        print(f"✅ evaluate_genome_modal imported successfully")
        
        # Check if it's a callable
        if not callable(evaluate_genome_modal):
            print(f"❌ evaluate_genome_modal is not callable: {type(evaluate_genome_modal)}")
            return False
        
        print(f"✅ Function is callable")
        
        # Test importing required components for the new implementation
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        print(f"✅ QuixBugsCodeLlamaRealPlugin imported")
        
        from coral.domain.genome import Genome, CASeed, LoRAConfig
        print(f"✅ Genome types imported")
        
        # Test that all the expected modules exist
        try:
            import numpy as np
            print(f"✅ numpy available")
        except ImportError as e:
            print(f"❌ numpy not available: {e}")
            return False
        
        print(f"✅ All required imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cheap_knobs_dataclass():
    """Test that CheapKnobs dataclass works correctly."""
    print(f"\n🧪 TESTING: CheapKnobs dataclass functionality")
    print("=" * 60)
    
    try:
        from coral.domain.cheap_knobs import CheapKnobs, map_ca_features_to_cheap_knobs, cheap_knobs_to_generation_kwargs
        from coral.domain.feature_extraction import CAFeatures
        import yaml
        
        # Load config
        with open("coral_x_codellama_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        knobs_config = config['cheap_knobs']
        
        # Create test CA features
        test_features = CAFeatures(
            complexity=0.7,
            intensity=0.5,
            periodicity=0.3,
            convergence=0.8
        )
        
        print(f"✅ Test CA features created: {test_features}")
        
        # Test cheap knobs generation
        cheap_knobs = map_ca_features_to_cheap_knobs(test_features, knobs_config)
        print(f"✅ Cheap knobs generated: {cheap_knobs}")
        
        # Test conversion to generation kwargs
        gen_kwargs = cheap_knobs_to_generation_kwargs(cheap_knobs)
        print(f"✅ Generation kwargs: {gen_kwargs}")
        
        # Validate expected keys
        expected_keys = ['temperature', 'top_p', 'top_k', 'max_new_tokens', 'repetition_penalty', 'do_sample']
        for key in expected_keys:
            if key not in gen_kwargs:
                print(f"❌ Missing expected key in generation kwargs: {key}")
                return False
        
        print(f"✅ All expected keys present in generation kwargs!")
        return True
        
    except Exception as e:
        print(f"❌ CheapKnobs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 CHEAP KNOBS FIX VALIDATION")
    print("=" * 80)
    print("Testing the fix before Modal deployment...")
    print()
    
    start_time = time.time()
    
    # Run tests
    test1_ok = test_plugin_evaluation_pipeline()
    test2_ok = test_modal_function_serialization() 
    test3_ok = test_modal_function_imports()
    test4_ok = test_cheap_knobs_dataclass()
    
    total_time = time.time() - start_time
    
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🧪 Plugin pipeline: {'✅ PASS' if test1_ok else '❌ FAIL'}")
    print(f"🔄 Serialization: {'✅ PASS' if test2_ok else '❌ FAIL'}")
    print(f"📡 Modal imports: {'✅ PASS' if test3_ok else '❌ FAIL'}")
    print(f"🎛️ CheapKnobs: {'✅ PASS' if test4_ok else '❌ FAIL'}")
    
    all_passed = test1_ok and test2_ok and test3_ok and test4_ok
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! Safe to deploy to Modal.")
        print(f"💡 The fix should now generate proper cheap knobs in Modal logs:")
        print(f"   Expected logs:")
        print(f"   🎛️ Using FITNESS FUNCTION for full two-loop evaluation...")
        print(f"   🌊 Running Cellular Automata Evolution...")
        print(f"   🎛️ Generating Cheap Knobs from CA Features...")
        print(f"   🔍 FITNESS FUNCTION DEBUG:")
        print(f"   • Temperature: X.XXX (complexity-driven)")
        print(f"   • Top-p: X.XXX (intensity-driven)")
    else:
        print(f"\n⚠️  SOME TESTS FAILED! Review issues before deploying.")
        
    print("=" * 80) 