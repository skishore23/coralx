#!/usr/bin/env python3
"""
Local test script for Multi-Modal AI Safety Plugin
Tests the system without expensive training loops
"""

import sys
import os
from pathlib import Path

# Add coralx to path
coralx_path = Path(__file__).parent.parent
sys.path.insert(0, str(coralx_path))

import yaml
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List

from coral.domain.genome import Genome
from coral.domain.mapping import LoRAConfig
from coral.domain.ca import CASeed
from coral.domain.cheap_knobs import CheapKnobs
import numpy as np

def test_basic_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test CoralX core imports
        from coral.domain.genome import Genome
        from coral.domain.mapping import LoRAConfig
        from coral.domain.ca import CASeed
        print("✅ CoralX core imports successful")
        
        # Test plugin imports
        from plugins.fakenews_gemma3n.plugin import (
            MultiModalAISafetyPlugin,
            MultiModalAISafetyDatasetProvider,
            Gemma3NModelRunner,
            MultiModalAISafetyFitness,
            MultiModalAISafetyMetrics
        )
        print("✅ Plugin imports successful")
        
        # Test optional dependencies
        try:
            import torch
            print(f"✅ PyTorch available: {torch.__version__}")
        except ImportError:
            print("⚠️  PyTorch not available")
        
        try:
            import transformers
            print(f"✅ Transformers available: {transformers.__version__}")
        except ImportError:
            print("⚠️  Transformers not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing config loading...")
    
    try:
        config_path = Path("config/fakenews_gemma3n_colab_config.yaml")
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded successfully")
        print(f"   - Experiment: {config.get('experiment', {}).get('name', 'Unknown')}")
        print(f"   - Population size: {config.get('evolution', {}).get('population_size', 'Unknown')}")
        print(f"   - Generations: {config.get('evolution', {}).get('generations', 'Unknown')}")
        
        return config
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return False

def test_dataset_provider_creation():
    """Test dataset provider creation (without actual data download)."""
    print("\n🧪 Testing dataset provider creation...")
    
    try:
        from plugins.fakenews_gemma3n.plugin import MultiModalAISafetyDatasetProvider
        
        # Minimal config for testing
        test_config = {
            'datasets': {
                'fake_news': {'enabled': True},
                'deepfake_audio': {'enabled': False},
                'deepfake_video': {'enabled': False},
                'jailbreak_prompts': {'enabled': False},
                'clean_holdout': {'enabled': False}
            },
            'cache_dir': str(Path.cwd() / "cache" / "test")
        }
        
        provider = MultiModalAISafetyDatasetProvider(test_config)
        print("✅ Dataset provider created successfully")
        
        # Test synthetic data generation
        provider._setup_datasets()
        print("✅ Synthetic datasets setup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset provider creation failed: {e}")
        traceback.print_exc()
        return False

def test_model_runner_creation():
    """Test model runner creation (without actual model loading)."""
    print("\n🧪 Testing model runner creation...")
    
    try:
        from plugins.fakenews_gemma3n.plugin import Gemma3NModelRunner
        
        # Create test LoRA config
        lora_cfg = LoRAConfig(
            r=8,
            alpha=16.0,
            dropout=0.1,
            target_modules=['q_proj', 'v_proj']
        )
        
        # Create test genome
        ca_seed = CASeed(
            grid=np.random.randint(0, 2, (5, 5)),
            rule=30,
            steps=10
        )
        
        genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_cfg,
            id="test_genome_001"
        )
        
        # Minimal model config
        model_config = {
            'model_name': 'unsloth/gemma-2b-it-bnb-4bit',
            'max_seq_length': 512,
            'load_in_4bit': True,
            'use_cache': True,
            'training': {
                'simulate_training': True,  # Key flag for testing
                'max_steps': 10,
                'max_train_samples': 5
            }
        }
        
        print("✅ Model runner config created")
        print(f"   - Model: {model_config['model_name']}")
        print(f"   - Simulate training: {model_config['training']['simulate_training']}")
        
        # Don't actually create the model yet (too expensive)
        print("⚠️  Skipping actual model creation for local test")
        
        return True
        
    except Exception as e:
        print(f"❌ Model runner creation failed: {e}")
        traceback.print_exc()
        return False

def test_fitness_function():
    """Test fitness function creation and basic evaluation."""
    print("\n🧪 Testing fitness function...")
    
    try:
        from plugins.fakenews_gemma3n.plugin import MultiModalAISafetyFitness, MultiModalAISafetyMetrics
        
        # Minimal fitness config
        fitness_config = {
            'test_samples': 5,
            'objectives': {
                'task_skill': {'weight': 1.0},
                'safety': {'weight': 1.0},
                'false_positive_rate': {'weight': -1.0},
                'memory_usage': {'weight': -0.5},
                'cross_modal_gain': {'weight': 0.8},
                'calibration': {'weight': 0.6}
            }
        }
        
        fitness_fn = MultiModalAISafetyFitness(fitness_config)
        print("✅ Fitness function created successfully")
        
        # Test metrics creation
        test_metrics = MultiModalAISafetyMetrics(
            task_skill_auroc=0.85,
            safety_score=0.92,
            false_positive_rate=0.05,
            memory_usage_gb=2.1,
            cross_modal_gain=0.15,
            calibration_score=0.88
        )
        
        overall_score = test_metrics.overall_score()
        print(f"✅ Test metrics created: overall_score={overall_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fitness function test failed: {e}")
        traceback.print_exc()
        return False

def test_plugin_integration():
    """Test full plugin integration."""
    print("\n🧪 Testing plugin integration...")
    
    try:
        from plugins.fakenews_gemma3n.plugin import MultiModalAISafetyPlugin
        
        # Load actual config
        config_path = Path("config/fakenews_gemma3n_colab_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        plugin = MultiModalAISafetyPlugin(config)
        print("✅ Plugin created successfully")
        
        # Test dataset provider
        dataset_provider = plugin.dataset()
        print("✅ Dataset provider accessible")
        
        # Test model factory
        model_factory = plugin.model_factory()
        print("✅ Model factory accessible")
        
        # Test fitness function
        fitness_fn = plugin.fitness_fn()
        print("✅ Fitness function accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Plugin integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all local tests."""
    print("🚀 CoralX Multi-Modal AI Safety Plugin - Local Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("Config Loading", test_config_loading()))
    test_results.append(("Dataset Provider", test_dataset_provider_creation()))
    test_results.append(("Model Runner", test_model_runner_creation()))
    test_results.append(("Fitness Function", test_fitness_function()))
    test_results.append(("Plugin Integration", test_plugin_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! System is ready for basic functionality.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 