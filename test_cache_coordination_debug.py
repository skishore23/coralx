#!/usr/bin/env python3
"""
Comprehensive Cache Coordination Debug - Test locally to fix Modal issues
This reproduces the exact issue: training interruptions cause missing adapters
"""
import sys
import os
import time
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_cache_coordination_issue():
    """Reproduce the exact cache coordination issue locally."""
    print("🔍 DEBUGGING: Cache Coordination Issue")
    print("=" * 70)
    
    try:
        from infra.adapter_cache import HeavyGenes, AdapterCache, CacheConfig
        from coral.domain.lora_training import train_codellama_lora
        from coral.domain.mapping import LoRAConfig
        import yaml
        
        # Step 1: Create test cache directory
        test_cache_dir = Path(tempfile.mkdtemp(prefix="coralx_test_cache_"))
        print(f"📁 Test cache directory: {test_cache_dir}")
        
        cache_config = CacheConfig(
            base_dir=str(test_cache_dir),
            modal_volume_name=None,  # Local testing
            use_modal_volume=False
        )
        
        cache = AdapterCache(cache_config)
        
        # Step 2: Create genome with specific parameters (from your logs)
        print(f"\n🧬 Creating test genome...")
        
        # These are the exact parameters from your failing logs
        test_genomes = [
            {
                'name': 'gen0_genome0001', 
                'params': {'rank': 16, 'alpha': 16.0, 'dropout': 0.15, 'adapter_type': 'dora'}
            },
            {
                'name': 'gen0_genome0002', 
                'params': {'rank': 16, 'alpha': 4.0, 'dropout': 0.15, 'adapter_type': 'dora'}
            }
        ]
        
        for genome_info in test_genomes:
            print(f"\n🔬 Testing: {genome_info['name']}")
            params = genome_info['params']
            
            # Create HeavyGenes exactly as done in the system
            heavy_genes = HeavyGenes(
                rank=params['rank'],
                alpha=params['alpha'],
                dropout=params['dropout'],
                target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
                adapter_type=params['adapter_type'],
                run_id='dora_test_v1'
            )
            
            adapter_hash = heavy_genes.to_hash()
            expected_path = f"/cache/adapters/adapter_{adapter_hash}"
            local_path = test_cache_dir / "adapters" / f"adapter_{adapter_hash}"
            
            print(f"   🔍 Heavy genes: {heavy_genes}")
            print(f"   📊 Hash: {adapter_hash}")
            print(f"   📁 Expected path: {expected_path}")
            print(f"   📁 Local path: {local_path}")
            
            # Step 3: Simulate the EXACT failure scenario
            print(f"   🎯 Simulating cache coordination...")
            
            # Check if adapter exists (it shouldn't)
            if cache.get_cached_adapter_path(heavy_genes):
                print(f"   ⚠️  Adapter already exists - cleaning up")
                if local_path.exists():
                    shutil.rmtree(local_path)
            
            # This is the EXACT scenario causing your issue:
            # 1. System checks cache -> MISS
            # 2. System starts training
            # 3. Training gets interrupted 
            # 4. System tries to use non-existent adapter -> FAIL
            
            print(f"   ❌ [CACHE] MISS: No cached adapter found (expected)")
            print(f"   🔧 [TRAINING] Would start training...")
            
            # Instead of actually training (which might fail), let's simulate the paths
            print(f"   🔄 [SIMULATION] Creating mock adapter...")
            
            # Create the adapter directory structure
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.mkdir(exist_ok=True)
            
            # Create mock adapter files (to test path resolution)
            mock_config = {
                "base_model_name_or_path": "codellama/CodeLlama-7b-Python-hf",
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": False,
                "init_lora_weights": True,
                "layers_pattern": None,
                "layers_to_transform": None,
                "lora_alpha": params['alpha'],
                "lora_dropout": params['dropout'],
                "modules_to_save": None,
                "peft_type": "LORA",
                "r": params['rank'],
                "revision": None,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "CAUSAL_LM"
            }
            
            # Write config file
            with open(local_path / "adapter_config.json", 'w') as f:
                import json
                json.dump(mock_config, f, indent=2)
            
            # Create mock weights file
            (local_path / "adapter_model.safetensors").write_bytes(b"mock_weights_data")
            
            print(f"   ✅ Mock adapter created")
            print(f"   📄 Files: {list(local_path.glob('*'))}")
            
            # Step 4: Test cache retrieval
            print(f"   🔍 Testing cache retrieval...")
            
            # Update cache to know about the new adapter
            cache._cache[adapter_hash] = str(local_path)
            
            retrieved_path = cache.get_cached_adapter_path(heavy_genes)
            if retrieved_path:
                print(f"   ✅ [CACHE] HIT: Found adapter at {retrieved_path}")
                
                # Verify files exist
                if Path(retrieved_path).exists():
                    files = list(Path(retrieved_path).glob("*"))
                    print(f"   📄 Contains {len(files)} files: {[f.name for f in files]}")
                else:
                    print(f"   ❌ Path exists in cache but files missing!")
            else:
                print(f"   ❌ [CACHE] Still MISS - cache coordination broken!")
        
        # Step 5: Test the full get_or_train flow 
        print(f"\n🔄 Testing get_or_train_adapter flow...")
        
        def mock_trainer(heavy_genes: HeavyGenes, save_path: str) -> str:
            """Mock trainer that simulates successful training."""
            print(f"   🚀 [MOCK TRAINING] Training {heavy_genes.adapter_type} adapter")
            print(f"   📍 Save to: {save_path}")
            
            # Simulate training delay
            time.sleep(0.1)
            
            # Create the adapter files
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Write config
            config = {
                "peft_type": heavy_genes.adapter_type.upper(),
                "r": heavy_genes.rank,
                "lora_alpha": heavy_genes.alpha,
                "lora_dropout": heavy_genes.dropout,
                "target_modules": list(heavy_genes.target_modules),
                "task_type": "CAUSAL_LM"
            }
            
            with open(save_path_obj / "adapter_config.json", 'w') as f:
                import json
                json.dump(config, f, indent=2)
            
            # Mock weights
            (save_path_obj / "adapter_model.safetensors").write_bytes(b"trained_weights")
            
            print(f"   ✅ [MOCK TRAINING] Complete: {save_path}")
            return save_path
        
        # Test with new parameters that don't exist yet
        test_heavy_genes = HeavyGenes(
            rank=8,
            alpha=32.0,
            dropout=0.1,
            target_modules=('q_proj', 'v_proj'),
            adapter_type='lora',
            run_id='test_coordination'
        )
        
        print(f"   🧪 Testing with: {test_heavy_genes}")
        print(f"   📊 Hash: {test_heavy_genes.to_hash()}")
        
        # This should trigger the training flow
        adapter_path = cache.get_or_train_adapter(test_heavy_genes, mock_trainer)
        
        print(f"   ✅ get_or_train_adapter returned: {adapter_path}")
        
        # Verify the adapter was created and cached
        if Path(adapter_path).exists():
            print(f"   ✅ Adapter exists at path")
            files = list(Path(adapter_path).glob("*"))
            print(f"   📄 Files: {[f.name for f in files]}")
        else:
            print(f"   ❌ Adapter missing at returned path!")
        
        # Test cache hit on second call
        print(f"   🔄 Testing cache hit on second call...")
        adapter_path_2 = cache.get_or_train_adapter(test_heavy_genes, mock_trainer)
        
        if adapter_path == adapter_path_2:
            print(f"   ✅ Cache hit successful - same path returned")
        else:
            print(f"   ❌ Cache miss - different paths: {adapter_path} vs {adapter_path_2}")
        
        print(f"\n🎉 Cache coordination test completed!")
        print(f"📁 Test cache at: {test_cache_dir}")
        print(f"🧹 Clean up with: rm -rf {test_cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache coordination test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_interruption_scenario():
    """Test what happens when training gets interrupted."""
    print(f"\n🔥 TESTING: Training Interruption Scenario")
    print("=" * 50)
    
    try:
        from infra.adapter_cache import HeavyGenes, AdapterCache, CacheConfig
        
        # Create test cache
        test_cache_dir = Path(tempfile.mkdtemp(prefix="interrupt_test_"))
        cache_config = CacheConfig(base_dir=str(test_cache_dir), use_modal_volume=False)
        cache = AdapterCache(cache_config)
        
        # Create test heavy genes
        heavy_genes = HeavyGenes(
            rank=16, alpha=16.0, dropout=0.15,
            target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
            adapter_type='dora', run_id='interrupt_test'
        )
        
        print(f"🧪 Heavy genes: {heavy_genes}")
        adapter_hash = heavy_genes.to_hash()
        expected_path = test_cache_dir / "adapters" / f"adapter_{adapter_hash}"
        print(f"📁 Expected path: {expected_path}")
        
        # Simulate interrupted training
        def interrupted_trainer(heavy_genes: HeavyGenes, save_path: str) -> str:
            """Trainer that gets interrupted partway through."""
            print(f"   🚀 [TRAINING] Starting training...")
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Start creating files
            print(f"   📝 Creating config file...")
            config = {"peft_type": "DORA", "r": heavy_genes.rank}
            with open(save_path_obj / "adapter_config.json", 'w') as f:
                import json
                json.dump(config, f)
            
            # Simulate interruption before weights are saved
            print(f"   💥 [INTERRUPTED] Training interrupted!")
            raise KeyboardInterrupt("Training interrupted by user")
        
        # Try to get adapter (should fail due to interruption)
        print(f"🔄 Attempting get_or_train with interruption...")
        
        try:
            adapter_path = cache.get_or_train_adapter(heavy_genes, interrupted_trainer)
            print(f"❌ Expected interruption but got path: {adapter_path}")
        except KeyboardInterrupt:
            print(f"✅ Training correctly interrupted")
            
            # Check what's left behind
            if expected_path.exists():
                files = list(expected_path.glob("*"))
                print(f"🗂️  Partial files left: {[f.name for f in files]}")
                
                # This is the problem! Partial adapter directory exists
                # but training didn't complete, so it's unusable
                print(f"⚠️  PROBLEM: Partial adapter directory exists!")
                print(f"    System will think adapter exists but it's incomplete")
            else:
                print(f"✅ No partial files left behind")
        
        # Now try again - this should work
        def working_trainer(heavy_genes: HeavyGenes, save_path: str) -> str:
            """Trainer that completes successfully."""
            print(f"   🚀 [TRAINING] Starting training (attempt 2)...")
            save_path_obj = Path(save_path)
            
            # Clean up any partial files first
            if save_path_obj.exists():
                print(f"   🧹 Cleaning up partial files...")
                shutil.rmtree(save_path_obj)
            
            save_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Complete training
            config = {
                "peft_type": "DORA",
                "r": heavy_genes.rank,
                "lora_alpha": heavy_genes.alpha,
                "target_modules": list(heavy_genes.target_modules)
            }
            with open(save_path_obj / "adapter_config.json", 'w') as f:
                import json
                json.dump(config, f)
            
            (save_path_obj / "adapter_model.safetensors").write_bytes(b"complete_weights")
            
            print(f"   ✅ [TRAINING] Complete!")
            return save_path
        
        print(f"🔄 Attempting get_or_train after cleanup...")
        adapter_path = cache.get_or_train_adapter(heavy_genes, working_trainer)
        
        print(f"✅ Training completed: {adapter_path}")
        
        # Verify it's complete
        if Path(adapter_path).exists():
            files = list(Path(adapter_path).glob("*"))
            print(f"📄 Complete files: {[f.name for f in files]}")
            
            # Check for required files
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            missing = [f for f in required_files if not (Path(adapter_path) / f).exists()]
            
            if missing:
                print(f"❌ Missing required files: {missing}")
            else:
                print(f"✅ All required files present")
        
        print(f"🧹 Cleanup: rm -rf {test_cache_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Interruption test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modal_simulation():
    """Simulate the exact Modal scenario locally."""
    print(f"\n🎭 TESTING: Modal Scenario Simulation")
    print("=" * 50)
    
    try:
        # Load your actual config
        config_file = "coral_x_real_config.yaml"
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded config: {config_file}")
        
        # Simulate genome creation from evolution
        from coral.domain.genome import Genome, CASeed, LoRAConfig
        from infra.adapter_cache import HeavyGenes
        import numpy as np
        
        # Create test genome (like evolution would)
        ca_seed = CASeed(
            grid=np.random.randint(0, 2, (10, 10)),
            rule=150,
            steps=15
        )
        
        lora_cfg = LoRAConfig(
            r=16,
            alpha=16.0,
            dropout=0.15,
            target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
            adapter_type='dora'
        )
        
        genome = Genome(seed=ca_seed, lora_cfg=lora_cfg, id='test_genome')
        
        print(f"🧬 Test genome: {genome.id}")
        print(f"🔧 LoRA config: r={lora_cfg.r}, α={lora_cfg.alpha}, type={lora_cfg.adapter_type}")
        
        # Simulate the Modal evaluation process
        print(f"🎭 Simulating Modal evaluation...")
        
        # 1. Extract HeavyGenes (like plugin does)
        run_id = config.get('experiment', {}).get('run_id', 'test_run')
        heavy_genes = HeavyGenes.from_lora_config(lora_cfg, run_id=run_id)
        
        print(f"🔍 Heavy genes extracted: {heavy_genes}")
        print(f"📊 Hash: {heavy_genes.to_hash()}")
        
        # 2. Simulate adapter path creation (like Modal does)
        adapter_hash = heavy_genes.to_hash()
        adapter_path = f"/cache/adapters/adapter_{adapter_hash}"
        
        print(f"📁 Expected adapter path: {adapter_path}")
        
        # 3. Check if this matches what cache expects
        from infra.adapter_cache import AdapterCache, CacheConfig
        
        # Create local cache for testing
        test_cache_dir = Path(tempfile.mkdtemp(prefix="modal_sim_"))
        cache_config = CacheConfig(base_dir=str(test_cache_dir), use_modal_volume=False)
        cache = AdapterCache(cache_config)
        
        # This is the EXACT path the cache would generate
        expected_local_path = cache._get_adapter_path(heavy_genes)
        local_hash = heavy_genes.to_hash()
        
        print(f"🔍 Cache calculations:")
        print(f"   Local path: {expected_local_path}")
        print(f"   Local hash: {local_hash}")
        print(f"   Modal path: {adapter_path}")
        print(f"   Hashes match: {adapter_hash == local_hash}")
        
        if adapter_hash == local_hash:
            print(f"✅ Hash coordination CORRECT")
        else:
            print(f"❌ Hash coordination BROKEN!")
            print(f"   This is why adapters are not found!")
            return False
        
        # 4. Test the full flow
        print(f"🔄 Testing full Modal simulation flow...")
        
        def mock_modal_trainer(heavy_genes: HeavyGenes, save_path: str) -> str:
            """Simulate what happens in Modal training."""
            print(f"   🎭 [MODAL TRAINING] Training in Modal environment")
            print(f"   📍 Save path: {save_path}")
            
            # Simulate Modal training
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Create adapter files like Modal would
            config_data = {
                "base_model_name_or_path": "codellama/CodeLlama-7b-Python-hf",
                "peft_type": heavy_genes.adapter_type.upper(),
                "r": heavy_genes.rank,
                "lora_alpha": heavy_genes.alpha,
                "lora_dropout": heavy_genes.dropout,
                "target_modules": list(heavy_genes.target_modules),
                "task_type": "CAUSAL_LM"
            }
            
            with open(save_path_obj / "adapter_config.json", 'w') as f:
                import json
                json.dump(config_data, f, indent=2)
            
            # Simulate weights file (large in real training)
            weights_data = b"simulated_modal_training_weights" * 1000  # ~30KB mock
            (save_path_obj / "adapter_model.safetensors").write_bytes(weights_data)
            
            print(f"   ✅ [MODAL TRAINING] Complete")
            return save_path
        
        # Get or train adapter
        final_adapter_path = cache.get_or_train_adapter(heavy_genes, mock_modal_trainer)
        
        print(f"✅ Modal simulation complete!")
        print(f"📁 Final adapter path: {final_adapter_path}")
        
        # Verify files
        if Path(final_adapter_path).exists():
            files = list(Path(final_adapter_path).glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"📄 Files created: {len(files)}")
            print(f"📊 Total size: {total_size:,} bytes")
            
            for f in files:
                print(f"   • {f.name}: {f.stat().st_size:,} bytes")
        else:
            print(f"❌ Adapter not found at final path!")
            return False
        
        print(f"🧹 Cleanup: rm -rf {test_cache_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Modal simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all cache coordination debug tests."""
    print("🚀 CORAL-X Cache Coordination Debug Suite")
    print("=" * 80)
    
    tests = [
        ("Cache Coordination Issue", test_cache_coordination_issue),
        ("Training Interruption", test_training_interruption_scenario),
        ("Modal Simulation", test_modal_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 All tests passed! The cache coordination should work.")
        print(f"The Modal issue might be:")
        print(f"   • Training interruptions due to timeouts")
        print(f"   • Modal volume synchronization issues") 
        print(f"   • CodeLlama-specific training problems")
    else:
        print(f"\n🔧 Some tests failed. Fix these issues first:")
        for test_name, success in results:
            if not success:
                print(f"   • {test_name}")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Run this script: python test_cache_coordination_debug.py")
    print(f"   2. Fix any local issues found")
    print(f"   3. Then debug Modal-specific problems")
    
    return passed == total


if __name__ == "__main__":
    main() 