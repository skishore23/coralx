#!/usr/bin/env python3
"""
Debug script to isolate LoRA training issues.
"""
import sys
from pathlib import Path
from coral.domain.lora_training import train_codellama_lora
from infra.adapter_cache import HeavyGenes, AdapterCache, CacheConfig


def test_lora_training():
    """Test LoRA training with debug output."""
    print("🧪 TESTING LORA TRAINING")
    print("=" * 50)
    
    # Create test heavy genes
    heavy_genes = HeavyGenes(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=('q_proj', 'v_proj'),
        adapter_type='lora',  # Add required field
        run_id='test_run'     # Add required field
    )
    
    print(f"🔧 Heavy genes: {heavy_genes}")
    print(f"📊 Hash: {heavy_genes.to_hash()}")
    
    # Test training
    test_adapter_path = "/tmp/test_lora_adapter"
    print(f"📍 Training to: {test_adapter_path}")
    
    try:
        print("🚀 Starting LoRA training...")
        result = train_codellama_lora(
            'codellama/CodeLlama-7b-Python-hf',
            heavy_genes,
            test_adapter_path
        )
        
        print(f"✅ Training completed successfully!")
        print(f"📁 Result path: {result}")
        
        # Check if adapter was actually saved
        result_path = Path(result)
        if result_path.exists():
            print(f"✅ Adapter exists at: {result_path}")
            if result_path.is_dir():
                files = list(result_path.glob("*"))
                print(f"📄 Contains {len(files)} files:")
                for f in files[:5]:  # Show first 5 files
                    print(f"   • {f.name}")
            else:
                print(f"📄 Single file: {result_path.stat().st_size} bytes")
        else:
            print(f"❌ Adapter NOT found at: {result_path}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_adapter_cache():
    """Test adapter cache system."""
    print("\n🗂️  TESTING ADAPTER CACHE")
    print("=" * 50)
    
    # Create cache config
    cache_config = CacheConfig(
        artifacts_dir="./coral_cache/adapters",
        base_checkpoint="codellama/CodeLlama-7b-Python-hf",
        cache_metadata=True,
        cleanup_threshold=100
    )
    
    print(f"📁 Cache config: {cache_config}")
    
    # Test cache
    cache = AdapterCache(cache_config)
    
    # Create test heavy genes
    heavy_genes = HeavyGenes(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=('q_proj', 'v_proj'),
        adapter_type='lora',  # Add required field
        run_id='test_run'     # Add required field
    )
    
    def mock_trainer(genes: HeavyGenes, base_ckpt: str) -> str:
        """Mock trainer that doesn't actually train."""
        print(f"🔄 Mock training: {genes} with {base_ckpt}")
        
        # Create a mock adapter directory
        mock_path = Path("/tmp/mock_adapter")
        mock_path.mkdir(exist_ok=True)
        
        # Create some mock files
        (mock_path / "adapter_config.json").write_text('{"test": "mock"}')
        (mock_path / "adapter_model.bin").write_text('mock_model_data')
        
        return str(mock_path)
    
    try:
        print("🔄 Testing cache get_or_train...")
        adapter_path = cache.get_or_train_adapter(heavy_genes, mock_trainer)
        print(f"✅ Cache returned: {adapter_path}")
        
        # Check if adapter exists
        if Path(adapter_path).exists():
            print(f"✅ Adapter exists and accessible")
        else:
            print(f"❌ Adapter path returned but doesn't exist")
            
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("🧪 CORAL-X LORA TRAINING DEBUG")
    print("=" * 60)
    
    # Test 1: LoRA training
    training_success = test_lora_training()
    
    # Test 2: Adapter cache
    cache_success = test_adapter_cache()
    
    print("\n📊 SUMMARY")
    print("=" * 60)
    print(f"✅ LoRA Training: {'PASS' if training_success else 'FAIL'}")
    print(f"✅ Adapter Cache: {'PASS' if cache_success else 'FAIL'}")
    
    if training_success and cache_success:
        print("🎉 All tests passed - LoRA system is working!")
    else:
        print("❌ Some tests failed - debug needed")
        sys.exit(1) 