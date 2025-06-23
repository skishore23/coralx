#!/usr/bin/env python3
"""
Test script to check CORAL-X cache mode and configuration
"""
import yaml
from pathlib import Path


def test_cache_configuration():
    """Test cache configuration and show expected behavior."""
    print("🧪 CORAL-X CACHE MODE TEST")
    print("=" * 60)
    
    # Load configuration
    config_file = "coral_x_codellama_config.yaml"
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        return
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Check cache configuration
    cache_config = config.get('cache', {})
    modal_native = cache_config.get('modal_native', False)
    artifacts_dir = cache_config.get('artifacts_dir', 'NOT SET')
    auto_sync = config.get('infra', {}).get('cache_volume', {}).get('auto_sync', False)
    
    print(f"📋 Cache Configuration:")
    print(f"   • modal_native: {modal_native}")
    print(f"   • artifacts_dir: {artifacts_dir}")
    print(f"   • auto_sync: {auto_sync}")
    
    # Check execution modes
    print(f"\n🚀 Execution Mode Behaviors:")
    
    if modal_native:
        print(f"   🎯 MODAL MODE (--executor=modal):")
        print(f"      • Everything runs on Modal")
        print(f"      • All caching in /cache/adapters")
        print(f"      • No local syncing needed")
        print(f"      • Local machine is just a client")
        print(f"      ✅ RECOMMENDED for production")
        
        print(f"\n   💻 LOCAL MODE (--executor=local):")
        print(f"      • Evolution runs locally")
        print(f"      • Uses Modal for individual operations")
        print(f"      • Primary cache: /cache/adapters")
        print(f"      • Backup cache: {artifacts_dir}")
        print(f"      ⚠️  More complex - for development only")
    else:
        print(f"   💻 LOCAL-ONLY MODE:")
        print(f"      • Everything runs locally")
        print(f"      • Cache: {artifacts_dir}")
        print(f"      • No Modal integration")
    
    # Check current environment
    import os
    print(f"\n🔍 Current Environment:")
    print(f"   • /cache exists: {os.path.exists('/cache')}")
    print(f"   • Local cache exists: {Path(artifacts_dir).exists()}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if modal_native:
        print(f"   ✅ Use: python run_coral_x_evolution.py --executor=modal")
        print(f"   ✅ Everything will run on Modal with native caching")
        print(f"   ✅ No local cache syncing - clean and fast")
    else:
        print(f"   💡 Consider setting cache.modal_native: true for cleaner Modal execution")


if __name__ == "__main__":
    test_cache_configuration() 