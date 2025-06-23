#!/usr/bin/env python3
"""
Test cache coordination between training and generation phases on Modal.
"""
import modal


def run_cache_coordination_test():
    """Run the cache coordination test on Modal."""
    print("🧪 Running Cache Coordination Test on Modal")
    print("=" * 50)
    
    try:
        # Get the Modal function
        app_name = "coral-x-production"
        test_fn = modal.Function.from_name(app_name, "test_cache_coordination_modal")
        
        print(f"📡 Calling Modal function: {app_name}/test_cache_coordination_modal")
        
        # Run the test
        result = test_fn.remote()
        
        print(f"\n📊 TEST RESULTS:")
        print("=" * 30)
        
        if result.get('success', False):
            print(f"✅ CACHE COORDINATION TEST PASSED!")
            print(f"   • Hash consistency: ✅")
            print(f"   • Training hash: {result.get('training_hash', 'N/A')}")
            print(f"   • Generation hash: {result.get('generation_hash', 'N/A')}")
            print(f"   • Total adapters found: {result.get('total_adapters', 0)}")
        else:
            print(f"❌ CACHE COORDINATION TEST FAILED!")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            else:
                print(f"   • Hash consistency: {'✅' if result.get('hash_consistent', False) else '❌'}")
                print(f"   • Training hash: {result.get('training_hash', 'N/A')}")
                print(f"   • Generation hash: {result.get('generation_hash', 'N/A')}")
                print(f"   • Expected adapter exists: {'✅' if result.get('expected_adapter_exists', False) else '❌'}")
                print(f"   • Total adapters found: {result.get('total_adapters', 0)}")
        
        print(f"\n🔧 Test Details:")
        print(f"   • Test genome ID: {result.get('test_genome_id', 'N/A')}")
        print(f"   • Run ID: {result.get('run_id', 'N/A')}")
        print(f"   • Adapter type: {result.get('adapter_type', 'N/A')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Failed to run cache coordination test: {e}")
        return False


if __name__ == "__main__":
    success = run_cache_coordination_test()
    
    print(f"\n{'🎉 Success!' if success else '⚠️  Issues detected'}")
    print("=" * 50)
    
    if success:
        print("Cache coordination is working correctly!")
        print("Your CoralX evolution should work without cache misses.")
    else:
        print("Cache coordination issues detected.")
        print("This explains why adapters are not being found during generation.")
        print("The fix may need further refinement.")
    
    exit(0 if success else 1) 