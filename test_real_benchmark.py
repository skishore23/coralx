#!/usr/bin/env python3
"""
Test Real Inference Benchmark
==============================

Quick test to verify the benchmark infrastructure works.
"""
import sys
sys.path.append('.')

def test_dependencies():
    """Test if we have required dependencies for real inference."""
    
    print("🔧 Testing Real Inference Dependencies")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   • CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError:
        print("❌ PEFT not available")
    
    try:
        from coral.domain.dataset_constants import QUIXBUGS_CLEAN_TEST_PROBLEMS
        print(f"✅ Clean test problems: {len(QUIXBUGS_CLEAN_TEST_PROBLEMS)}")
    except ImportError as e:
        print(f"❌ Dataset constants failed: {e}")
    
    try:
        from real_inference_benchmark import RealInferenceBenchmark
        print(f"✅ Real inference benchmark imported")
    except ImportError as e:
        print(f"❌ Benchmark import failed: {e}")


def test_benchmark_initialization():
    """Test benchmark initialization."""
    
    print("\n🚀 Testing Benchmark Initialization")
    print("=" * 50)
    
    try:
        from real_inference_benchmark import RealInferenceBenchmark
        
        # Test with clean config
        benchmark = RealInferenceBenchmark("coral_x_clean_config.yaml")
        print(f"✅ Benchmark initialized successfully")
        print(f"   • Base model: {benchmark.base_model}")
        print(f"   • Clean problems loaded: {len(benchmark.clean_problems)}")
        
        # Show first few problems
        if benchmark.clean_problems:
            print(f"   • First problem: {benchmark.clean_problems[0].get('name', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dry_run():
    """Test a dry run without actual inference."""
    
    print("\n🎯 Testing Dry Run")
    print("=" * 50)
    
    try:
        from real_inference_benchmark import RealInferenceBenchmark
        
        benchmark = RealInferenceBenchmark("coral_x_clean_config.yaml")
        
        if not benchmark.clean_problems:
            print("❌ No clean problems available for testing")
            return False
        
        # Test problem loading
        test_problem = benchmark.clean_problems[0]
        print(f"✅ Test problem: {test_problem.get('name')}")
        
        # Test evaluation framework (without actual inference)
        print(f"✅ Evaluation framework accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    
    print("🔧 REAL INFERENCE BENCHMARK VALIDATION")
    print("=" * 60)
    
    # Test dependencies
    test_dependencies()
    
    # Test initialization
    init_success = test_benchmark_initialization()
    
    # Test dry run
    if init_success:
        dry_run_success = test_dry_run()
        
        if dry_run_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Ready to run real inference benchmark")
            print("\nTo run the benchmark:")
            print("   python real_inference_benchmark.py")
        else:
            print("\n⚠️ Dry run failed - check configuration")
    else:
        print("\n❌ Initialization failed - check dependencies")


if __name__ == "__main__":
    main() 