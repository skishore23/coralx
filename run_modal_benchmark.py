#!/usr/bin/env python3
"""
Run complete benchmark on Modal with hardcoded evolution results.
Uses your successful evolution parameters (fitness: 0.945, r=16, alpha=32.0, DoRA).
"""

import modal
import json
import time
from pathlib import Path

def main(num_problems: int = 3):
    """Run complete benchmark on Modal."""
    
    print("🎯 MODAL COMPLETE BENCHMARK")
    print("=" * 60)
    print(f"Running benchmark with {num_problems} problems")
    print("Using hardcoded evolution parameters from your successful run")
    print("(fitness: 0.945, r=16, alpha=32.0, DoRA)")
    
    try:
        # Get Modal function
        app_name = "coral-x-production"
        benchmark_fn = modal.Function.from_name(app_name, "run_complete_benchmark_modal")
        
        print(f"\n🌐 Connecting to Modal app: {app_name}")
        
        # Basic config (not used much since parameters are hardcoded)
        config_dict = {
            'experiment': {
                'model': {'name': 'codellama/CodeLlama-7b-Python-hf'},
                'dataset': {'path': '/cache/quixbugs_dataset'}
            }
        }
        
        # Run benchmark on Modal
        print(f"🚀 Starting Modal benchmark...")
        start_time = time.time()
        
        results = benchmark_fn.remote(
            config_dict=config_dict,
            num_problems=num_problems
        )
        
        total_time = time.time() - start_time
        
        # Save results locally
        results_dir = Path("results/modal_benchmark")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"modal_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 MODAL BENCHMARK COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"💾 Results saved: {results_file}")
        
        # Show summary
        summary = results['summary']
        improvements = results['average_improvements']
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   • Problems tested: {summary['total_problems']}")
        print(f"   • Evolved wins: {summary['evolved_wins']}/{summary['total_problems']} ({summary['win_rate']:.1f}%)")
        
        print(f"\n📈 AVERAGE IMPROVEMENTS:")
        for metric, improvement in improvements.items():
            if metric != 'tests_passed_diff':
                status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "➖" if improvement == 0 else "❌"
                print(f"   • {metric.capitalize()}: {improvement:+.3f} {status}")
        
        if improvements.get('tests_passed_diff', 0) != 0:
            print(f"   • Extra tests passed: {improvements['tests_passed_diff']:+.1f} per problem")
        
        # Show evolved vs baseline parameters
        evolved = results['evolved_parameters']
        baseline = results['baseline_parameters']
        
        print(f"\n🧬 PARAMETERS COMPARISON:")
        print(f"   Evolved:  r={evolved['r']}, α={evolved['lora_alpha']}, {evolved['adapter_type']}")
        print(f"   Baseline: r={baseline['r']}, α={baseline['lora_alpha']}, {baseline['adapter_type']}")
        
        # Show detailed results
        print(f"\n📊 DETAILED RESULTS:")
        for result in results['detailed_results']:
            problem = result['problem']
            evolved_tests = result['evolved_result']['evaluation_result'].get('test_cases_passed', 0)
            evolved_total = result['evolved_result']['evaluation_result'].get('test_cases_run', 0)
            baseline_tests = result['baseline_result']['evaluation_result'].get('test_cases_passed', 0)
            baseline_total = result['baseline_result']['evaluation_result'].get('test_cases_run', 0)
            
            print(f"   {problem}:")
            print(f"     • Evolved:  {evolved_tests}/{evolved_total} tests")
            print(f"     • Baseline: {baseline_tests}/{baseline_total} tests")
            
            improvements_detailed = result['improvements']
            for metric, improvement in improvements_detailed.items():
                if metric != 'tests_passed_diff' and improvement != 0:
                    status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "❌"
                    print(f"     • {metric}: {improvement:+.3f} {status}")
        
        return results
        
    except Exception as e:
        print(f"❌ Modal benchmark failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Modal complete benchmark")
    parser.add_argument("--problems", type=int, default=3,
                       help="Number of problems to test (default: 3)")
    
    args = parser.parse_args()
    
    main(args.problems) 