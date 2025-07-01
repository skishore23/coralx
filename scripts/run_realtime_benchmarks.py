#!/usr/bin/env python3
"""
Realtime Benchmark Runner
========================

Runs continuous benchmarks using the separate coral-x-benchmarks Modal app.
This system monitors adapter performance with neutral parameters while
evolution runs with CA-derived parameters in parallel.

Usage:
    python run_realtime_benchmarks.py --interval 300 --problems 8
"""

import time
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import asyncio

# Import the separate benchmark app
from coral_modal_benchmark_app import benchmark_app

def get_evolved_adapters(config: Dict[str, Any] = None) -> List[str]:
    """Get list of evolved adapters from cache using config paths."""
    try:
        if config:
            from coral.config.path_utils import get_adapters_path
            adapters_path = Path(get_adapters_path(config))
            print(f"📁 Using adapters path from config: {adapters_path}")
        else:
            # Fallback: try local first, then modal
            local_cache = Path("coral_cache/adapters")
            if local_cache.exists():
                adapters_path = local_cache
                print(f"📁 Using local adapters path: {adapters_path}")
            else:
                adapters_path = Path("/cache/adapters")
                print(f"📁 Using default Modal adapters path: {adapters_path}")
        
        if adapters_path.exists():
            adapters = [str(p) for p in adapters_path.iterdir() if p.is_dir()]
            print(f"📁 Found {len(adapters)} adapters")
            return adapters[:5]  # Limit for testing
        else:
            print(f"⚠️  Adapters directory not found: {adapters_path}")
            return []
        
    except Exception as e:
        print(f"⚠️  Adapter discovery failed: {e}")
        return []

def get_test_problems() -> List[Dict[str, Any]]:
    """Get clean test problems for benchmarking using centralized dataset constants."""
    from coral.domain.dataset_constants import QUIXBUGS_CLEAN_TEST_PROBLEMS
    from adapters.quixbugs_real import QuixBugsRealAdapter
    
    try:
        # Use the real dataset adapter to get actual problems
        # Create adapter with minimal fallback config for dataset path
        fallback_config = {
            'paths': {
                'modal': {
                    'dataset': '/cache/quixbugs_dataset'
                }
            },
            'infra': {
                'executor': 'modal'
            }
        }
        adapter = QuixBugsRealAdapter(config=fallback_config)
        all_problems = list(adapter.problems())
        
        # Filter to only clean test problems (no contamination)
        clean_problems = [
            problem for problem in all_problems 
            if problem.get('name') in QUIXBUGS_CLEAN_TEST_PROBLEMS
        ]
        
        print(f"📊 Using {len(clean_problems)} clean test problems from centralized constants")
        print(f"   • Problems: {sorted([p.get('name') for p in clean_problems])}")
        
        return clean_problems
        
    except Exception as e:
        print(f"⚠️  Failed to load real problems: {e}")
        print(f"   • Falling back to subset of clean problems")
        
        # Fallback: return subset of known clean problems
        fallback_problems = []
        known_clean = ['kheapsort', 'sieve', 'to_base']  # Subset of QUIXBUGS_CLEAN_TEST_PROBLEMS
        
        for name in known_clean:
            fallback_problems.append({
                "name": name,
                "buggy_code": f"# Placeholder for {name} - use real dataset adapter",
                "prompt": f"Fix the buggy function in {name}.py"
            })
        
        return fallback_problems

class RealtimeBenchmarkRunner:
    """Runs continuous benchmarks using separate Modal app."""
    
    def __init__(self, interval: int = 300):
        self.interval = interval
        self.benchmark_history = []
        
    def run_continuous_benchmarks(self, max_iterations: int = None):
        """Run continuous benchmarks in background."""
        print(f"🚀 REALTIME BENCHMARK RUNNER STARTING")
        print(f"   • Interval: {self.interval}s ({self.interval/60:.1f} minutes)")
        print(f"   • Modal app: coral-x-benchmarks (isolated)")
        print(f"   • Parameters: NEUTRAL ONLY (temp=0.7, top_p=0.9)")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n🔄 BENCHMARK ITERATION {iteration}")
                print(f"{'='*50}")
                
                # Get current evolved adapters
                adapters = get_evolved_adapters()
                if not adapters:
                    print(f"   ⚠️  No adapters found, waiting...")
                    time.sleep(self.interval)
                    continue
                
                # Get test problems
                problems = get_test_problems()
                
                print(f"📊 Running benchmarks:")
                print(f"   • Adapters: {len(adapters)}")
                print(f"   • Problems: {len(problems)}")
                print(f"   • Mode: Neutral parameters only")
                
                # Run benchmarks using separate Modal app
                try:
                    # Use the separate benchmark app
                    with benchmark_app.run():
                        from coral_modal_benchmark_app import run_realtime_benchmark_modal
                        
                        result = run_realtime_benchmark_modal.remote(
                            adapter_paths=adapters,
                            test_problems=problems,
                            benchmark_interval=self.interval
                        )
                        
                        # Store results
                        self.benchmark_history.append({
                            "iteration": iteration,
                            "timestamp": time.time(),
                            "results": result,
                            "adapters_tested": len(adapters),
                            "problems_tested": len(problems)
                        })
                        
                        print(f"✅ Benchmark iteration {iteration} completed")
                        self._print_benchmark_summary(result)
                        
                        # Save results
                        self._save_benchmark_results()
                        
                except Exception as e:
                    print(f"❌ Benchmark iteration {iteration} failed: {e}")
                
                # Check if we should stop
                if max_iterations and iteration >= max_iterations:
                    print(f"🏁 Reached maximum iterations ({max_iterations})")
                    break
                
                # Wait for next iteration
                print(f"⏳ Waiting {self.interval}s for next benchmark...")
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Benchmark runner stopped by user")
        except Exception as e:
            print(f"❌ Benchmark runner failed: {e}")
        finally:
            print(f"📊 Total benchmark iterations completed: {iteration}")
            self._print_final_summary()
    
    def _print_benchmark_summary(self, result: Dict[str, Any]):
        """Print summary of benchmark results."""
        print(f"\n📈 BENCHMARK SUMMARY:")
        
        if "benchmark_results" in result:
            for adapter_result in result["benchmark_results"]:
                adapter_path = adapter_result["adapter_path"]
                adapter_name = Path(adapter_path).name
                
                successes = sum(1 for r in adapter_result["results"] if "error" not in r)
                total = len(adapter_result["results"])
                
                print(f"   • {adapter_name}: {successes}/{total} problems succeeded")
        
        print(f"   • Parameters: {result.get('parameters_used', 'unknown')}")
        print(f"   • Benchmark type: {result.get('benchmark_type', 'unknown')}")
    
    def _save_benchmark_results(self):
        """Save benchmark results to file."""
        try:
            results_dir = Path("results/realtime_benchmarks")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            results_file = results_dir / f"benchmark_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.benchmark_history, f, indent=2)
            
            print(f"💾 Results saved: {results_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    def _print_final_summary(self):
        """Print final summary of all benchmark runs."""
        if not self.benchmark_history:
            return
        
        print(f"\n🏆 FINAL BENCHMARK SUMMARY")
        print(f"{'='*50}")
        print(f"   • Total iterations: {len(self.benchmark_history)}")
        print(f"   • Time span: {self._get_time_span()}")
        print(f"   • Average adapters per iteration: {self._get_avg_adapters():.1f}")
        print(f"   • Results saved to: results/realtime_benchmarks/")
    
    def _get_time_span(self) -> str:
        """Get time span of benchmark runs."""
        if len(self.benchmark_history) < 2:
            return "0 minutes"
        
        start_time = self.benchmark_history[0]["timestamp"]
        end_time = self.benchmark_history[-1]["timestamp"]
        duration_minutes = (end_time - start_time) / 60
        
        return f"{duration_minutes:.1f} minutes"
    
    def _get_avg_adapters(self) -> float:
        """Get average number of adapters tested per iteration."""
        if not self.benchmark_history:
            return 0.0
        
        total_adapters = sum(h["adapters_tested"] for h in self.benchmark_history)
        return total_adapters / len(self.benchmark_history)

def main():
    parser = argparse.ArgumentParser(description="Run realtime benchmarks with neutral parameters")
    parser.add_argument("--interval", type=int, default=300, 
                       help="Benchmark interval in seconds (default: 300)")
    parser.add_argument("--max-iterations", type=int, default=None,
                       help="Maximum benchmark iterations (default: unlimited)")
    parser.add_argument("--problems", type=int, default=3,
                       help="Number of test problems (default: 3)")
    
    args = parser.parse_args()
    
    print(f"🚀 REALTIME BENCHMARK SYSTEM")
    print(f"   • Interval: {args.interval}s")
    print(f"   • Max iterations: {args.max_iterations or 'unlimited'}")
    print(f"   • Test problems: {args.problems}")
    print(f"   • Modal app: coral-x-benchmarks (isolated)")
    print(f"   • Parameters: NEUTRAL ONLY")
    
    runner = RealtimeBenchmarkRunner(interval=args.interval)
    runner.run_continuous_benchmarks(max_iterations=args.max_iterations)

if __name__ == "__main__":
    main() 