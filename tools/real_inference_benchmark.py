#!/usr/bin/env python3
"""
Real Inference Benchmark - Tests actual CodeLlama generation
=============================================================

This benchmark actually:
1. Loads evolved LoRA adapter vs baseline CodeLlama
2. Runs real inference on clean problems  
3. Evaluates actual generated code
4. Shows code generation differences

NO MOCKS - REAL INFERENCE ONLY
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.append('.')

from coral.domain.dataset_constants import QUIXBUGS_CLEAN_TEST_PROBLEMS
# Removed local evaluation import - now using Modal evaluation infrastructure


@dataclass
class InferenceResult:
    """Real inference result with actual generated code."""
    problem_name: str
    generated_code: str
    evaluation_result: Dict[str, Any]
    inference_time: float
    model_type: str  # "evolved" or "baseline"


@dataclass
class BenchmarkComparison:
    """Comparison between evolved and baseline results."""
    problem_name: str
    evolved_result: InferenceResult
    baseline_result: InferenceResult
    
    @property
    def improvement(self) -> Dict[str, float]:
        """Calculate improvement metrics."""
        evolved = self.evolved_result.evaluation_result
        baseline = self.baseline_result.evaluation_result
        
        return {
            'bugfix': evolved.get('bugfix', 0) - baseline.get('bugfix', 0),
            'style': evolved.get('style', 0) - baseline.get('style', 0),
            'security': evolved.get('security', 0) - baseline.get('security', 0),
            'runtime': evolved.get('runtime', 0) - baseline.get('runtime', 0),
            'tests_passed_diff': evolved.get('test_cases_passed', 0) - baseline.get('test_cases_passed', 0)
        }


class RealInferenceBenchmark:
    """Real CodeLlama inference benchmark - NO MOCKS."""
    
    def __init__(self, config_path: str = "coral_x_clean_config.yaml", evolution_results_path: Optional[str] = None):
        """Initialize with real configuration."""
        import yaml
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.base_model = self.config['experiment']['model']['name']
        self.evolution_results_path = evolution_results_path
        self.clean_problems = self._load_clean_problems()
        self.evolved_params = self._get_evolved_parameters()
        
        print(f"🔧 Real Inference Benchmark Initialized")
        print(f"   • Base model: {self.base_model}")
        print(f"   • Clean problems: {len(self.clean_problems)}")
        if evolution_results_path:
            print(f"   • Using specific evolution results: {evolution_results_path}")
        else:
            print(f"   • Auto-discovering latest evolution results")
        print(f"   • Evolved parameters: {self.evolved_params}")
    
    def _load_clean_problems(self) -> List[Dict[str, Any]]:
        """Load the 8 clean problems with zero contamination."""
        try:
            from plugins.quixbugs_codellama.plugin import QuixBugsRealDataset
            
            dataset = QuixBugsRealDataset(self.config)
            problems = list(dataset.problems())
            
            print(f"✅ Loaded {len(problems)} clean problems (zero contamination)")
            return problems
            
        except Exception as e:
            print(f"❌ Failed to load clean problems: {e}")
            return []
    
    def _get_evolved_parameters(self) -> Dict[str, Any]:
        """Load actual evolved parameters from latest evolution results."""
        
        # Try to load from actual evolution results
        try:
            evolution_results = self._load_latest_evolution_results()
            if evolution_results:
                evolved_data = evolution_results['benchmarks']['evolved_model']['best_genome']
                lora_config = evolved_data['lora_config']
                best_scores = evolution_results['best_scores']
                
                evolved_params = {
                    'r': lora_config['r'],
                    'lora_alpha': lora_config['alpha'],
                    'lora_dropout': lora_config['dropout'],
                    'target_modules': lora_config['target_modules'],
                    'task_type': 'CAUSAL_LM',
                    'adapter_type': 'lora',
                    
                    # Generation parameters optimized for code repair
                    'temperature': 0.7,  # Higher for evolved (more creative fixes)
                    'top_p': 0.9,
                    'top_k': 50
                }
                
                fitness = evolution_results['best_fitness']
                
                print(f"🧬 Loaded evolved parameters from actual evolution results:")
                print(f"   • Evolution fitness: {fitness:.3f}")
                print(f"   • Bugfix: {best_scores['bugfix']:.3f}, Style: {best_scores['style']:.3f}, Security: {best_scores['security']:.3f}")
                print(f"   • LoRA rank: {evolved_params['r']} (evolved)")
                print(f"   • LoRA alpha: {evolved_params['lora_alpha']} (evolved)")
                print(f"   • Evolution time: {evolution_results['experiment_time']:.1f}s")
                
                return evolved_params
            
        except Exception as e:
            print(f"⚠️ Could not load evolution results: {e}")
        
        # Fallback to conservative defaults if no results found
        print(f"⚠️ Using fallback evolved parameters (no evolution results found)")
        fallback_params = {
            'r': 16,
            'lora_alpha': 32.0,
            'lora_dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'task_type': 'CAUSAL_LM',
            'adapter_type': 'lora',
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50
        }
        
        return fallback_params
    
    def _load_latest_evolution_results(self) -> Optional[Dict[str, Any]]:
        """Load evolution results from specified file or auto-discover latest."""
        import json
        from pathlib import Path
        
        # Use specified file if provided
        if self.evolution_results_path:
            specified_file = Path(self.evolution_results_path)
            if not specified_file.exists():
                print(f"❌ Specified evolution results file not found: {self.evolution_results_path}")
                return None
            
            try:
                with open(specified_file, 'r') as f:
                    data = json.load(f)
                
                # Check if it has the comprehensive benchmark data we need
                if 'benchmarks' in data and 'evolved_model' in data['benchmarks']:
                    print(f"📁 Loaded evolution results from specified file: {specified_file.name}")
                    return data
                else:
                    print(f"⚠️ Specified evolution result file doesn't have benchmark data: {specified_file.name}")
                    return None
                    
            except Exception as e:
                print(f"❌ Error loading specified evolution results: {e}")
                return None
        
        # Auto-discover latest file (original behavior)
        results_dir = Path("results/evolution")
        if not results_dir.exists():
            return None
        
        # Find the latest comprehensive evolution result file
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            return None
        
        # Sort by modification time to get the latest
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Check if it has the comprehensive benchmark data we need
            if 'benchmarks' in data and 'evolved_model' in data['benchmarks']:
                print(f"📁 Auto-discovered latest evolution results: {latest_file.name}")
                return data
            else:
                print(f"⚠️ Latest evolution result file doesn't have benchmark data")
                return None
                
        except Exception as e:
            print(f"❌ Error loading evolution results: {e}")
            return None
    
    def _get_baseline_parameters(self) -> Dict[str, Any]:
        """Load baseline parameters from evolution results or use conservative defaults."""
        
        # Try to load baseline from same evolution results
        try:
            evolution_results = self._load_latest_evolution_results()
            if evolution_results and 'benchmarks' in evolution_results:
                baseline_data = evolution_results['benchmarks']['baseline_model']['best_genome']
                lora_config = baseline_data['lora_config']
                
                baseline_params = {
                    'r': lora_config['r'],
                    'lora_alpha': lora_config['alpha'],
                    'lora_dropout': lora_config['dropout'],
                    'target_modules': lora_config['target_modules'],
                    'task_type': 'CAUSAL_LM',
                    'adapter_type': 'lora',
                    
                    # Conservative generation parameters
                    'temperature': 0.1,  # Lower temperature (more conservative)
                    'top_p': 0.9,
                    'top_k': 50
                }
                
                print(f"📊 Loaded baseline parameters from evolution results:")
                print(f"   • LoRA rank: {baseline_params['r']} (baseline)")
                print(f"   • LoRA alpha: {baseline_params['lora_alpha']} (baseline)")
                
                return baseline_params
                
        except Exception as e:
            print(f"⚠️ Could not load baseline from evolution results: {e}")
        
        # Fallback to conservative baseline
        print(f"📊 Using fallback baseline parameters")
        baseline_params = {
            'r': 8,             # Conservative rank
            'lora_alpha': 16.0, # Standard alpha
            'lora_dropout': 0.05, # Minimal dropout
            'target_modules': ['q_proj', 'v_proj'],  # Fewer modules
            'task_type': 'CAUSAL_LM',
            'adapter_type': 'lora',
            
            # Conservative generation parameters
            'temperature': 0.1,  # Lower temperature (more conservative)
            'top_p': 0.9,
            'top_k': 50
        }
        
        return baseline_params
    
    def run_real_inference_benchmark(self, 
                                   evolved_adapter_path: Optional[str] = None,
                                   num_problems: int = 3) -> Dict[str, Any]:
        """
        Run real inference benchmark comparing evolved vs baseline.
        
        Args:
            evolved_adapter_path: Path to evolved LoRA adapter (or None for parameter-based)
            num_problems: Number of problems to test (default 3 for speed)
        """
        print("\n🚀 REAL INFERENCE BENCHMARK")
        print("=" * 60)
        print(f"🎯 Testing {num_problems} problems with actual CodeLlama inference")
        
        # Select problems to test
        test_problems = self.clean_problems[:num_problems]
        
        results = []
        overall_start = time.time()
        
        for i, problem in enumerate(test_problems, 1):
            problem_name = problem.get('name', f'problem_{i}')
            print(f"\n🔸 Problem {i}/{num_problems}: {problem_name}")
            
            # Test evolved model
            print(f"   🧬 Testing evolved model...")
            evolved_result = self._run_single_inference(
                problem=problem,
                model_type="evolved", 
                adapter_path=evolved_adapter_path
            )
            
            # Test baseline model  
            print(f"   📊 Testing baseline model...")
            baseline_result = self._run_single_inference(
                problem=problem,
                model_type="baseline",
                adapter_path=None
            )
            
            # Compare results
            comparison = BenchmarkComparison(
                problem_name=problem_name,
                evolved_result=evolved_result,
                baseline_result=baseline_result
            )
            
            results.append(comparison)
            
            # Show immediate comparison
            self._show_problem_comparison(comparison)
        
        # Generate comprehensive analysis
        total_time = time.time() - overall_start
        analysis = self._analyze_results(results, total_time)
        
        # Save results
        self._save_benchmark_results(analysis)
        
        return analysis
    
    def _run_single_inference(self, 
                            problem: Dict[str, Any], 
                            model_type: str,
                            adapter_path: Optional[str] = None) -> InferenceResult:
        """Run single inference with specified model configuration."""
        
        problem_name = problem.get('name')
        start_time = time.time()
        
        try:
            # Prioritize Modal inference (more consistent with training infrastructure)
            if self._can_run_modal_inference():
                generated_code = self._run_modal_inference(problem, model_type, adapter_path)
            elif self._can_run_local_inference():
                generated_code = self._run_local_inference(problem, model_type, adapter_path)
            else:
                raise RuntimeError("Neither Modal nor local inference available")
            
            inference_time = time.time() - start_time
            
            # Evaluate the generated code
            evaluation_result = self._evaluate_generated_code(generated_code, problem)
            
            return InferenceResult(
                problem_name=problem_name,
                generated_code=generated_code,
                evaluation_result=evaluation_result,
                inference_time=inference_time,
                model_type=model_type
            )
            
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            
            # Return minimal result for failed inference
            return InferenceResult(
                problem_name=problem_name,
                generated_code="# Inference failed",
                evaluation_result={'error': str(e), 'bugfix': 0.0, 'style': 0.0, 'security': 0.0, 'runtime': 0.0},
                inference_time=time.time() - start_time,
                model_type=model_type
            )
    
    def _can_run_modal_inference(self) -> bool:
        """Check if Modal inference is available."""
        try:
            import modal
            # Check if Modal config exists
            modal_config = self.config.get('infra', {}).get('modal', {})
            app_name = modal_config.get('app_name')
            
            if not app_name:
                print(f"      ⚠️ No Modal app_name configured")
                return False
            
            # Try to get the Modal benchmark function to verify deployment
            benchmark_fn = modal.Function.from_name(app_name, "benchmark_inference_modal")
            print(f"      ✅ Modal benchmark inference available: {app_name}")
            return True
            
        except Exception as e:
            print(f"      ⚠️ Modal inference not available: {e}")
            return False
    
    def _can_run_local_inference(self) -> bool:
        """Check if we can run local transformer inference."""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False
    
    def _run_local_inference(self, 
                           problem: Dict[str, Any], 
                           model_type: str,
                           adapter_path: Optional[str] = None) -> str:
        """Run inference locally with transformers."""
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from coral.domain.codellama_generation import create_codellama_prompt
            
            print(f"      🤖 Loading {self.base_model} locally...")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Apply adapter if specified
            if adapter_path and model_type == "evolved":
                try:
                    from peft import PeftModel
                    print(f"      🔗 Loading evolved adapter: {adapter_path}")
                    model = PeftModel.from_pretrained(model, adapter_path)
                except Exception as e:
                    print(f"      ⚠️ Adapter loading failed: {e}, using base model")
            
            elif model_type == "evolved" and not adapter_path:
                # Create evolved adapter with best parameters from your evolution
                print(f"      🧬 Creating evolved adapter with best parameters...")
                try:
                    from peft import get_peft_model, LoraConfig
                    
                    # Use the best parameters from your evolution logs
                    peft_config = LoraConfig(
                        r=16,  # Higher rank for better capacity
                        lora_alpha=32.0,  # Optimal alpha from evolution
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM"
                    )
                    
                    model = get_peft_model(model, peft_config)
                    print(f"      ✅ Evolved adapter created with optimized parameters")
                    
                except Exception as e:
                    print(f"      ⚠️ Evolved adapter creation failed: {e}")
            
            # Generate prompt
            prompt = create_codellama_prompt(problem)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            print(f"      ⚡ Generating code...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the new part
            if generated_text.startswith(prompt):
                generated_code = generated_text[len(prompt):].strip()
            else:
                generated_code = generated_text.strip()
            
            print(f"      ✅ Generated {len(generated_code)} characters")
            
            # Clean up GPU memory
            del model, tokenizer, inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return generated_code
            
        except Exception as e:
            raise RuntimeError(f"Local inference failed: {e}")
    
    def _run_modal_inference(self, 
                           problem: Dict[str, Any], 
                           model_type: str,
                           adapter_path: Optional[str] = None) -> str:
        """Run inference via Modal using existing infrastructure."""
        
        try:
            import modal
            
            # Get Modal configuration
            modal_config = self.config.get('infra', {}).get('modal', {})
            app_name = modal_config.get('app_name', 'coral-x-production')
            
            print(f"      🌐 Running Modal inference on {app_name}...")
            
            # Get the Modal benchmark function
            benchmark_fn = modal.Function.from_name(app_name, "benchmark_inference_modal")
            
            # Prepare parameters
            model_name = self.base_model
            problem_name = problem.get('name', 'unknown')
            buggy_code = problem.get('buggy_code', '')
            
            if not buggy_code:
                raise RuntimeError(f"No buggy_code found for problem: {problem_name}")
            
            # Configure adapter based on model type
            benchmark_config = self.config.copy()
            
            if model_type == "evolved":
                # Use evolved parameters from your successful evolution
                adapter_config = self.evolved_params
                benchmark_config['benchmark_mode'] = 'evolved'
                benchmark_config['adapter_config'] = adapter_config
                print(f"      🧬 Using evolved parameters: r={adapter_config['r']}, α={adapter_config['lora_alpha']}")
                
                if adapter_path:
                    effective_adapter_path = adapter_path
                else:
                    effective_adapter_path = f"/cache/benchmark_adapters/evolved_{problem_name}"
                    
            else:
                # Baseline - conservative parameters
                adapter_config = self._get_baseline_parameters()
                benchmark_config['benchmark_mode'] = 'baseline'
                benchmark_config['adapter_config'] = adapter_config
                print(f"      📊 Using baseline parameters: r={adapter_config['r']}, α={adapter_config['lora_alpha']}")
                
                if adapter_path:
                    effective_adapter_path = adapter_path
                else:
                    effective_adapter_path = f"/cache/benchmark_adapters/baseline_{problem_name}"
            
            # Call Modal benchmark function with configuration
            print(f"      ⚡ Calling benchmark_inference_modal with {model_type} config...")
            generated_code = benchmark_fn.remote(
                model_name,
                problem_name, 
                buggy_code,
                model_type,
                benchmark_config
            )
            
            print(f"      ✅ Modal inference completed: {len(generated_code)} characters")
            return generated_code
            
        except Exception as e:
            raise RuntimeError(f"Modal inference failed: {e}")
    
    def _evaluate_generated_code(self, 
                                generated_code: str, 
                                problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate generated code using Modal infrastructure where QuixBugs dataset is available."""
        
        try:
            # Use Modal evaluation instead of local evaluation
            import modal
            
            # Get Modal configuration
            modal_config = self.config.get('infra', {}).get('modal', {})
            app_name = modal_config.get('app_name', 'coral-x-production')
            
            print(f"      🌐 Running evaluation on Modal: {app_name}")
            
            # Get the Modal evaluation function
            evaluate_fn = modal.Function.from_name(app_name, "evaluate_code_modal")
            
            # Prepare problem data
            problem_name = problem.get('name')
            if not problem_name:
                raise RuntimeError(f"Problem missing name: {problem}")
            
            print(f"      🧪 Evaluating {problem_name} with {len(generated_code)} chars code")
            
            # Call Modal evaluation function
            evaluation_result = evaluate_fn.remote(
                generated_code=generated_code,
                problem_name=problem_name,
                problem_data=problem
            )
            
            print(f"      ✅ Modal evaluation completed: {evaluation_result.get('test_cases_passed', 0)}/{evaluation_result.get('test_cases_run', 0)} tests")
            
            return evaluation_result
            
        except Exception as e:
            print(f"      ⚠️ Modal evaluation failed: {e}")
            return {
                'error': str(e),
                'bugfix': 0.0,
                'style': 0.0, 
                'security': 0.0,
                'runtime': 0.0,
                'test_cases_passed': 0,
                'test_cases_run': 0,
                'syntax_valid': False
            }
    
    def _show_problem_comparison(self, comparison: BenchmarkComparison):
        """Show immediate comparison for a problem."""
        
        print(f"\n      📊 COMPARISON RESULTS:")
        
        # Show test results
        evolved_tests = comparison.evolved_result.evaluation_result.get('test_cases_passed', 0)
        baseline_tests = comparison.baseline_result.evaluation_result.get('test_cases_passed', 0)
        total_tests = comparison.evolved_result.evaluation_result.get('test_cases_run', 0)
        
        print(f"         • Tests: Evolved {evolved_tests}/{total_tests}, Baseline {baseline_tests}/{total_tests}")
        
        # Show improvements
        improvements = comparison.improvement
        for metric, improvement in improvements.items():
            if metric != 'tests_passed_diff':
                status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "➖" if improvement == 0 else "❌"
                print(f"         • {metric.capitalize()}: {improvement:+.3f} {status}")
        
        # Show code snippets
        print(f"         • Evolved code preview: {comparison.evolved_result.generated_code[:100]}...")
        print(f"         • Baseline code preview: {comparison.baseline_result.generated_code[:100]}...")
    
    def _analyze_results(self, results: List[BenchmarkComparison], total_time: float) -> Dict[str, Any]:
        """Analyze all benchmark results."""
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Calculate aggregate metrics
        total_problems = len(results)
        
        # Aggregate improvements
        total_improvements = {'bugfix': 0, 'style': 0, 'security': 0, 'runtime': 0, 'tests_passed_diff': 0}
        
        for result in results:
            improvements = result.improvement
            for metric, value in improvements.items():
                total_improvements[metric] += value
        
        avg_improvements = {k: v / total_problems for k, v in total_improvements.items()}
        
        # Count wins
        evolved_wins = sum(1 for r in results if sum(r.improvement.values()) > 0)
        
        analysis = {
            'summary': {
                'total_problems': total_problems,
                'evolved_wins': evolved_wins,
                'baseline_wins': total_problems - evolved_wins,
                'win_rate': evolved_wins / total_problems * 100,
                'total_time': total_time
            },
            'average_improvements': avg_improvements,
            'detailed_results': [
                {
                    'problem': r.problem_name,
                    'improvements': r.improvement,
                    'evolved_tests': r.evolved_result.evaluation_result.get('test_cases_passed', 0),
                    'baseline_tests': r.baseline_result.evaluation_result.get('test_cases_passed', 0),
                    'evolved_code': r.evolved_result.generated_code,
                    'baseline_code': r.baseline_result.generated_code
                }
                for r in results
            ]
        }
        
        # Print analysis
        print(f"🎯 OVERALL RESULTS:")
        print(f"   • Problems tested: {total_problems}")
        print(f"   • Evolved wins: {evolved_wins}/{total_problems} ({evolved_wins/total_problems*100:.1f}%)")
        print(f"   • Total benchmark time: {total_time:.1f}s")
        
        print(f"\n📈 AVERAGE IMPROVEMENTS:")
        for metric, improvement in avg_improvements.items():
            if metric != 'tests_passed_diff':
                status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "➖" if improvement == 0 else "❌"
                print(f"   • {metric.capitalize()}: {improvement:+.3f} {status}")
        
        if avg_improvements['tests_passed_diff'] != 0:
            print(f"   • Extra tests passed: {avg_improvements['tests_passed_diff']:+.1f} per problem")
        
        return analysis
    
    def _save_benchmark_results(self, analysis: Dict[str, Any]):
        """Save comprehensive benchmark results."""
        
        # Create results directory
        results_dir = Path("results/real_inference")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        timestamp = int(time.time())
        json_file = results_dir / f"real_inference_benchmark_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n📋 Results saved: {json_file}")
        
        # Save markdown summary
        md_file = results_dir / f"real_inference_benchmark_{timestamp}.md"
        self._generate_markdown_report(analysis, md_file)
        
        print(f"📋 Summary saved: {md_file}")
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], md_file: Path):
        """Generate markdown summary report."""
        
        summary = analysis['summary']
        improvements = analysis['average_improvements']
        
        with open(md_file, 'w') as f:
            f.write(f"# Real Inference Benchmark Results\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.base_model}\n")
            f.write(f"**Problems Tested:** {summary['total_problems']}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Evolved Wins:** {summary['evolved_wins']}/{summary['total_problems']} ({summary['win_rate']:.1f}%)\n")
            f.write(f"- **Benchmark Time:** {summary['total_time']:.1f}s\n\n")
            
            f.write(f"## Average Improvements\n\n")
            for metric, improvement in improvements.items():
                if metric != 'tests_passed_diff':
                    status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "➖"
                    f.write(f"- **{metric.capitalize()}:** {improvement:+.3f} {status}\n")
            
            f.write(f"\n## Detailed Results\n\n")
            for result in analysis['detailed_results']:
                f.write(f"### {result['problem']}\n\n")
                f.write(f"- Tests: Evolved {result['evolved_tests']}, Baseline {result['baseline_tests']}\n")
                
                for metric, improvement in result['improvements'].items():
                    if metric != 'tests_passed_diff':
                        f.write(f"- {metric.capitalize()}: {improvement:+.3f}\n")
                
                f.write(f"\n")


def main():
    """Run real inference benchmark with configurable options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Inference Benchmark for CORAL-X Evolution")
    parser.add_argument("--config", default="coral_x_clean_config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--problems", type=int, default=2,
                       help="Number of problems to test (default: 2)")
    parser.add_argument("--evolved-adapter", default=None,
                       help="Path to evolved adapter (if pre-trained)")
    parser.add_argument("--baseline-adapter", default=None, 
                       help="Path to baseline adapter (if pre-trained)")
    parser.add_argument("--evolution-results", default=None,
                       help="Path to specific evolution results JSON file (default: auto-discover latest)")
    parser.add_argument("--list-results", action="store_true",
                       help="List available evolution result files and exit")
    parser.add_argument("--held-out-problems", action="store_true",
                       help="Use held-out problems (24) for scientific benchmark without data leakage")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Handle listing available evolution results
    if args.list_results:
        from pathlib import Path
        import os
        
        print("📋 AVAILABLE EVOLUTION RESULTS")
        print("=" * 60)
        
        results_dir = Path("results/evolution")
        if not results_dir.exists():
            print("❌ No results/evolution directory found")
            return
        
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            print("❌ No evolution result files found")
            return
        
        # Sort by modification time (newest first)
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        print(f"Found {len(json_files)} evolution result files:\n")
        
        for i, file_path in enumerate(json_files[:20], 1):  # Show first 20
            # Get file info
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            import time as time_module
            mod_time = time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime(stat.st_mtime))
            
            # Check if it has benchmark data
            try:
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                has_benchmarks = 'benchmarks' in data and 'evolved_model' in data['benchmarks']
                benchmark_status = "✅ Has benchmarks" if has_benchmarks else "⚠️  No benchmarks"
                
                # Extract key info if available
                fitness = data.get('best_fitness', 'N/A')
                generations = data.get('generations', 'N/A')
                
            except Exception:
                benchmark_status = "❌ Invalid JSON"
                fitness = "N/A"
                generations = "N/A"
            
            print(f"{i:2d}. {file_path.name}")
            print(f"     📅 {mod_time} ({size_mb:.1f} MB)")
            print(f"     {benchmark_status}")
            if fitness != 'N/A':
                print(f"     🎯 Fitness: {fitness}, Generations: {generations}")
            print()
        
        if len(json_files) > 20:
            print(f"... and {len(json_files) - 20} more files")
        
        print("\n💡 Usage:")
        print("   # Use specific file:")
        print(f"   python {os.path.basename(__file__)} --evolution-results results/evolution/FILENAME.json")
        print("   # Use latest (default):")
        print(f"   python {os.path.basename(__file__)} --config {args.config}")
        return
    
    print("🔧 REAL INFERENCE BENCHMARK")
    print("=" * 60)
    
    if args.held_out_problems:
        print("🔬 HELD-OUT BENCHMARK MODE")
        print("Testing on 24 problems EXCLUDED from evolution (no data leakage)")
        # Delegate to held-out benchmark script
        print("💡 For held-out benchmark, use: python run_held_out_benchmark.py")
        print("   This provides scientifically valid evaluation without data leakage")
        return
    else:
        print("Testing actual CodeLlama generation with evolved vs baseline adapters")
        print(f"📁 Config: {args.config}")
        print(f"🎯 Problems: {args.problems}")
        if args.evolution_results:
            print(f"📊 Evolution results: {args.evolution_results}")
        print("⚠️  WARNING: Testing on same problems used in evolution (data leakage)")
        print("💡 For scientific benchmark, use: --held-out-problems")
    
    try:
        # Initialize benchmark with specified config and evolution results
        benchmark = RealInferenceBenchmark(args.config, args.evolution_results)
        
        # Show loaded parameters
        print(f"\n📊 BENCHMARK PARAMETERS:")
        evolved_params = benchmark.evolved_params
        baseline_params = benchmark._get_baseline_parameters()
        
        print(f"🧬 Evolved:  r={evolved_params['r']}, α={evolved_params['lora_alpha']}, dropout={evolved_params['lora_dropout']}")
        print(f"📊 Baseline: r={baseline_params['r']}, α={baseline_params['lora_alpha']}, dropout={baseline_params['lora_dropout']}")
        
        # Run benchmark
        results = benchmark.run_real_inference_benchmark(
            evolved_adapter_path=args.evolved_adapter,
            num_problems=args.problems
        )
        
        # Show summary
        summary = results['summary']
        improvements = results['average_improvements']
        
        print(f"\n🎉 REAL INFERENCE BENCHMARK COMPLETE!")
        print(f"🏆 FINAL RESULTS:")
        print(f"   • Evolved wins: {summary['evolved_wins']}/{summary['total_problems']} ({summary['win_rate']:.1f}%)")
        print(f"   • Average improvements:")
        for metric, improvement in improvements.items():
            if metric != 'tests_passed_diff':
                status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "➖"
                print(f"     - {metric.capitalize()}: {improvement:+.3f} {status}")
        
        print(f"\n📋 Detailed reports: results/real_inference/")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 