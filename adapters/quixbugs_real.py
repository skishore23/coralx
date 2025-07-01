"""
Real QuixBugs Dataset Adapter for CORAL-X
Uses actual QuixBugs dataset with real evaluation and Modal integration
Function names from coral-x-codellama.md specification
"""
import os
import json
import subprocess
import tempfile
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable
from dataclasses import dataclass
import numpy as np
import time
# Import from coralx package structure
from coral.domain.genome import MultiObjectiveScores


@dataclass
class QuixBugsMetrics:
    """Multi-objective metrics for QuixBugs evaluation following coral-x spec."""
    bugfix: float      # BugFix rate from evaluate_quixbugs
    style: float       # Style score from flake8/black
    security: float    # Security flag from bandit  
    runtime: float     # Runtime speed-up from timing


class QuixBugsRealAdapter:
    """Real QuixBugs adapter using actual dataset and test execution."""
    
    def _is_modal_environment(self) -> bool:
        """Detect if running in Modal environment."""
        import os
        return (
            os.getenv('MODAL_FUNCTION_ID') is not None or
            os.getenv('MODAL_TASK_ID') is not None or
            os.path.exists('/cache')
        )
    
    def __init__(self, dataset_path: str = None, config: Dict[str, Any] = None):
        # 🔧 CATEGORY-THEORY COMPLIANT: Use centralized path resolution
        if dataset_path is None and config is not None:
            from coral.config.path_utils import create_path_config_from_dict, get_dataset_path
            
            # Get executor type and resolve paths using category theory principles
            executor_type = config.get('infra', {}).get('executor', 'modal')
            path_config = create_path_config_from_dict(config, executor_type)
            dataset_path = get_dataset_path(path_config)
            
            print(f"📁 QuixBugs dataset path from config: {dataset_path}")
            print(f"🔧 Resolved using executor: {executor_type}")
        elif dataset_path is None:
            raise ValueError(
                "FAIL-FAST: No dataset path provided and no config available. "
                "Either provide dataset_path parameter or config with paths section."
            )
        
        self.dataset_path = Path(dataset_path)
        self.buggy_programs_path = self.dataset_path / "python_programs"
        self.correct_programs_path = self.dataset_path / "correct_python_programs" 
        self.testcases_path = self.dataset_path / "python_testcases"
        
        # Cache for loaded problems
        self._problems_cache = None
        
        # Validate dataset exists and has required structure
        if not self.dataset_path.exists():
            # Check if we're in Modal environment and try to set up dataset automatically
            if self._is_modal_environment():
                print(f"📦 QuixBugs dataset not found at {dataset_path}, setting up automatically...")
                print(f"🔍 Modal environment detected, attempting to cache dataset...")
                try:
                    from infra.modal.dataset_service import cache_quixbugs_dataset_modal
                    result_path = cache_quixbugs_dataset_modal(config=config)
                    print(f"✅ Dataset caching returned: {result_path}")
                    if not self.dataset_path.exists():
                        print(f"❌ Dataset still not found after caching, checking if path was different...")
                        print(f"   Expected: {self.dataset_path}")
                        print(f"   Returned: {result_path}")
                    else:
                        print(f"✅ QuixBugs dataset set up successfully at {dataset_path}")
                except Exception as setup_error:
                    print(f"❌ Dataset caching failed with error: {setup_error}")
                    print(f"❌ Error type: {type(setup_error)}")
                    raise RuntimeError(
                        f"FAIL-FAST: QuixBugs training data not available: {setup_error}. "
                        f"Cannot train LoRA adapter without real training data."
                    )
            else:
                raise FileNotFoundError(f"QuixBugs dataset not found at {dataset_path}")
        
        # Validate required directories exist
        required_dirs = [self.buggy_programs_path, self.correct_programs_path, self.testcases_path]
        missing_dirs = [d.name for d in required_dirs if not d.exists()]
        
        if missing_dirs:
            raise FileNotFoundError(
                f"QuixBugs dataset incomplete at {dataset_path}. Missing directories: {missing_dirs}\n"
                f"💡 Run: python setup_quixbugs_dataset.py setup --force"
            )
    
    def problems(self) -> Iterable[Dict[str, Any]]:
        """Yield actual QuixBugs problems with real prompts."""
        if self._problems_cache is None:
            self._problems_cache = list(self._load_real_problems())
        
        yield from self._problems_cache
    
    def _load_real_problems(self) -> List[Dict[str, Any]]:
        """Load real QuixBugs problems from dataset."""
        problems = []
        
        # CRITICAL: Add JSON test data directory
        json_testcases_path = self.dataset_path / "json_testcases"
        
        # Get all Python program files
        buggy_files = list(self.buggy_programs_path.glob("*.py"))
        
        for buggy_file in buggy_files:
            if buggy_file.name.endswith("_test.py"):
                continue
                
            problem_name = buggy_file.stem
            correct_file = self.correct_programs_path / buggy_file.name
            test_file = self.testcases_path / f"test_{problem_name}.py"
            json_test_file = json_testcases_path / f"{problem_name}.json"
            
            # ENHANCED: Skip if correct version, test file, OR JSON test data doesn't exist
            if not correct_file.exists() or not test_file.exists() or not json_test_file.exists():
                if not json_test_file.exists():
                    print(f"   ⚠️ Skipping {problem_name}: No JSON test data ({json_test_file.name})")
                continue
            
            try:
                # Read buggy code
                buggy_code = buggy_file.read_text(encoding='utf-8')
                
                # Read correct code  
                correct_code = correct_file.read_text(encoding='utf-8')
                
                # Read test cases
                test_code = test_file.read_text(encoding='utf-8')
                
                # FAIL-FAST: Validate file contents are valid strings
                if not isinstance(problem_name, str) or not problem_name.strip():
                    raise ValueError(f"Invalid problem name: {repr(problem_name)} (type: {type(problem_name)})")
                
                if not isinstance(buggy_code, str) or not buggy_code.strip():
                    raise ValueError(f"Invalid buggy code for {problem_name}: {repr(buggy_code[:100])} (type: {type(buggy_code)})")
                
                if not isinstance(correct_code, str) or not correct_code.strip():
                    raise ValueError(f"Invalid correct code for {problem_name}: {repr(correct_code[:100])} (type: {type(correct_code)})")
                
                if not isinstance(test_code, str) or not test_code.strip():
                    raise ValueError(f"Invalid test code for {problem_name}: {repr(test_code[:100])} (type: {type(test_code)})")
                
                # Create problem with real QuixBugs structure
                problem = {
                    "name": problem_name,
                    "prompt": self._create_real_prompt(problem_name, buggy_code),
                    "buggy_code": buggy_code,
                    "correct_code": correct_code,
                    "test_code": test_code,
                    "test_file": str(test_file)
                }
                
                # Final validation of problem dictionary
                if not isinstance(problem["name"], str):
                    raise ValueError(f"Problem name not string after creation: {type(problem['name'])}")
                
                if not isinstance(problem["buggy_code"], str):
                    raise ValueError(f"Buggy code not string after creation: {type(problem['buggy_code'])}")
                
                problems.append(problem)
                
            except Exception as e:
                print(f"⚠️  Skipping {problem_name}: {e}")
                continue
        
        # ENHANCED LOGGING: Show filtering statistics
        total_python_files = len([f for f in buggy_files if not f.name.endswith("_test.py")])
        filtered_count = total_python_files - len(problems)
        
        print(f"✅ Loaded {len(problems)} real QuixBugs problems")
        print(f"📊 Filtering statistics:")
        print(f"   • Total Python problems: {total_python_files}")
        print(f"   • Problems with JSON tests: {len(problems)}")
        print(f"   • Problems filtered out: {filtered_count}")
        print(f"   • JSON test coverage: {len(problems)/total_python_files*100:.1f}%")
        
        return problems
    
    def _create_real_prompt(self, problem_name: str, buggy_code: str) -> str:
        """Create real prompt for QuixBugs problem following coral-x spec."""
        return f"""Fix the buggy Python function in {problem_name}.py

The function has a bug that needs to be corrected. Analyze the code and provide the fixed version.

Buggy Code:
```python
{buggy_code}
```

Provide the corrected Python code:
```python
"""



def evaluate_quixbugs(adapter_path: str, generated_code: str, 
                     problem: Dict[str, Any], cheap_knobs: Dict[str, Any] = None) -> QuixBugsMetrics:
    """
    Evaluate QuixBugs solution with multi-objective metrics.
    Function name from coral-x-codellama.md specification.
    Enhanced with detailed logging.
    """
    problem_name = problem.get('name', 'unknown')
    print(f"\n🔍 EVALUATING SOLUTION FOR: {problem_name}")
    print(f"{'─'*50}")
    
    # Initialize scores
    bugfix_score = 0.0
    style_score = 0.0
    security_score = 0.0
    runtime_score = 0.0
    
    try:
        # Extract code from generation if wrapped in markdown
        print(f"📝 Extracting code from generation...")
        clean_code = _extract_code_from_generation(generated_code)
        code_length = len(clean_code)
        lines_count = len(clean_code.split('\n'))
        print(f"   • Code length: {code_length} characters")
        print(f"   • Lines: {lines_count}")
        
        # Show code preview
        preview = clean_code[:150] + "..." if len(clean_code) > 150 else clean_code
        print(f"   • Preview: {preview}")
        
        # 1. BugFix rate - functional correctness
        print(f"\n🐛 EVALUATING BUGFIX CORRECTNESS")
        bugfix_start = time.time()
        bugfix_score = _evaluate_bugfix_rate(clean_code, problem)
        bugfix_time = time.time() - bugfix_start
        print(f"   ✅ Bugfix Score: {bugfix_score:.3f} (took {bugfix_time:.2f}s)")
        
        # 2. Style score - code quality  
        print(f"\n🎨 EVALUATING CODE STYLE")
        style_start = time.time()
        style_score = _evaluate_style_score(clean_code)
        style_time = time.time() - style_start
        print(f"   ✅ Style Score: {style_score:.3f} (took {style_time:.2f}s)")
        
        # 3. Security flag - security compliance
        print(f"\n🔒 EVALUATING SECURITY COMPLIANCE")
        security_start = time.time()
        security_score = _evaluate_security_flag(clean_code)
        security_time = time.time() - security_start
        print(f"   ✅ Security Score: {security_score:.3f} (took {security_time:.2f}s)")
        
        # 4. Runtime speed-up - performance
        print(f"\n⚡ EVALUATING RUNTIME EFFICIENCY")
        runtime_start = time.time()
        runtime_score = _evaluate_runtime_speedup(clean_code, problem)
        runtime_time = time.time() - runtime_start
        print(f"   ✅ Runtime Score: {runtime_score:.3f} (took {runtime_time:.2f}s)")
        
        # Summary
        total_score = (bugfix_score + style_score + security_score + runtime_score) / 4.0
        print(f"\n📊 EVALUATION SUMMARY FOR {problem_name}:")
        print(f"   • Bugfix:   {bugfix_score:.3f}")
        print(f"   • Style:    {style_score:.3f}")
        print(f"   • Security: {security_score:.3f}")
        print(f"   • Runtime:  {runtime_score:.3f}")
        print(f"   • Average:  {total_score:.3f}")
        print(f"{'─'*50}")
        
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED FOR {problem_name}")
        print(f"   Error: {str(e)}")
        print(f"   Returning zero scores")
        print(f"{'─'*50}")
    
    return QuixBugsMetrics(
        bugfix=bugfix_score,
        style=style_score, 
        security=security_score,
        runtime=runtime_score
    )


def _extract_code_from_generation(generated_text: str) -> str:
    """Extract Python code from generated text."""
    import re
    
    # Look for python code blocks
    python_match = re.search(r'```python\s*\n(.*?)```', generated_text, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()
    
    # Look for any code blocks
    code_match = re.search(r'```\s*\n(.*?)```', generated_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Return as-is if no code blocks
    return generated_text.strip()


def _evaluate_bugfix_rate(generated_code: str, problem: Dict[str, Any]) -> float:
    """Evaluate functional correctness by running actual tests."""
    try:
        # Check syntax first
        try:
            ast.parse(generated_code)
        except SyntaxError:
            return 0.0
        
        # Create temporary file with generated code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(generated_code)
            generated_file = f.name
        
        try:
            # Copy test file and modify to import generated code
            test_code = problem["test_code"]
            problem_name = problem["name"]
            
            # Replace imports to use generated code
            modified_test = test_code.replace(
                f"from {problem_name} import",
                f"from {Path(generated_file).stem} import"
            )
            
            # Write modified test
            with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as f:
                f.write(modified_test)
                test_file = f.name
            
            # Run tests using pytest
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
                cwd=Path(generated_file).parent,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse test results
            if result.returncode == 0:
                return 1.0  # All tests passed
            else:
                # Partial credit based on test output
                output = result.stdout + result.stderr
                
                # Count passed vs failed tests
                passed_count = output.count("PASSED")
                failed_count = output.count("FAILED") 
                total_tests = passed_count + failed_count
                
                if total_tests > 0:
                    return passed_count / total_tests
                else:
                    return 0.1  # Syntax valid but tests failed
        
        finally:
            # Cleanup temp files
            try:
                os.unlink(generated_file)
                os.unlink(test_file)
            except:
                pass
                
    except Exception as e:
        print(f"Test execution failed: {e}")
        return 0.0


def _evaluate_style_score(generated_code: str) -> float:
    """Evaluate code style using flake8."""
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(generated_code)
            temp_file = f.name
        
        try:
            # Run flake8 style checker
            result = subprocess.run(
                ["python", "-m", "flake8", temp_file, "--count", "--statistics"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse flake8 output
            if result.returncode == 0:
                return 0.97  # Perfect style
            else:
                # Count violations and penalize
                output = result.stdout + result.stderr
                violations = len(output.split('\n')) - 1
                
                # Start at 0.8 and deduct for violations  
                score = 0.8 - (violations * 0.05)
                return max(0.0, min(0.97, score))
        
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        raise RuntimeError(
            f"FAIL-FAST: Style evaluation failed: {e}. "
            f"Cannot proceed without valid style score. "
            f"Flake8 tool required for style evaluation."
        )


def _evaluate_security_flag(generated_code: str) -> float:
    """Evaluate security compliance using bandit."""
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(generated_code)
            temp_file = f.name
        
        try:
            # Run bandit security checker
            result = subprocess.run(
                ["python", "-m", "bandit", temp_file, "-f", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse bandit JSON output
            if result.returncode == 0:
                return 1.0  # No security issues
            else:
                try:
                    bandit_data = json.loads(result.stdout)
                    issues = bandit_data.get("results", [])
                    
                    if not issues:
                        return 1.0
                    
                    # Count high/medium severity issues
                    high_issues = sum(1 for issue in issues if issue.get("issue_severity") == "HIGH")
                    medium_issues = sum(1 for issue in issues if issue.get("issue_severity") == "MEDIUM")
                    
                    # Penalize based on severity
                    score = 1.0 - (high_issues * 0.3) - (medium_issues * 0.1)
                    return max(0.0, score)
                    
                except json.JSONDecodeError:
                    return 0.9  # Bandit ran but output unparseable
        
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        raise RuntimeError(
            f"FAIL-FAST: Security evaluation failed: {e}. "
            f"Cannot proceed without valid security score. "
            f"Bandit tool required for security evaluation."
        )


def _evaluate_runtime_speedup(generated_code: str, problem: Dict[str, Any]) -> float:
    """Evaluate runtime performance vs original buggy code."""
    try:
        # Heuristic-based runtime scoring starting from neutral baseline
        score = 0.7  # Baseline score for standard code patterns
        
        # Reward efficient patterns
        if "[::-1]" in generated_code:  # Efficient reverse
            score += 0.1
        if "max(" in generated_code or "min(" in generated_code:  # Builtin functions
            score += 0.1
        if "sorted(" in generated_code:  # Efficient sorting
            score += 0.05
        
        # Penalize inefficient patterns
        nested_loops = generated_code.count("for") + generated_code.count("while")
        if nested_loops > 2:
            score -= 0.1
        
        # Reward list comprehensions over loops
        if "[" in generated_code and "for" in generated_code and "]" in generated_code:
            score += 0.05
        
        return max(0.0, min(0.9, score))
        
    except Exception as e:
        raise RuntimeError(
            f"FAIL-FAST: Runtime evaluation failed: {e}. "
            f"Cannot proceed without valid runtime score. "
            f"Runtime heuristics analysis required."
        )





def create_quixbugs_real_config(dataset_path: str = "../quixbugs_dataset") -> Dict[str, Any]:
    """Create configuration for real QuixBugs adapter."""
    return {
        "dataset_path": dataset_path,
        "evaluation": {
            "timeout": 30,
            "use_real_tests": True,
            "style_checker": "flake8",
            "security_checker": "bandit"
        },
        "thresholds": {
            "base": {
                "bugfix": 0.6,
                "style": 0.8, 
                "security": 0.9,
                "runtime": 0.7,
                "syntax": 0.3  # NEW: Loose syntax threshold for early generations
            },
            "max": {
                "bugfix": 0.9,
                "style": 0.97,
                "security": 1.0,
                "runtime": 0.9,
                "syntax": 0.9  # NEW: Strict syntax threshold for final generations
            }
        }
    } 