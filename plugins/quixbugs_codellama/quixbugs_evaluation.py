"""
Pure QuixBugs evaluation domain logic.
No side effects, no Modal dependencies, pure functions only.
"""
import ast
import json
import os
import re
import subprocess
import tempfile
import time
import sys
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResults:
    """Immutable evaluation results from QuixBugs assessment."""
    problem_name: str
    compilation_status: str
    syntax_valid: bool
    function_defined: bool
    test_cases_run: int
    test_cases_passed: int
    test_execution_time: float
    style_violations: int
    security_issues: List[str]
    bugfix: float
    style: float
    security: float
    runtime: float


@dataclass(frozen=True)
class TestCaseResult:
    """Result of executing test cases."""
    test_cases_run: int
    test_cases_passed: int
    test_execution_time: float
    test_output: str
    test_errors: List[str]


@dataclass(frozen=True)
class CodeValidationResult:
    """Result of code validation for infinite loops and basic execution."""
    is_valid: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    test_result: Optional[Any] = None


@dataclass(frozen=True)
class TestResults:
    """Test execution results with comprehensive debugging info."""
    syntax_valid: bool
    function_defined: bool
    tests_executed: int
    tests_passed: int
    pass_rate: float
    execution_time: float
    error_message: Optional[str] = None
    return_code: Optional[int] = None


@dataclass(frozen=True)
class EvaluationResult:
    """Modern evaluation result format for enhanced debugging."""
    problem_name: str
    code: str
    test_results: TestResults


@dataclass(frozen=True)
class ValidationResult:
    """Enhanced validation result."""
    is_valid: bool
    error: Optional[str] = None
    execution_time: float = 0.0


def extract_code_from_generation(generated_text: str) -> str:
    """Extract Python code from generated text (pure function)."""
    # Look for python code blocks
    python_match = re.search(r'```python\s*\n(.*?)```', generated_text, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()
    
    # Look for any code blocks
    code_match = re.search(r'```\s*\n(.*?)```', generated_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no markdown blocks, try to extract function definition
    lines = generated_text.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
            code_lines.append(line)
        elif in_function:
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # End of function
                break
            code_lines.append(line)
        elif 'def ' in line:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Return as-is if no code structure found
    return generated_text.strip()


def validate_syntax(code: str) -> tuple[bool, Optional[str], List[str]]:
    """Validate Python syntax and return function definitions (pure function)."""
    try:
        parsed_ast = ast.parse(code)
        function_defs = [node.name for node in ast.walk(parsed_ast) if isinstance(node, ast.FunctionDef)]
        return True, None, function_defs
    except SyntaxError as e:
        error_msg = f"Line {e.lineno}, Column {e.offset}: {e.msg}"
        return False, error_msg, []


def generate_test_cases_for_problem(problem_name: str) -> str:
    """
    Generate test cases for QuixBugs problems.
    NO HARDCODED TEST CASES - must load from real QuixBugs dataset.
    """
    # No hardcoded test cases allowed
    # This function should receive test cases from external QuixBugs dataset
    raise NotImplementedError(
        f"NO HARDCODED TEST CASES: Must load real test cases for '{problem_name}' "
        f"from QuixBugs dataset. Hardcoded test cases not allowed."
    )


def execute_test_cases(clean_code: str, problem_name: str, test_cases: Optional[str] = None) -> TestCaseResult:
    """Execute test cases and return results (has side effects but isolated)."""
    
    # Require real test cases to be provided
    if test_cases is None:
        return TestCaseResult(
            test_cases_run=0,
            test_cases_passed=0,
            test_execution_time=0.0,
            test_output='',
            test_errors=[f"No real test cases provided for '{problem_name}'. Hardcoded fallbacks removed."]
        )
    
    print(f"🧪 Executing tests for {problem_name} ({len(test_cases)} chars of test code)")
    
    try:
        import tempfile
        import os
        import json
        
        # Create temporary directory for test setup
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create function module file
            function_file = temp_path / f"{problem_name}.py"
            function_file.write_text(clean_code)
            print(f"📝 Created function file: {function_file}")
            
            # Parse QuixBugs test format - get the actual test data
            test_data = parse_quixbugs_test_data(problem_name, test_cases)
            
            print(f"📝 Running {len(test_data)} tests directly (no pytest complexity)")
            
            # Execute tests directly in Python - no temporary files needed
            start_time = time.time()
            
            # Create function in isolated namespace
            test_namespace = {}
            
            # Execute the code to define the function
            exec(clean_code, test_namespace)
            
            # Get the function
            if problem_name not in test_namespace:
                execution_time = time.time() - start_time
                return TestCaseResult(
                    test_cases_run=0,
                    test_cases_passed=0,
                    test_execution_time=execution_time,
                    test_output=f'Function {problem_name} not found in code',
                    test_errors=[f'Function {problem_name} not defined']
                )
            
            function = test_namespace[problem_name]
            
            # Set up timeout protection for function calls
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Function execution timeout - likely infinite loop")
            
            # Run tests directly
            passed = 0
            total = len(test_data)
            output_lines = []
            
            output_lines.append(f"🧪 Running {total} tests for {problem_name}")
            
            for i, (inputs, expected) in enumerate(test_data, 1):
                output_lines.append(f"Test {i}/{total}: inputs={inputs}, expected={expected}")
                
                try:
                    # Set 5-second timeout for each test case
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(5)
                    
                    try:
                        # Call function with inputs
                        if isinstance(inputs, list):
                            if len(inputs) == 0:
                                result = function()
                            else:
                                result = function(*inputs)
                        else:
                            result = function(inputs)
                        
                        # Cancel timeout if function completed
                        signal.alarm(0)
                        
                        # Compare result with expected
                        success = False
                        if isinstance(expected, (list, tuple)) and hasattr(result, '__iter__') and not isinstance(result, str):
                            success = list(result) == list(expected)
                        elif isinstance(expected, float) and isinstance(result, (int, float)):
                            success = abs(result - expected) < 1e-10
                        else:
                            success = result == expected
                        
                        if success:
                            passed += 1
                            status = "PASS"
                        else:
                            status = f"FAIL (got {result})"
                        
                        output_lines.append(f"   Test {i}/{total}: {status}")
                        
                    except TimeoutError:
                        signal.alarm(0)  # Cancel timeout
                        output_lines.append(f"   Test {i}/{total}: TIMEOUT (5s) - likely infinite loop")
                        print(f"Test {i}/{total} TIMEOUT - infinite loop detected")
                        continue
                        
                except Exception as e:
                    signal.alarm(0)  # Ensure timeout is cancelled
                    output_lines.append(f"   Test {i}/{total}: ERROR - {e}")
                    continue
            
            execution_time = time.time() - start_time
            
            output_lines.append(f"Results: {passed}/{total} tests passed")
            
            print(f"Test execution completed using direct Python execution")
            print(f"   Return code: {0 if passed == total else 1}")
            print(f"Test results: {passed}/{total} passed")
            
            return TestCaseResult(
                test_cases_run=total,
                test_cases_passed=passed,
                test_execution_time=execution_time,
                test_output='\n'.join(output_lines)[:1000],  # Limit output length
                test_errors=[] if passed > 0 else ["All tests failed"]
            )
                
    except Exception as e:
        print(f"Test setup error: {e}")
        return TestCaseResult(
            test_cases_run=0,
            test_cases_passed=0,
            test_execution_time=0.0,
            test_output='',
            test_errors=[f"Test setup error: {str(e)}"]
        )


def parse_quixbugs_test_data(problem_name: str, test_cases_content: str) -> List[Tuple[Any, Any]]:
    """Parse QuixBugs test content to extract JSON test data."""
    import json
    import re
    
    # Try to find JSON test data in various formats
    test_data = []
    
    # Method 1: Look for embedded test_data variable
    if "test_data = [" in test_cases_content:
        try:
            # Extract the test_data assignment
            start = test_cases_content.find("test_data = [")
            if start != -1:
                # Find the matching closing bracket
                bracket_count = 0
                i = start + len("test_data = ")
                while i < len(test_cases_content):
                    if test_cases_content[i] == '[':
                        bracket_count += 1
                    elif test_cases_content[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            # Extract and evaluate the list
                            test_data_str = test_cases_content[start + len("test_data = "):i+1]
                            test_data = eval(test_data_str)  # Safe since we control the input
                            print(f"   Parsed {len(test_data)} test cases from embedded test_data")
                            return test_data
                    i += 1
        except Exception as e:
            print(f"   Failed to parse embedded test_data: {e}")
    
    # Method 2: Look for JSON test case file reference and load directly
    json_match = re.search(r'load_json_testcases\(([^)]+)\)', test_cases_content)
    if json_match:
        # Modal volume path only
        json_file_candidates = [
            f"/cache/quixbugs_dataset/json_testcases/{problem_name}.json"
        ]
        
        for json_file_path in json_file_candidates:
            try:
                json_path = Path(json_file_path)
                if json_path.exists():
                    print(f"Loading JSON test data from: {json_path}")
                    with open(json_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    test_case = json.loads(line)
                                    if isinstance(test_case, list) and len(test_case) == 2:
                                        inputs, expected = test_case
                                        test_data.append((inputs, expected))
                                except json.JSONDecodeError:
                                    continue
                    break
            except Exception as e:
                print(f"   Failed to load {json_file_path}: {e}")
                continue
    
    # Method 3: Look for pytest parametrize data directly in content
    if not test_data:
        print(f"   📋 No JSON found, checking for pytest parametrize data...")
        
        # Look for @pytest.mark.parametrize with test data
        param_match = re.search(r'@pytest\.mark\.parametrize.*?\[(.*?)\]', test_cases_content, re.DOTALL)
        if param_match:
            try:
                param_data = param_match.group(1)
                # Try to parse the parametrize data
                print(f"   🔄 Attempting to parse parametrize data")
                # This is complex parsing - fail fast instead of fallback
                raise NotImplementedError(f"Complex parametrize parsing not implemented for {problem_name}")
            except Exception as e:
                print(f"   Failed to parse parametrize data: {e}")
    
    # EVOLUTIONARY PRESSURE: Return empty test data instead of failing
    if not test_data:
        print(f"No test data found for '{problem_name}' - returning empty test set")
        print(f"   • This will result in low scores via evolutionary pressure")
        return []  # Return empty list instead of failing
    
    print(f"   Final test data: {len(test_data)} test cases")
    return test_data


# REMOVED: create_clean_pytest_file function
# This function was replaced with direct Python execution to eliminate
# return code 2 issues and pytest complexity. See _run_tests_directly() for the working implementation.


def analyze_code_style(clean_code: str, problem_name: str) -> tuple[float, int, List[str]]:
    """Analyze code style and return score, violations, details (pure function with side effects)."""
    style_score = 0.8
    style_violations = 0
    style_details = []
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(clean_code)
            temp_file = f.name
        
        # Try flake8 analysis
        try:
            result = subprocess.run(
                ["python", "-m", "flake8", temp_file, "--statistics"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                style_score = 0.95
            else:
                violations = result.stdout.count('\n') if result.stdout else 0
                style_violations = violations
                style_score = max(0.6, 0.95 - violations * 0.05)
                
                if result.stdout:
                    style_details.append(result.stdout.strip()[:300])
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(f"flake8 not available for '{problem_name}'")
            
        Path(temp_file).unlink(missing_ok=True)
        
    except Exception as e:
        style_score = 0.7
        style_details.append(f"Style analysis error: {str(e)}")
    
    return style_score, style_violations, style_details


def analyze_code_security(clean_code: str) -> tuple[float, List[str]]:
    """Analyze code security and return score and issues (pure function)."""
    security_score = 0.9
    security_issues = []
    
    dangerous_patterns = {
        'eval(': 'Use of eval() function',
        'exec(': 'Use of exec() function',
        '__import__': 'Dynamic imports',
        'open(': 'File operations',
        'subprocess': 'System command execution',
        'os.system': 'Direct system calls'
    }
    
    for pattern, description in dangerous_patterns.items():
        if pattern in clean_code:
            security_issues.append(description)
            security_score -= 0.15
    
    security_score = max(0.0, security_score)
    return security_score, security_issues


def analyze_code_performance(clean_code: str, problem_name: str) -> tuple[float, List[str]]:
    """Analyze code performance and return score and notes (pure function)."""
    runtime_score = 0.7
    performance_notes = []
    
    # Check for efficient patterns
    if any(pattern in clean_code for pattern in ['set(', 'dict(', 'deque']):
        runtime_score += 0.1
        performance_notes.append("Uses efficient data structures")
    
    # Check for potentially inefficient patterns
    nested_loops = clean_code.count('for ') + clean_code.count('while ')
    if nested_loops > 2:
        runtime_score -= 0.2
        performance_notes.append(f"Multiple loops detected ({nested_loops})")
    
    # Problem-specific performance analysis
    if problem_name in ['quicksort', 'mergesort'] and 'recursion' in clean_code.lower():
        performance_notes.append("Recursive implementation detected")
    
    runtime_score = max(0.2, min(0.9, runtime_score))
    return runtime_score, performance_notes


def calculate_comprehensive_scores(evaluation_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate final multi-objective scores based on comprehensive evaluation.
    EMPHASIS: Reward actual bug fixing, not just code reproduction.
    """
    
    # ENHANCED BUGFIX SCORE: Heavy emphasis on functional correctness
    bugfix_score = 0.0
    
    # Base score for function existence (but low - not enough to just define function)
    if evaluation_data.get('function_defined', False):
        bugfix_score += 0.2  # Reduced from 0.4
    
    # MAJOR REWARD: Actual test passing (the real measure of bug fixing)
    test_cases_run = evaluation_data.get('test_cases_run', 0)
    test_cases_passed = evaluation_data.get('test_cases_passed', 0)
    
    if test_cases_run > 0:
        pass_rate = test_cases_passed / test_cases_run
        
        # Enhanced scoring for actual bug fixes
        if pass_rate == 1.0:
            # Perfect score - ALL tests pass (actual bug fix!)
            bugfix_score += 0.8
        elif pass_rate >= 0.8:
            # Very good - most tests pass
            bugfix_score += 0.6 + (pass_rate - 0.8) * 1.0  # Scale up high performance
        elif pass_rate >= 0.5:
            # Partial fix - some tests pass
            bugfix_score += 0.4 * pass_rate
        else:
            # Poor performance - minimal credit
            bugfix_score += 0.1 * pass_rate
    else:
        # NO TESTS RUN: Major penalty (likely infinite loop or broken code)
        bugfix_score = max(0.0, bugfix_score - 0.3)
    
    # STYLE SCORE: Aligned with bug fixing quality
    style_score = evaluation_data.get('style_score', 0.8)
    
    # Bonus style points for clean bug fixes
    if bugfix_score >= 0.8:
        style_score = min(1.0, style_score + 0.1)  # Reward good bug fixes with style bonus
    
    # SECURITY SCORE: Enhanced for safe bug fixes
    security_score = evaluation_data.get('security_score', 0.9)
    
    # RUNTIME SCORE: Reward efficient bug fixes over inefficient ones
    runtime_score = evaluation_data.get('runtime_score', 0.7)
    
    # Bonus for efficient bug fixes
    if bugfix_score >= 0.8 and runtime_score >= 0.8:
        runtime_score = min(1.0, runtime_score + 0.1)  # Reward both correct AND efficient
    
    return {
        'bugfix': min(1.0, bugfix_score),     # Primary objective - actual bug fixing
        'style': min(1.0, style_score),      # Secondary - clean fixes
        'security': min(1.0, security_score), # Important - safe fixes
        'runtime': min(1.0, runtime_score)   # Efficiency - fast fixes
    }


def validate_code_execution(clean_code: str, function_name: str, timeout: int = 5) -> CodeValidationResult:
    """
    Validate that generated code doesn't have infinite loops or critical bugs.
    Returns validation result with timing information.
    """
    start_time = time.time()
    
    try:
        # First check: syntax validation
        try:
            ast.parse(clean_code)
        except SyntaxError as e:
            return CodeValidationResult(
                is_valid=False, 
                error_message=f"Syntax error: {e}",
                execution_time=time.time() - start_time
            )
        
        # Second check: function definition exists
        if f"def {function_name}(" not in clean_code:
            return CodeValidationResult(
                is_valid=False,
                error_message=f"Function {function_name} not defined",
                execution_time=time.time() - start_time
            )
        
        # Third check: execution with timeout
        exec_globals = {}
        try:
            exec(clean_code, exec_globals)
        except Exception as e:
            return CodeValidationResult(
                is_valid=False,
                error_message=f"Import/definition error: {e}",
                execution_time=time.time() - start_time
            )
        
        if function_name not in exec_globals:
            return CodeValidationResult(
                is_valid=False,
                error_message=f"Function {function_name} not found after execution",
                execution_time=time.time() - start_time
            )
        
        # No hardcoded validation test cases
        # Basic validation stops here - real test cases must come from QuixBugs dataset
        print(f"   Basic validation passed: syntax valid, function defined, imports successful")
        
        return CodeValidationResult(
            is_valid=True,
            execution_time=time.time() - start_time,
            test_result="Basic validation passed - no hardcoded test execution"
        )
        
    except Exception as e:
        return CodeValidationResult(
            is_valid=False,
            error_message=f"Validation failed: {e}",
            execution_time=time.time() - start_time
        )


def evaluate_quixbugs_code(generated_code: str, problem: Dict[str, Any], test_cases: Optional[str] = None) -> EvaluationResults:
    """
    Main evaluation function for QuixBugs problems.
    Orchestrates all evaluation steps and returns comprehensive results.
    """
    problem_name = problem.get('name', 'unknown')
    
    print(f"\nEVALUATING: {problem_name}")
    print(f"{'─'*50}")
    
    # Step 1: Extract and analyze generated code
    clean_code = extract_code_from_generation(generated_code)
    print(f"📝 Code extracted ({len(clean_code)} chars)")
    
    # Step 2: CRITICAL - Validate for infinite loops and execution issues
    print(f"🔒 VALIDATING CODE EXECUTION (timeout protection)...")
    validation_result = validate_code_execution(clean_code, problem_name)
    
    if not validation_result.is_valid:
        print(f"CODE VALIDATION FAILED: {validation_result.error_message}")
        print(f"⏱️  Validation time: {validation_result.execution_time:.3f}s")
        print(f"🚫 EARLY TERMINATION - Skipping tests to prevent hangs")
        
        # Return early with low scores for failed validation (prevents infinite loops)
        return EvaluationResults(
            problem_name=problem_name,
            compilation_status=f"Validation failed: {validation_result.error_message}",
            syntax_valid=False,
            function_defined=False,
            test_cases_run=0,
            test_cases_passed=0,
            test_execution_time=validation_result.execution_time,
            style_violations=10,  # High penalty for invalid code
            security_issues=["Code validation failed"],
            bugfix=0.0,
            style=0.1,
            security=0.1,
            runtime=0.1
        )
    else:
        print(f"✅ CODE VALIDATION PASSED ({validation_result.execution_time:.3f}s)")
    
    # Step 3: Syntax validation
    syntax_valid, syntax_error, function_defs = validate_syntax(clean_code)
    
    if not syntax_valid:
        return EvaluationResults(
            problem_name=problem_name,
            compilation_status='syntax_error',
            syntax_valid=False,
            function_defined=False,
            test_cases_run=0,
            test_cases_passed=0,
            test_execution_time=0.0,
            style_violations=0,
            security_issues=[f"Syntax error: {syntax_error}"],
            bugfix=0.0,
            style=0.0,
            security=0.0,
            runtime=0.0
        )
    
    # Check if required function is defined
    function_defined = problem_name in function_defs
    
    # Step 4: Execute test cases (requires real test cases)
    test_result = execute_test_cases(clean_code, problem_name, test_cases)
    
    # Step 5: Style analysis
    style_score, style_violations, style_details = analyze_code_style(clean_code, problem_name)
    
    # Step 6: Security analysis
    security_score, security_issues = analyze_code_security(clean_code)
    
    # Step 7: Performance analysis
    runtime_score, performance_notes = analyze_code_performance(clean_code, problem_name)
    
    # Step 8: Calculate comprehensive scores
    evaluation_data = {
        'function_defined': function_defined,
        'test_cases_run': test_result.test_cases_run,
        'test_cases_passed': test_result.test_cases_passed,
        'style_score': style_score,
        'security_score': security_score,
        'runtime_score': runtime_score
    }
    
    final_scores = calculate_comprehensive_scores(evaluation_data)
    
    return EvaluationResults(
        problem_name=problem_name,
        compilation_status='success',
        syntax_valid=True,
        function_defined=function_defined,
        test_cases_run=test_result.test_cases_run,
        test_cases_passed=test_result.test_cases_passed,
        test_execution_time=test_result.test_execution_time,
        style_violations=style_violations,
        security_issues=security_issues,
        bugfix=final_scores['bugfix'],
        style=final_scores['style'],
        security=final_scores['security'],
        runtime=final_scores['runtime']
    )


def evaluate_quixbugs_code_with_simple_emergent_tracking(
    generated_code: str, 
    problem: Dict[str, Any], 
    test_cases: Optional[str] = None,
    # Simple parameters for emergent behavior detection
    genome_id: Optional[str] = None,
    generation: Optional[int] = None,
    ca_features: Optional[Dict[str, Any]] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    simple_tracker: Optional[Any] = None  # SimpleEmergentTracker
) -> EvaluationResults:
    """
    Enhanced evaluation function with simple emergent behavior tracking.
    
    This is the simplified integration point for emergent behavior tracking during LoRA testing.
    """
    # Run standard evaluation first
    evaluation_result = evaluate_quixbugs_code(generated_code, problem, test_cases)
    
    # Add simple emergent behavior tracking if all parameters are provided
    if all(x is not None for x in [genome_id, generation, ca_features, lora_config, simple_tracker]):
        try:
            # Convert evaluation result to dict for behavior detection
            evaluation_dict = {
                'bugfix': evaluation_result.bugfix,
                'style': evaluation_result.style,
                'security': evaluation_result.security,
                'runtime': evaluation_result.runtime,
                'test_cases_passed': evaluation_result.test_cases_passed,
                'test_cases_run': evaluation_result.test_cases_run,
                'test_execution_time': evaluation_result.test_execution_time
            }
            
            # Track behaviors with simple tracker
            behaviors = simple_tracker.track_evaluation(
                problem_name=problem.get('name', 'unknown'),
                genome_id=genome_id,
                generation=generation,
                ca_features=ca_features,
                lora_config=lora_config,
                evaluation_result=evaluation_dict,
                generated_code=generated_code
            )
            
            # Summary if behaviors detected
            if behaviors:
                behavior_types = [b.behavior_type for b in behaviors]
                print(f"📊 Detected {len(behaviors)} emergent patterns: {', '.join(behavior_types)}")
            
        except Exception as e:
            print(f"⚠️  Simple emergent behavior tracking failed: {e}")
            # Continue with standard evaluation - don't fail the entire evaluation
    
    return evaluation_result


def evaluate_code(problem_name: str, code: str, test_data: str, config: Dict[str, Any]) -> EvaluationResult:
    """
    Evaluate generated code against test cases with comprehensive debugging.
    
    ENHANCED DEBUG MODE: Comprehensive test execution monitoring
    """
    print(f"🔍 EVALUATING: {problem_name}")
    print(f"{'─' * 50}")
    
    # Enhanced code validation with debugging
    print(f"📝 Code extracted ({len(code)} chars)")
    print(f"🔒 VALIDATING CODE EXECUTION (timeout protection)...")
    
    validation_start = time.time()
    
    try:
        validation_result = _validate_code_execution(code, problem_name)
        validation_time = time.time() - validation_start
        
        if not validation_result.is_valid:
            print(f"❌ CODE VALIDATION FAILED ({validation_time:.3f}s)")
            print(f"   • Syntax error: {validation_result.error}")
            return EvaluationResult(
                problem_name=problem_name,
                code=code,
                test_results=TestResults(
                    syntax_valid=False,
                    function_defined=False,
                    tests_executed=0,
                    tests_passed=0,
                    pass_rate=0.0,
                    execution_time=validation_time,
                    error_message=f"Validation failed: {validation_result.error}"
                )
            )
        
        print(f"   Basic validation passed: syntax valid, function defined, imports successful")
        print(f"✅ CODE VALIDATION PASSED ({validation_time:.3f}s)")
        
    except Exception as e:
        validation_time = time.time() - validation_start
        print(f"❌ CODE VALIDATION ERROR ({validation_time:.3f}s): {e}")
        return EvaluationResult(
            problem_name=problem_name,
            code=code,
            test_results=TestResults(
                syntax_valid=False,
                function_defined=False,
                tests_executed=0,
                tests_passed=0,
                pass_rate=0.0,
                execution_time=validation_time,
                error_message=f"Validation exception: {e}"
            )
        )
    
    # Enhanced test execution with comprehensive debugging
    print(f"🧪 Executing tests for {problem_name} ({len(test_data)} chars of test code)")
    
    test_start = time.time()
    try:
        # Execute tests with direct Python execution
        test_results = _execute_tests_with_debug(problem_name, code, test_data, config)
        execution_time = time.time() - test_start
        
        # Comprehensive test result debugging
        print(f"📊 Test execution completed using direct Python execution")
        print(f"   Return code: {0 if test_results.get('tests_passed', 0) == test_results.get('tests_executed', 0) else 1}")
        
        if test_results.get('stdout'):
            stdout_preview = test_results['stdout'][:200] + "..." if len(test_results['stdout']) > 200 else test_results['stdout']
            print(f"   Stdout: {stdout_preview}")
        
        if test_results.get('stderr') and test_results['stderr'].strip():
            stderr_preview = test_results['stderr'][:200] + "..." if len(test_results['stderr']) > 200 else test_results['stderr']
            print(f"   Stderr: {stderr_preview}")
        
        # Enhanced result analysis
        tests_passed = test_results.get('tests_passed', 0)
        tests_executed = test_results.get('tests_executed', 0)
        pass_rate = (tests_passed / tests_executed * 100) if tests_executed > 0 else 0.0
        
        print(f"✅ Test results: {tests_passed}/{tests_executed} passed")
        
        # Pytest return code analysis
        return_code = test_results.get('return_code', -1)
        if return_code == 2:
            print(f"⚠️  PYTEST RETURN CODE 2 DETECTED - Internal pytest error!")
            print(f"   • This indicates pytest execution problems, not just test failures")
            print(f"   • Check test file format, imports, and pytest compatibility")
            if test_results.get('stderr'):
                print(f"   • Stderr details: {test_results['stderr']}")
        elif return_code == 1:
            print(f"📊 PYTEST RETURN CODE 1 - Normal test failures")
        elif return_code == 0:
            print(f"✅ PYTEST RETURN CODE 0 - All tests passed")
        else:
            print(f"❓ PYTEST RETURN CODE {return_code} - Unexpected")
        
        print(f"   ✅ Evaluation completed in {execution_time:.2f}s")
        
        # Create detailed test results
        final_test_results = TestResults(
            syntax_valid=True,
            function_defined=True,
            tests_executed=tests_executed,
            tests_passed=tests_passed,
            pass_rate=pass_rate,
            execution_time=execution_time,
            error_message=test_results.get('stderr', '') if return_code != 0 else None,
            return_code=return_code
        )
        
        # Enhanced result summary
        print(f"   Detailed Results:")
        print(f"      • Syntax valid: {final_test_results.syntax_valid}")
        print(f"      • Function defined: {final_test_results.function_defined}")
        print(f"      • Tests executed: {final_test_results.tests_executed}")
        print(f"      • Tests passed: {final_test_results.tests_passed}")
        print(f"      • Pass rate: {final_test_results.pass_rate:.1f}% ({tests_passed}/{tests_executed})")
        
        if tests_passed == tests_executed and tests_executed > 0:
            print(f"      ALL TESTS PASSED!")
        elif tests_passed == 0 and tests_executed > 0:
            print(f"      ALL TESTS FAILED")
        else:
            print(f"      PARTIAL SUCCESS")
        
        print(f"      • Test execution time: {execution_time:.3f}s")
        
        # Style analysis (if available)
        style_violations = _count_style_violations(code)
        print(f"      • Style violations: {style_violations}")
        
        return EvaluationResult(
            problem_name=problem_name,
            code=code,
            test_results=final_test_results
        )
        
    except Exception as e:
        execution_time = time.time() - test_start
        print(f"TEST EXECUTION ERROR ({execution_time:.3f}s): {e}")
        
        return EvaluationResult(
            problem_name=problem_name,
            code=code,
            test_results=TestResults(
                syntax_valid=True,
                function_defined=True,
                tests_executed=0,
                tests_passed=0,
                pass_rate=0.0,
                execution_time=execution_time,
                error_message=f"Test execution exception: {e}"
            )
        )


def _execute_tests_with_debug(problem_name: str, code: str, test_data: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tests with direct Python execution - no pytest complexity."""
    
    # Load test data from Modal volume
    dataset_path = config.get('dataset', {}).get('path')
    if not dataset_path:
        raise RuntimeError("Dataset path not configured")
    json_file = Path(dataset_path) / 'json_testcases' / f'{problem_name}.json'
    
    print(f"Loading JSON test data from: {json_file}")
    
    try:
        # Load QuixBugs test data
        json_test_data = []
        with open(json_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Parse Python array literal: [[input], output]
                        test_case = ast.literal_eval(line)
                        json_test_data.append(test_case)
                    except Exception as parse_error:
                        print(f"   Failed to parse line: {line[:50]}... Error: {parse_error}")
                        continue
        
        print(f"   Final test data: {len(json_test_data)} test cases")
        
    except Exception as e:
        raise RuntimeError(f"Could not load test data for '{problem_name}': {e}")
    
    # Execute tests directly - no pytest complexity
    return _run_tests_directly(problem_name, code, json_test_data)


def _run_tests_directly(problem_name: str, code: str, test_cases: List[List]) -> Dict[str, Any]:
    """Run tests directly in Python - no temporary files, no pytest."""
    
    print(f"🧪 Running {len(test_cases)} tests directly for {problem_name}")
    
    # Create function in isolated namespace
    test_namespace = {}
    
    try:
        # Execute the code to define the function
        exec(code, test_namespace)
        
        # Get the function
        if problem_name not in test_namespace:
            return {
                'return_code': 1,
                'stdout': f'Function {problem_name} not found in code',
                'stderr': f'Function {problem_name} not defined',
                'tests_passed': 0,
                'tests_executed': 0
            }
        
        function = test_namespace[problem_name]
        
    except Exception as e:
        return {
            'return_code': 1,
            'stdout': f'Code execution error: {e}',
            'stderr': str(e),
            'tests_passed': 0,
            'tests_executed': 0
        }
    
    # Set up timeout protection for function calls
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Function execution timeout - likely infinite loop")
    
    # Run tests directly
    passed = 0
    total = len(test_cases)
    output_lines = []
    
    output_lines.append(f"🧪 Running {total} tests for {problem_name}")
    
    for i, test_case in enumerate(test_cases, 1):
        # QuixBugs format: [input_array, expected_output]
        inputs = test_case[0] if len(test_case) > 0 else []
        expected = test_case[1] if len(test_case) > 1 else None
        
        output_lines.append(f"Test {i}/{total}: inputs={inputs}, expected={expected}")
        
        try:
            # Set 5-second timeout for each test case
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            try:
                # Call function with inputs
                if isinstance(inputs, list):
                    if len(inputs) == 0:
                        result = function()
                    else:
                        result = function(*inputs)
                else:
                    result = function(inputs)
                
                # Cancel timeout if function completed
                signal.alarm(0)
                
                # Compare result with expected
                success = False
                if isinstance(expected, (list, tuple)) and hasattr(result, '__iter__') and not isinstance(result, str):
                    success = list(result) == list(expected)
                elif isinstance(expected, float) and isinstance(result, (int, float)):
                    success = abs(result - expected) < 1e-10
                else:
                    success = result == expected
                
                if success:
                    passed += 1
                    status = "PASS"
                else:
                    status = f"FAIL (got {result})"
                
                output_lines.append(f"   Test {i}/{total}: {status}")
                
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                output_lines.append(f"   Test {i}/{total}: TIMEOUT (5s) - likely infinite loop")
                continue
                
        except Exception as e:
            signal.alarm(0)  # Ensure timeout is cancelled
            output_lines.append(f"   Test {i}/{total}: ERROR - {e}")
            continue
    
    output_lines.append(f"Results: {passed}/{total} tests passed")
    
    # Return results in pytest-compatible format
    return {
        'return_code': 0 if passed == total else 1,  # 0 = all pass, 1 = some fail
        'stdout': '\n'.join(output_lines),
        'stderr': '',
        'tests_passed': passed,
        'tests_executed': total
    }


def _validate_code_execution(code: str, function_name: str) -> ValidationResult:
    """Enhanced code validation with comprehensive error detection."""
    start_time = time.time()
    
    try:
        # Basic syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                is_valid=False,
                error=f"Syntax error: {e}",
                execution_time=execution_time
            )
        
        # Check for function definition
        tree = ast.parse(code)
        function_found = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                function_found = True
                break
        
        if not function_found:
            execution_time = time.time() - start_time
            return ValidationResult(
                is_valid=False,
                error=f"Function '{function_name}' not found in code",
                execution_time=execution_time
            )
        
        # Test basic execution (import simulation)
        test_code = f"""
{code}

# Test function existence
if callable({function_name}):
    print("Function is callable")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=tempfile.gettempdir()
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                return ValidationResult(
                    is_valid=False,
                    error=f"Execution error: {result.stderr}",
                    execution_time=execution_time
                )
            
            return ValidationResult(
                is_valid=True,
                execution_time=execution_time
            )
            
        finally:
            try:
                Path(test_file).unlink()
            except:
                pass
        
    except Exception as e:
        execution_time = time.time() - start_time
        return ValidationResult(
            is_valid=False,
            error=f"Validation exception: {e}",
            execution_time=execution_time
        )


def _count_style_violations(code: str) -> int:
    """Simple style violation counter."""
    violations = 0
    
    lines = code.split('\n')
    for line in lines:
        # Check for common style issues
        if len(line) > 100:  # Line too long
            violations += 1
        if line.endswith(' '):  # Trailing whitespace
            violations += 1
        if '\t' in line:  # Tabs instead of spaces
            violations += 1
    
    return violations


def check_syntax_score(code: str) -> float:
    """
    Pure function to check Python syntax and return a score [0.0, 1.0].
    
    This is used as the 5th objective in multi-objective evolution:
    - Early generations: Loose syntax requirements (allow some errors)
    - Later generations: Strict syntax requirements (require perfect syntax)
    
    Args:
        code: Python code string to check
        
    Returns:
        float: Syntax score from 0.0 (completely invalid) to 1.0 (perfect syntax)
    """
    if not code or not isinstance(code, str):
        return 0.0
    
    try:
        # Remove common markdown artifacts that aren't syntax errors
        cleaned_code = code.strip()
        if cleaned_code.startswith('```python'):
            cleaned_code = cleaned_code[9:]
        if cleaned_code.startswith('```'):
            cleaned_code = cleaned_code[3:]
        if cleaned_code.endswith('```'):
            cleaned_code = cleaned_code[:-3]
        cleaned_code = cleaned_code.strip()
        
        if not cleaned_code:
            return 0.0
        
        # Try to parse the code
        import ast
        ast.parse(cleaned_code)
        
        # If parsing succeeds, check for common quality indicators
        score = 1.0
        
        # Deduct points for common syntax quality issues
        lines = cleaned_code.split('\n')
        
        # Check for basic structure
        has_function_def = any(line.strip().startswith('def ') for line in lines)
        if not has_function_def:
            score -= 0.2  # Deduct for missing function definition
        
        # Check for reasonable indentation
        indented_lines = [line for line in lines if line.strip() and line.startswith('    ')]
        if has_function_def and not indented_lines:
            score -= 0.3  # Deduct for missing function body
        
        # Check for incomplete statements (common in generation errors)
        if cleaned_code.endswith(':') or cleaned_code.endswith(','):
            score -= 0.4  # Deduct for incomplete statements
        
        # Check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for char in cleaned_code:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack.pop() != char:
                    score -= 0.3  # Deduct for unbalanced brackets
                    break
        
        if stack:  # Unclosed brackets
            score -= 0.3
        
        return max(0.0, score)
        
    except SyntaxError as e:
        # Parse failed - return partial score based on error location
        if hasattr(e, 'lineno') and e.lineno:
            # If error is later in code, give partial credit
            total_lines = len(code.split('\n'))
            error_position = e.lineno / max(total_lines, 1)
            return max(0.0, error_position * 0.5)  # Max 0.5 for partial syntax
        else:
            return 0.0
    
    except Exception:
        # Other parsing errors
        return 0.0


def evaluate_syntax_multi_objective(code: str, problem_name: str) -> float:
    """
    Evaluate syntax for multi-objective evolution.
    
    This function is designed to be used with the σ-wave threshold system:
    - Generation 0: Accepts syntax score ≥ 0.3 (very loose)
    - Generation 20: Accepts syntax score ≥ 0.6 (moderate)  
    - Generation 40: Accepts syntax score ≥ 0.9 (strict)
    
    Args:
        code: Generated Python code
        problem_name: Name of the problem (for logging)
    
    Returns:
        float: Syntax score [0.0, 1.0]
    """
    syntax_score = check_syntax_score(code)
    
    # Add problem-specific adjustments if needed
    if not code.strip():
        return 0.0
    
    # Check if the code contains the expected function name
    if problem_name and problem_name != 'unknown':
        if f'def {problem_name}(' in code:
            syntax_score = min(1.0, syntax_score + 0.1)  # Bonus for correct function name
    
    return syntax_score

# ... existing code ... 