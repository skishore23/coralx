"""
QuixBugs Mini Plugin for M1 - End-to-End Tiny Run
Minimal implementation with 3 specific bugs for M1 testing
"""
from typing import Iterable, Dict, Any, Callable

# Import from clean coralx package structure
from core.ports.interfaces import DatasetProvider, ModelRunner, FitnessFn
from core.domain.mapping import LoRAConfig
from core.domain.genome import Genome, MultiObjectiveScores


# M1 Mini Dataset - 3 specific bugs for testing
QUIXBUGS_MINI_PROBLEMS = [
    {
        'name': 'gcd',
        'prompt': '''```python
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
```''',
        'expected_behavior': 'Calculate greatest common divisor using Euclidean algorithm'
    },
    {
        'name': 'is_valid_parenthesization',
        'prompt': '''```python
def is_valid_parenthesization(parens):
    if not parens:
        return True
    if parens[0] == ')':
        return False
    if parens[-1] == '(':
        return False
    return True
```''',
        'expected_behavior': 'Check if parentheses are properly balanced'
    },
    {
        'name': 'sqrt',
        'prompt': '''```python
def sqrt(x):
    if x < 0:
        return -1
    if x == 0:
        return 0
    if x == 1:
        return 1
    return x / 2
```''',
        'expected_behavior': 'Calculate square root using Newton-Raphson method'
    }
]


class QuixBugsMiniDataset(DatasetProvider):
    """Mini QuixBugs dataset provider with 3 specific bugs for M1 testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print(f"QuixBugs Mini dataset loaded: {len(QUIXBUGS_MINI_PROBLEMS)} problems")
        for problem in QUIXBUGS_MINI_PROBLEMS:
            print(f"   • {problem['name']}: {problem['expected_behavior']}")

    def problems(self) -> Iterable[Dict[str, Any]]:
        """Yield the 3 mini QuixBugs problems."""
        for problem in QUIXBUGS_MINI_PROBLEMS:
            yield problem


class CodeLlamaMiniRunner(ModelRunner):
    """Mini CodeLlama model runner for M1 testing."""

    def __init__(self, lora_cfg: LoRAConfig, config: Dict[str, Any], genome: Genome = None):
        self.lora_cfg = lora_cfg
        self.config = config
        self.genome = genome
        self._model_loaded = False
        self._adapter_path = None
        print(f"CodeLlama Mini runner initialized for genome {genome.id if genome else 'unknown'}")

    def generate(self, prompt: str, max_tokens: int = 512, cheap_knobs=None) -> str:
        """Generate code completion for mini QuixBugs problems."""
        if not self._model_loaded:
            self._setup_model()

        # For M1, use a simple mock generation that returns a fixed response
        # In real implementation, this would call the actual model
        print(f"   Generating code for prompt: {prompt[:100]}...")

        # Extract problem name from prompt
        problem_name = self._extract_problem_name(prompt)

        # Return a mock "fixed" version of the code
        mock_fixes = {
            'gcd': '''def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)''',
            'is_valid_parenthesization': '''def is_valid_parenthesization(parens):
    if not parens:
        return True
    stack = []
    for char in parens:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0''',
            'sqrt': '''def sqrt(x):
    if x < 0:
        return -1
    if x == 0:
        return 0
    if x == 1:
        return 1
    # Newton-Raphson method
    guess = x / 2
    for _ in range(10):
        guess = (guess + x / guess) / 2
    return guess'''
        }

        return mock_fixes.get(problem_name, prompt)

    def _extract_problem_name(self, prompt: str) -> str:
        """Extract problem name from prompt."""
        for problem in QUIXBUGS_MINI_PROBLEMS:
            if problem['name'] in prompt:
                return problem['name']
        return 'unknown'

    def _setup_model(self):
        """Setup model for M1 testing."""
        print("   Setting up CodeLlama Mini model...")
        print(f"   LoRA config: r={self.lora_cfg.r}, α={self.lora_cfg.alpha}, dropout={self.lora_cfg.dropout}")
        self._model_loaded = True
        print("   Model setup complete (mock implementation for M1)")


class QuixBugsMiniFitness(FitnessFn):
    """Mini fitness function for QuixBugs M1 testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print("QuixBugs Mini fitness function initialized")

    def __call__(self,
                 genome: Genome,
                 model: ModelRunner,
                 problems: Iterable[Dict[str, Any]],
                 ca_features = None) -> float:
        """Single-objective evaluation for M1 compatibility."""
        multi_scores = self.evaluate_multi_objective(genome, model, problems, ca_features)
        return multi_scores.overall_fitness()

    def evaluate_multi_objective(self,
                                genome: Genome,
                                model: ModelRunner,
                                problems: Iterable[Dict[str, Any]],
                                ca_features = None) -> MultiObjectiveScores:
        """Multi-objective evaluation for QuixBugs Mini."""

        print("\nQUIXBUGS MINI EVALUATION")
        print(f"{'='*40}")
        print(f"Genome ID: {genome.id if hasattr(genome, 'id') else 'unknown'}")

        problems_list = list(problems)
        print(f"Evaluating on {len(problems_list)} mini problems")

        # Initialize scores
        bugfix_scores = []
        style_scores = []
        security_scores = []
        runtime_scores = []
        syntax_scores = []

        for i, problem in enumerate(problems_list, 1):
            problem_name = problem['name']
            print(f"\nProblem {i}/{len(problems_list)}: {problem_name}")

            try:
                # Generate solution
                print("   Generating solution...")
                generated_code = model.generate(problem["prompt"], max_tokens=512)
                print(f"   Generated {len(generated_code)} characters")

                # Evaluate the generated code
                scores = self._evaluate_generated_code(generated_code, problem)

                bugfix_scores.append(scores['bugfix'])
                style_scores.append(scores['style'])
                security_scores.append(scores['security'])
                runtime_scores.append(scores['runtime'])
                syntax_scores.append(scores['syntax'])

                print(f"   Scores: B:{scores['bugfix']:.3f} S:{scores['style']:.3f} Sec:{scores['security']:.3f} R:{scores['runtime']:.3f} Syn:{scores['syntax']:.3f}")

            except Exception as e:
                print(f"   Evaluation failed: {e}")
                # Add zero scores for failed problems
                bugfix_scores.append(0.0)
                style_scores.append(0.0)
                security_scores.append(0.0)
                runtime_scores.append(0.0)
                syntax_scores.append(0.0)

        # Calculate averages
        avg_bugfix = sum(bugfix_scores) / max(len(bugfix_scores), 1)
        avg_style = sum(style_scores) / max(len(style_scores), 1)
        avg_security = sum(security_scores) / max(len(security_scores), 1)
        avg_runtime = sum(runtime_scores) / max(len(runtime_scores), 1)
        avg_syntax = sum(syntax_scores) / max(len(syntax_scores), 1)

        print("\nFINAL MINI SCORES")
        print(f"{'─'*40}")
        print(f"Average Scores Across {len(bugfix_scores)} Problems:")
        print(f"   • Bugfix:   {avg_bugfix:.3f}")
        print(f"   • Style:    {avg_style:.3f}")
        print(f"   • Security: {avg_security:.3f}")
        print(f"   • Runtime:  {avg_runtime:.3f}")
        print(f"   • Syntax:   {avg_syntax:.3f}")

        return MultiObjectiveScores(
            bugfix=avg_bugfix,
            style=avg_style,
            security=avg_security,
            runtime=avg_runtime,
            syntax=avg_syntax
        )

    def _evaluate_generated_code(self, code: str, problem: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate generated code across multiple objectives."""

        # Basic syntax check
        syntax_valid = self._check_syntax(code)

        # Style check (basic)
        style_score = self._check_style(code)

        # Security check (basic)
        security_score = self._check_security(code)

        # Runtime efficiency (basic)
        runtime_score = self._check_runtime_efficiency(code)

        # Bugfix score - check if the code looks like it fixes the problem
        bugfix_score = self._check_bugfix(code, problem)

        return {
            'bugfix': bugfix_score,
            'style': style_score,
            'security': security_score,
            'runtime': runtime_score,
            'syntax': 1.0 if syntax_valid else 0.0
        }

    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _check_style(self, code: str) -> float:
        """Check code style using flake8."""
        try:
            import subprocess
            import tempfile
            import os

            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run flake8
                result = subprocess.run(
                    ['flake8', '--count', '--select=E,W', temp_file],
                    capture_output=True, text=True, timeout=10
                )

                # Parse flake8 output
                error_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0

                # Convert to score (fewer errors = higher score)
                max_errors = 10  # Normalize to 10 errors = 0 score
                score = max(0.0, 1.0 - (error_count / max_errors))

            finally:
                # Clean up temp file
                os.unlink(temp_file)

        except (subprocess.TimeoutExpired, FileNotFoundError, ImportError, Exception):
            # Fallback to basic checks if flake8 not available
            score = 1.0
            lines = code.split('\n')

            # Check for proper indentation
            for line in lines:
                if line.strip() and not line.startswith((' ', '\t')):
                    if line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ')):
                        score -= 0.1

            # Check for reasonable line length
            long_lines = sum(1 for line in lines if len(line) > 100)
            if long_lines > 0:
                score -= min(0.3, long_lines * 0.1)

        return max(0.0, score)

    def _check_security(self, code: str) -> float:
        """Check for security issues using bandit."""
        try:
            import subprocess
            import tempfile
            import os
            import json

            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run bandit
                result = subprocess.run(
                    ['bandit', '-f', 'json', temp_file],
                    capture_output=True, text=True, timeout=10
                )

                # Parse bandit output
                if result.returncode == 0:
                    score = 1.0  # No security issues found
                else:
                    try:
                        bandit_output = json.loads(result.stdout)
                        issues = bandit_output.get('results', [])

                        # Count high and medium severity issues
                        high_severity = sum(1 for issue in issues if issue.get('issue_severity') == 'HIGH')
                        medium_severity = sum(1 for issue in issues if issue.get('issue_severity') == 'MEDIUM')

                        # Calculate score (penalize high severity more)
                        score = max(0.0, 1.0 - (high_severity * 0.3 + medium_severity * 0.1))

                    except (json.JSONDecodeError, KeyError):
                        score = 0.5  # Default score if parsing fails

            finally:
                # Clean up temp file
                os.unlink(temp_file)

        except (subprocess.TimeoutExpired, FileNotFoundError, ImportError, Exception):
            # Fallback to basic checks if bandit not available
            score = 1.0

            # Check for dangerous functions
            dangerous_patterns = ['eval(', 'exec(', '__import__', 'open(']
            for pattern in dangerous_patterns:
                if pattern in code:
                    score -= 0.2

        return max(0.0, score)

    def _check_runtime_efficiency(self, code: str) -> float:
        """Check runtime efficiency with actual performance measurement."""
        try:
            import time
            import subprocess
            import tempfile
            import os

            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Measure execution time
                start_time = time.time()
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True, text=True, timeout=5
                )
                execution_time = time.time() - start_time

                # Convert to score (faster = higher score)
                # Normalize: 0.1s = 1.0, 1.0s = 0.5, 5.0s = 0.0
                if execution_time <= 0.1:
                    score = 1.0
                elif execution_time <= 1.0:
                    score = 1.0 - (execution_time - 0.1) * 0.5 / 0.9
                else:
                    score = max(0.0, 0.5 - (execution_time - 1.0) * 0.5 / 4.0)

            finally:
                # Clean up temp file
                os.unlink(temp_file)

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # Fallback to basic checks if execution fails
            score = 1.0

            # Check for obvious inefficiencies
            if 'for i in range(len(' in code:
                score -= 0.1
            if 'while True:' in code and 'break' not in code:
                score -= 0.3
            if 'import numpy' in code and 'numpy' not in code:
                score -= 0.1  # Unnecessary import

        return max(0.0, score)

    def _check_bugfix(self, code: str, problem: Dict[str, Any]) -> float:
        """Check if code appears to fix the problem."""
        problem_name = problem['name']

        # Basic heuristics for each problem
        if problem_name == 'gcd':
            # Should have recursive call and modulo operation
            if 'gcd(' in code and '%' in code:
                return 0.8
            elif 'gcd(' in code:
                return 0.6
            else:
                return 0.3

        elif problem_name == 'is_valid_parenthesization':
            # Should have stack-like logic
            if 'stack' in code or 'append' in code or 'pop' in code:
                return 0.8
            elif '(' in code and ')' in code:
                return 0.6
            else:
                return 0.3

        elif problem_name == 'sqrt':
            # Should have iterative improvement
            if 'for' in code and 'guess' in code:
                return 0.8
            elif 'sqrt' in code:
                return 0.6
            else:
                return 0.3

        return 0.5  # Default score


class QuixBugsMiniPlugin:
    """Main QuixBugs Mini plugin class for M1 testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print("QuixBugs Mini plugin initialized for M1 testing")
        print(f"   Problems: {len(QUIXBUGS_MINI_PROBLEMS)} mini bugs")
        print(f"   Model: {config.get('experiment', {}).get('model', {}).get('name', 'not specified')}")

    def get_modal_config(self, coral_config) -> Dict[str, Any]:
        """Get Modal-compatible configuration."""
        return {
            'evo': self.config.get('evo', {}),
            'execution': coral_config.execution,
            'experiment': coral_config.experiment,
            'infra': coral_config.infra,
            'cache': coral_config.cache,
            'evaluation': coral_config.evaluation,
            'seed': coral_config.seed,
            'adapter_type': getattr(coral_config, 'adapter_type', 'lora'),
        }

    def dataset(self) -> DatasetProvider:
        """Create mini dataset provider."""
        return QuixBugsMiniDataset(self.config)

    def model_factory(self) -> Callable[[LoRAConfig], ModelRunner]:
        """Create model factory."""
        def create_model(lora_cfg: LoRAConfig, genome: Genome = None) -> ModelRunner:
            return CodeLlamaMiniRunner(lora_cfg, self.config, genome=genome)
        return create_model

    def fitness_fn(self) -> FitnessFn:
        """Create fitness function."""
        return QuixBugsMiniFitness(self.config)
