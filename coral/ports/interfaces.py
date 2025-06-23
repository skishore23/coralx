###############################################################################
# These are the only "holes" the core knows about.
# Everything concrete must implement them in plugins/ or infra/
###############################################################################
from typing import Protocol, Iterable, Dict, Callable, Any
from concurrent.futures import Future


class ModelRunner(Protocol):
    """Protocol for running language models with LoRA adaptations."""
    
    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text from a prompt."""
        ...


class DatasetProvider(Protocol):
    """Protocol for providing training/evaluation datasets."""
    
    def problems(self) -> Iterable[Dict[str, Any]]:
        """Yield problem dictionaries with prompts and solutions."""
        ...


class FitnessFn(Protocol):
    """Protocol for fitness evaluation functions."""
    
    def __call__(self,
                 genome: 'Genome',
                 model: ModelRunner,
                 problems: Iterable[Dict[str, Any]]) -> float:
        """Evaluate fitness of a genome given model and problems."""
        ...


class Executor(Protocol):
    """Protocol for distributed/parallel execution."""
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a function for execution and return a Future."""
        ...


class ConfigLoader(Protocol):
    """Protocol for loading and parsing configurations."""
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration from file path."""
        ...


class ReportRenderer(Protocol):
    """Protocol for rendering experiment reports."""
    
    def render(self, report: 'BenchmarkReport', output_dir: str) -> None:
        """Render report to output directory."""
        ... 