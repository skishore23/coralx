"""
Protocol definitions for plugin and infrastructure interfaces.

This module defines the abstract interfaces that separate core CORAL-X
logic from specific implementations. All concrete implementations must
be provided through plugins or infrastructure modules.
"""
from typing import Protocol, Iterable, Dict, Callable, Any
from concurrent.futures import Future

from core.domain.genome import Genome


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
