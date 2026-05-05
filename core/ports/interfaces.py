"""
Protocol definitions for plugin and infrastructure interfaces.

This module defines the abstract interfaces that separate core CORAL-X
logic from specific implementations. All concrete implementations must
be provided through plugins or infrastructure modules.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from core.domain.cheap_knobs import CheapKnobs
from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.mapping import AdapterConfig


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ExecutionResult:
    """Result of an execution operation."""

    status: ExecutionStatus
    result: Any | None = None
    error: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] | None = None

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status in (
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED,
        )


class ModelRunner(Protocol):
    """Protocol for running target models with candidate parameters."""

    def generate(
        self, prompt: str, max_tokens: int, cheap_knobs: CheapKnobs | None = None
    ) -> str:
        """Generate text from a prompt."""
        ...


class DatasetProvider(Protocol):
    """Protocol for providing training/evaluation datasets."""

    def problems(self) -> Iterable[dict[str, Any]]:
        """Yield problem dictionaries with prompts and solutions."""
        ...


class FitnessFn(Protocol):
    """Protocol for fitness evaluation functions."""

    def __call__(
        self, genome: "Genome", model: ModelRunner, problems: Iterable[dict[str, Any]]
    ) -> float:
        """Evaluate fitness of a genome given model and problems."""
        ...

    def evaluate_multi_objective(
        self,
        genome: Genome,
        model: ModelRunner,
        problems: Iterable[dict[str, Any]],
        ca_features: Any | None = None,
    ) -> MultiObjectiveScores:
        """Evaluate all objectives for a genome."""
        ...


class Executor(Protocol):
    """Protocol for distributed/parallel execution."""

    def submit(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ExecutionResult:
        """Submit a function for execution and return an execution result."""
        ...


class Plugin(Protocol):
    """Protocol implemented by experiment plugins."""

    def dataset(self) -> DatasetProvider:
        """Create the dataset provider."""
        ...

    def model_factory(self) -> Callable[[AdapterConfig, Genome | None], ModelRunner]:
        """Create a model factory."""
        ...

    def fitness_fn(self) -> FitnessFn:
        """Create the fitness function."""
        ...


class ConfigLoader(Protocol):
    """Protocol for loading and parsing configurations."""

    def load(self, path: str) -> dict[str, Any]:
        """Load configuration from file path."""
        ...
