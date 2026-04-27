"""Base executor interface for CORAL-X."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from core.ports.interfaces import ExecutionResult, ExecutionStatus

__all__ = ["BaseExecutor", "ExecutionResult", "ExecutionStatus"]


class BaseExecutor(ABC):
    """Abstract base class for all executors.

    Executors are responsible for running functions in different environments
    (local, remote, distributed) with proper error handling and timeout management.
    """

    @abstractmethod
    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Submit a function for execution.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            timeout: Maximum execution time in seconds (None for no timeout)
            **kwargs: Keyword arguments for the function

        Returns:
            ExecutionResult with status and result/error information
        """
        pass

    @abstractmethod
    def submit_batch(
        self,
        tasks: list[tuple[Callable[..., Any], tuple, dict]],
        timeout: float | None = None,
    ) -> list[ExecutionResult]:
        """Submit multiple functions for batch execution.

        Args:
            tasks: List of (function, args_tuple, kwargs_dict) tuples
            timeout: Maximum execution time per task in seconds

        Returns:
            List of ExecutionResult objects corresponding to each task
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the executor is available for use."""
        pass
