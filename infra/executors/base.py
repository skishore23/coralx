"""Base executor interface for CORAL-X."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from enum import Enum


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
    """Result of an execution operation.
    
    Attributes:
        status: Execution status
        result: The actual result data (if successful)
        error: Error message (if failed)
        execution_time: Time taken for execution in seconds
        metadata: Additional metadata about the execution
    """
    status: ExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status in (ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT, ExecutionStatus.CANCELLED)


class BaseExecutor(ABC):
    """Abstract base class for all executors.
    
    Executors are responsible for running functions in different environments
    (local, remote, distributed) with proper error handling and timeout management.
    """

    @abstractmethod
    def submit(self,
               func: Callable[..., Any],
               *args: Any,
               timeout: Optional[float] = None,
               **kwargs: Any) -> ExecutionResult:
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
    def submit_batch(self,
                    tasks: list[tuple[Callable[..., Any], tuple, dict]],
                    timeout: Optional[float] = None) -> list[ExecutionResult]:
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
