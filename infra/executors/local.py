"""Local executor implementation for CORAL-X."""

import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import BaseExecutor, ExecutionResult, ExecutionStatus
from core.common.exceptions import ExecutorError
from core.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalExecutorConfig:
    """Configuration for local executor."""
    max_workers: int = 4
    default_timeout: float = 300.0  # 5 minutes
    enable_timeout: bool = True


class LocalExecutor(BaseExecutor):
    """Local executor that runs functions in thread pools with timeout support.
    
    This executor provides:
    - Thread-based parallel execution
    - Timeout handling using signals
    - Proper error handling and result collection
    - Resource cleanup
    """

    def __init__(self, config: Optional[LocalExecutorConfig] = None):
        """Initialize local executor.
        
        Args:
            config: Executor configuration
        """
        self.config = config or LocalExecutorConfig()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._shutdown = False
        logger.info(f"Local executor initialized with {self.config.max_workers} workers")

    def submit(self,
               func: Callable[..., Any],
               *args: Any,
               timeout: Optional[float] = None,
               **kwargs: Any) -> ExecutionResult:
        """Submit a function for local execution.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            timeout: Maximum execution time in seconds
            **kwargs: Keyword arguments for the function
            
        Returns:
            ExecutionResult with status and result/error information
        """
        if self._shutdown:
            raise ExecutorError("Executor has been shutdown")

        if not self.is_available:
            raise ExecutorError("Executor is not available")

        timeout = timeout or self.config.default_timeout
        start_time = time.time()

        try:
            logger.debug(f"Submitting function {func.__name__} for execution")

            # Use thread pool for execution
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

            # Submit to thread pool with timeout
            future = self._executor.submit(self._execute_with_timeout, func, args, kwargs, timeout)

            try:
                result = future.result(timeout=timeout + 1.0)  # Add small buffer
                execution_time = time.time() - start_time

                logger.debug(f"Function {func.__name__} completed in {execution_time:.3f}s")
                return ExecutionResult(
                    status=ExecutionStatus.COMPLETED,
                    result=result,
                    execution_time=execution_time,
                    metadata={"function": func.__name__}
                )

            except FutureTimeoutError:
                execution_time = time.time() - start_time
                logger.warning(f"Function {func.__name__} timed out after {execution_time:.3f}s")
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error=f"Function timed out after {timeout}s",
                    execution_time=execution_time,
                    metadata={"function": func.__name__, "timeout": timeout}
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={"function": func.__name__}
            )

    def submit_batch(self,
                    tasks: List[Tuple[Callable[..., Any], Tuple, Dict]],
                    timeout: Optional[float] = None) -> List[ExecutionResult]:
        """Submit multiple functions for batch execution.
        
        Args:
            tasks: List of (function, args_tuple, kwargs_dict) tuples
            timeout: Maximum execution time per task in seconds
            
        Returns:
            List of ExecutionResult objects corresponding to each task
        """
        if self._shutdown:
            raise ExecutorError("Executor has been shutdown")

        if not self.is_available:
            raise ExecutorError("Executor is not available")

        timeout = timeout or self.config.default_timeout
        logger.info(f"Submitting batch of {len(tasks)} tasks")

        results = []
        for i, (func, args, kwargs) in enumerate(tasks):
            try:
                result = self.submit(func, *args, timeout=timeout, **kwargs)
                results.append(result)
                logger.debug(f"Batch task {i+1}/{len(tasks)} completed with status {result.status}")
            except Exception as e:
                logger.error(f"Batch task {i+1}/{len(tasks)} failed: {e}")
                results.append(ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    metadata={"function": func.__name__, "task_index": i}
                ))

        successful = sum(1 for r in results if r.is_successful())
        logger.info(f"Batch execution completed: {successful}/{len(tasks)} successful")

        return results

    def _execute_with_timeout(self,
                            func: Callable[..., Any],
                            args: Tuple,
                            kwargs: Dict[str, Any],
                            timeout: float) -> Any:
        """Execute function with timeout using signal-based timeout.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Timeout in seconds
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If function exceeds timeout
        """
        if not self.config.enable_timeout:
            return func(*args, **kwargs)

        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout}s")

        # Store original signal handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)

        try:
            # Set alarm for timeout
            signal.alarm(int(timeout))

            # Execute function
            result = func(*args, **kwargs)

            # Cancel alarm
            signal.alarm(0)

            return result

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGALRM, original_handler)

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        if self._shutdown:
            return

        logger.info("Shutting down local executor")
        self._shutdown = True

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("Local executor shutdown complete")

    @property
    def is_available(self) -> bool:
        """Check if the executor is available for use."""
        return not self._shutdown and (self._executor is None or not self._executor._shutdown)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
