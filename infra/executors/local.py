"""Local executor implementation for CORAL-X."""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from core.common.exceptions import ExecutorError
from core.common.logging import get_logger

from .base import BaseExecutor, ExecutionResult, ExecutionStatus

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
    - Timeout status reporting via futures
    - Proper error handling and result collection
    - Resource cleanup
    """

    def __init__(self, config: LocalExecutorConfig | None = None):
        """Initialize local executor.

        Args:
            config: Executor configuration
        """
        self.config = config or LocalExecutorConfig()
        self._executor: ThreadPoolExecutor | None = None
        self._shutdown = False
        logger.info(
            f"Local executor initialized with {self.config.max_workers} workers"
        )

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
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

            future = self._executor.submit(func, *args, **kwargs)

            try:
                result_timeout = timeout + 1.0 if self.config.enable_timeout else None
                result = future.result(timeout=result_timeout)  # Add small buffer
                execution_time = time.time() - start_time

                logger.debug(
                    f"Function {func.__name__} completed in {execution_time:.3f}s"
                )
                return ExecutionResult(
                    status=ExecutionStatus.COMPLETED,
                    result=result,
                    execution_time=execution_time,
                    metadata={"function": func.__name__},
                )

            except FutureTimeoutError:
                future.cancel()
                execution_time = time.time() - start_time
                logger.warning(
                    f"Function {func.__name__} timed out after {execution_time:.3f}s"
                )
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error=f"Function timed out after {timeout}s",
                    execution_time=execution_time,
                    metadata={"function": func.__name__, "timeout": timeout},
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={"function": func.__name__},
            )

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
                logger.debug(
                    f"Batch task {i + 1}/{len(tasks)} completed with status {result.status}"
                )
            except Exception as e:
                logger.error(f"Batch task {i + 1}/{len(tasks)} failed: {e}")
                results.append(
                    ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=str(e),
                        metadata={"function": func.__name__, "task_index": i},
                    )
                )

        successful = sum(1 for r in results if r.is_successful())
        logger.info(f"Batch execution completed: {successful}/{len(tasks)} successful")

        return results

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
        return not self._shutdown and (
            self._executor is None or not self._executor._shutdown
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
