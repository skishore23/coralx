"""Executor implementations for CORAL-X."""

from .local import LocalExecutor
from .base import BaseExecutor, ExecutionResult

__all__ = ["BaseExecutor", "LocalExecutor", "ExecutionResult"]
