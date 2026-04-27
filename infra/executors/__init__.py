"""Executor implementations for CORAL-X."""

from .base import BaseExecutor, ExecutionResult
from .local import LocalExecutor

__all__ = ["BaseExecutor", "LocalExecutor", "ExecutionResult"]
