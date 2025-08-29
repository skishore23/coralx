"""Exception hierarchy for CORAL-X."""

from typing import Optional, Any, Dict


class CoralError(Exception):
    """Base exception for all CORAL-X errors.
    
    This is the root exception class that all other CORAL-X specific
    exceptions should inherit from. It provides structured error information
    and supports additional context.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
    
    def __str__(self) -> str:
        result = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            result += f" (context: {context_str})"
        if self.cause:
            result += f" (caused by: {self.cause})"
        return result


class ConfigurationError(CoralError):
    """Configuration validation or loading errors.
    
    Raised when there are issues with configuration files, validation,
    or configuration-related setup.
    """
    pass


class EvolutionError(CoralError):
    """Evolution process errors.
    
    Raised when there are issues during the evolutionary algorithm execution,
    genetic operations, or population management.
    """
    pass


class InfrastructureError(CoralError):
    """Infrastructure/executor errors.
    
    Raised when there are issues with external services, executors,
    caching, or other infrastructure concerns.
    """
    pass


class ValidationError(CoralError):
    """Data validation errors.
    
    Raised when input data, configurations, or other values fail
    validation checks.
    """
    pass


class CacheError(InfrastructureError):
    """Cache-related errors.
    
    Raised when there are issues with caching operations,
    cache corruption, or cache access problems.
    """
    pass


class ExecutorError(InfrastructureError):
    """Executor-related errors.
    
    Raised when there are issues with job execution, remote execution,
    or executor communication.
    """
    pass


class ModelError(EvolutionError):
    """Model-related errors.
    
    Raised when there are issues with model loading, training,
    or inference operations.
    """
    pass


class GeneticOperationError(EvolutionError):
    """Genetic operation errors.
    
    Raised when there are issues with crossover, mutation,
    selection, or other genetic operations.
    """
    pass