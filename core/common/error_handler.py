"""Error handling middleware for CORAL-X."""

from typing import Any, Callable, Optional
from .exceptions import CoralError
from .logging import get_logger


class ErrorHandler:
    """Centralized error handling with logging and recovery strategies."""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)
    
    def handle_error(self, error: Exception, context: str) -> None:
        """Handle an error with logging.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            
        Raises:
            The original error (always re-raised)
        """
        self.logger.error(f"Error occurred: error_type={type(error).__name__}, error_message={str(error)}, context={context}")
        raise error
    
    def with_error_handling(self, context: str):
        """Decorator for automatic error handling.
        
        Args:
            context: Context string for the operation
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(e, context)
            return wrapper
        return decorator