"""Structured logging framework for CORAL-X."""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def setup_logging(log_level: str = "INFO", structured: bool = False) -> None:
    """Setup structured logging for CORAL-X.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured logging (if available)
    """
    log_level = getattr(logging, log_level.upper())
    
    if STRUCTLOG_AVAILABLE and structured:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> Any:
    """Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance (structlog if available, stdlib otherwise)
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


class LoggingMixin:
    """Mixin class to add logging capability to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a structured log entry for function calls.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Additional context to include
        
    Returns:
        Dictionary with structured log data
    """
    return {
        "event": "function_call",
        "function": func_name,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }


def log_error(error: Exception, context: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a structured log entry for errors.
    
    Args:
        error: The exception that occurred
        context: Context where the error occurred
        **kwargs: Additional context to include
        
    Returns:
        Dictionary with structured error data
    """
    return {
        "event": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }