"""
Categorical Result Types - Error Handling Monads for CoralX
Pure functional error handling with compositional safety.
"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Union, Optional, Any
from abc import ABC, abstractmethod

A = TypeVar('A')
B = TypeVar('B')
E = TypeVar('E')


class Result(Generic[A, E], ABC):
    """
    Result monad for compositional error handling.
    Replaces FAIL-FAST exceptions with safer monadic composition.
    
    Mathematical Properties:
    - Monad Laws: return + bind + associativity
    - Functor Laws: fmap(id) = id, fmap(g ∘ f) = fmap(g) ∘ fmap(f)
    - Error Propagation: Automatically propagates errors through composition
    """
    
    @abstractmethod
    def is_success(self) -> bool:
        """Check if result represents success."""
        pass
    
    @abstractmethod
    def is_error(self) -> bool:
        """Check if result represents error."""
        pass
    
    def bind(self, f: Callable[[A], 'Result[B, E]']) -> 'Result[B, E]':
        """Monadic bind (>>=) - chain operations safely."""
        if self.is_success():
            return f(self.unwrap())
        else:
            return Error(self.unwrap_error())
    
    def fmap(self, f: Callable[[A], B]) -> 'Result[B, E]':
        """Functorial map - apply function to success value."""
        if self.is_success():
            try:
                return Success(f(self.unwrap()))
            except Exception as e:
                return Error(str(e))
        else:
            return Error(self.unwrap_error())
    
    def unwrap(self) -> A:
        """Extract success value (unsafe - use only when sure it's Success)."""
        if self.is_success():
            return self._value
        else:
            raise RuntimeError(f"Attempted to unwrap Error: {self._error}")
    
    def unwrap_error(self) -> E:
        """Extract error value (unsafe - use only when sure it's Error)."""
        if self.is_error():
            return self._error
        else:
            raise RuntimeError("Attempted to unwrap_error Success")
    
    def unwrap_or(self, default: A) -> A:
        """Extract value or return default."""
        return self._value if self.is_success() else default
    
    def and_then(self, f: Callable[[A], 'Result[B, E]']) -> 'Result[B, E]':
        """Alias for bind - more readable in some contexts."""
        return self.bind(f)
    
    def or_else(self, f: Callable[[E], 'Result[A, E]']) -> 'Result[A, E]':
        """Handle error case with recovery function."""
        if self.is_error():
            return f(self.unwrap_error())
        else:
            return self


@dataclass(frozen=True)
class Success(Result[A, E]):
    """Success case of Result monad."""
    _value: A
    
    def is_success(self) -> bool:
        return True
    
    def is_error(self) -> bool:
        return False


@dataclass(frozen=True)  
class Error(Result[A, E]):
    """Error case of Result monad."""
    _error: E
    
    def is_success(self) -> bool:
        return False
    
    def is_error(self) -> bool:
        return True


# Convenience functions for creating Results
def success(value: A) -> Result[A, str]:
    """Create Success result."""
    return Success(value)


def error(err: str) -> Result[Any, str]:
    """Create Error result."""
    return Error(err)


def safe_call(f: Callable[[], A], error_msg: str = None) -> Result[A, str]:
    """
    Safely call function, converting exceptions to Error results.
    Replaces try/catch with monadic error handling.
    """
    try:
        return success(f())
    except Exception as e:
        err_msg = error_msg or f"Function failed: {str(e)}"
        return error(err_msg)


def safe_call_with_args(f: Callable[..., A], *args, **kwargs) -> Result[A, str]:
    """Safely call function with arguments."""
    try:
        return success(f(*args, **kwargs))
    except Exception as e:
        return error(f"Function {f.__name__} failed: {str(e)}")


# Monadic composition helpers
def compose_results(*funcs) -> Callable[[A], Result[Any, str]]:
    """
    Compose multiple Result-returning functions.
    Forms a computational pipeline that stops on first error.
    """
    def composed(initial_value: A) -> Result[Any, str]:
        result = success(initial_value)
        for func in funcs:
            result = result.bind(func)
            if result.is_error():
                break
        return result
    
    return composed


def sequence_results(results: list[Result[A, E]]) -> Result[list[A], E]:
    """
    Convert list of Results to Result of list.
    Fails if any Result is Error, succeeds if all are Success.
    """
    successes = []
    for result in results:
        if result.is_error():
            return Error(result.unwrap_error())
        successes.append(result.unwrap())
    
    return Success(successes)


def traverse_results(items: list[A], f: Callable[[A], Result[B, E]]) -> Result[list[B], E]:
    """
    Apply function to list items, collecting Results.
    Equivalent to sequence_results(map(f, items)).
    """
    results = [f(item) for item in items]
    return sequence_results(results)


# Configuration validation with monads
class ConfigValidation:
    """Configuration validation using Result monads."""
    
    @staticmethod
    def validate_required_field(config, field: str) -> Result[Any, str]:
        """Validate that required field exists (supports dict and config objects)."""
        # Handle both dictionaries and config objects (like CoralConfig, EvolutionConfig)
        if hasattr(config, '__contains__') and hasattr(config, '__getitem__'):
            # Dict-like interface (works for both dict and CoralConfig)
            if field not in config:
                return error(f"Missing required field: '{field}'")
            return success(config[field])
        elif hasattr(config, field):
            # Dataclass-like interface (for EvolutionConfig, etc.)
            return success(getattr(config, field))
        else:
            return error(f"Missing required field: '{field}' in {type(config).__name__}")
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, field_name: str) -> Result[Any, str]:
        """Validate value type."""
        if not isinstance(value, expected_type):
            return error(f"Field '{field_name}' must be {expected_type.__name__}, got {type(value).__name__}")
        return success(value)
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: float, max_val: float, field_name: str) -> Result[Union[int, float], str]:
        """Validate numeric range."""
        if not (min_val <= value <= max_val):
            return error(f"Field '{field_name}' must be in range [{min_val}, {max_val}], got {value}")
        return success(value)
    
    @staticmethod
    def validate_choices(value: Any, choices: list, field_name: str) -> Result[Any, str]:
        """Validate value is in allowed choices."""
        if value not in choices:
            return error(f"Field '{field_name}' must be one of {choices}, got {value}")
        return success(value)


# Example usage patterns for replacing FAIL-FAST exceptions
def example_monadic_config_loading(config_path: str) -> Result[dict, str]:
    """
    Example of monadic configuration loading.
    Replaces exception-based FAIL-FAST with composable error handling.
    """
    def load_yaml(path: str) -> Result[dict, str]:
        import yaml
        from pathlib import Path
        
        if not Path(path).exists():
            return error(f"Config file not found: {path}")
        
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            return success(config) if config else error(f"Empty or invalid YAML: {path}")
        except yaml.YAMLError as e:
            return error(f"YAML parsing failed: {e}")
        except Exception as e:
            return error(f"Failed to load config: {e}")
    
    def validate_config(config: dict) -> Result[dict, str]:
        # Chain validations using monadic composition
        validation = (
            ConfigValidation.validate_required_field(config, 'evo')
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'execution'))
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'infra'))
            .bind(lambda _: success(config))
        )
        return validation
    
    # Compose the entire pipeline
    return (
        load_yaml(config_path)
        .bind(validate_config)
    ) 