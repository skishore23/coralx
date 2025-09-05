"""Configuration loader using Pydantic models."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import ValidationError

from .config import CoralConfig
from .exceptions import ConfigurationError
from .logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Centralized configuration management with environment support."""

    def __init__(self, environment: str = "development"):
        self.environment = environment
        logger.info(f"Config manager initialized for {environment} environment")

    def load_config(self, config_path: Optional[Path] = None) -> CoralConfig:
        """Load and validate configuration from multiple sources.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated CoralConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid or cannot be loaded
        """
        try:
            # Load base configuration
            base_config = self._load_base_config(config_path)
            logger.info(f"Base config loaded from {config_path}")

            # Apply environment overrides
            env_overrides = self._load_environment_overrides()
            merged_config = self._merge_configs(base_config, env_overrides)

            if env_overrides:
                logger.info(f"Environment overrides applied: {list(env_overrides.keys())}")

            # Validate with Pydantic
            validated_config = CoralConfig(**merged_config)

            logger.info(f"Configuration validated successfully for experiment: {validated_config.experiment.name}, executor: {validated_config.infra.executor}")

            return validated_config

        except FileNotFoundError as e:
            raise ConfigurationError(f"Configuration file not found: {e}",
                                   context={"config_path": str(config_path)})
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}",
                                   context={"validation_errors": e.errors()})
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}",
                                   context={"config_path": str(config_path)},
                                   cause=e)

    def _load_base_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Raw configuration dictionary
        """
        if config_path is None:
            config_path = self._find_default_config()

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}",
                                   context={"config_path": str(config_path)})

        if not raw_config:
            raise ConfigurationError("Configuration file is empty",
                                   context={"config_path": str(config_path)})

        return raw_config

    def _find_default_config(self) -> Path:
        """Find default configuration file."""
        # Look for environment-specific config first
        env_config = Path(f"config/{self.environment}.yaml")
        if env_config.exists():
            return env_config

        # Fall back to main config
        main_config = Path("config/main.yaml")
        if main_config.exists():
            return main_config

        raise ConfigurationError("No default configuration file found",
                               context={"environment": self.environment})

    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables.
        
        Returns:
            Dictionary with environment overrides
        """
        overrides = {}

        # Define environment variable mappings
        env_mappings = {
            "CORAL_POPULATION_SIZE": ("execution", "population_size"),
            "CORAL_GENERATIONS": ("execution", "generations"),
            "CORAL_SEED": ("seed",),
            "CORAL_OUTPUT_DIR": ("execution", "output_dir"),
            "CORAL_ARTIFACTS_DIR": ("cache", "artifacts_dir"),
            "CORAL_BASE_CHECKPOINT": ("cache", "base_checkpoint"),
            "CORAL_EXECUTOR": ("infra", "executor"),
            "CORAL_MAX_SAMPLES": ("experiment", "dataset", "max_samples"),
            "CORAL_MODEL_NAME": ("experiment", "model", "name"),
            "CORAL_BATCH_SIZE": ("training", "batch_size"),
            "CORAL_LEARNING_RATE": ("training", "learning_rate"),
        }

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = self._parse_env_value(os.environ[env_var])
                self._set_nested_value(overrides, config_path, value)
                logger.debug(f"Environment override applied: {env_var}={value} at path {config_path}")

        return overrides

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment
            
        Returns:
            Parsed value with appropriate type
        """
        # Try to parse as int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass

        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Return as string
        return value

    def _set_nested_value(self, config: Dict[str, Any], path: tuple, value: Any):
        """Set a nested value in a configuration dictionary.
        
        Args:
            config: Configuration dictionary to modify
            path: Tuple representing nested path
            value: Value to set
        """
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[path[-1]] = value

    def _merge_configs(self, base_config: Dict[str, Any],
                      overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base configuration with overrides.
        
        Args:
            base_config: Base configuration
            overrides: Override values
            
        Returns:
            Merged configuration
        """
        if not overrides:
            return base_config

        # Deep merge - this is a simplified version
        # In production, you might want to use a more sophisticated merge
        merged = base_config.copy()

        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value

        deep_merge(merged, overrides)
        return merged


def load_config(config_path: Optional[Path] = None,
                environment: str = "development") -> CoralConfig:
    """Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        
    Returns:
        Validated CoralConfig instance
    """
    manager = ConfigManager(environment)
    return manager.load_config(config_path)
