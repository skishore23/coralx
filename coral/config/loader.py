###############################################################################
# YAML → CoralCfg - NO FALLBACKS, strict config-driven
###############################################################################
import os
import yaml
from pathlib import Path
from typing import Dict, Any

from ..domain.mapping import EvolutionConfig
from ..domain.threshold_gate import ThresholdConfig, ObjectiveThresholds
from ..application.evolution_engine import CoralConfig


def load_config(path: str, env: Dict[str, str] = None) -> CoralConfig:
    """Load configuration from YAML file with environment overrides - FAIL-FAST."""
    if env is None:
        env = os.environ
    
    # Load YAML
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"FAIL-FAST: Config file not found: {path}")
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    if not raw_config:
        raise ValueError(f"FAIL-FAST: Config file is empty or invalid: {path}")
    
    # Apply environment overrides
    raw_config = _apply_env_overrides(raw_config, env)
    
    # Validate configuration
    _validate_config(raw_config)
    
    # Convert to structured config
    return _build_coral_config(raw_config)


def _apply_env_overrides(config: Dict[str, Any], env: Dict[str, str]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    # Create a copy to avoid mutating the original
    config = dict(config)
    
    # Define environment variable mappings
    env_mappings = {
        "CORALX_POPULATION_SIZE": ("execution", "population_size"),
        "CORALX_GENERATIONS": ("execution", "generations"),
        "CORALX_GPU": ("infra", "modal", "functions", "evaluate_genome", "gpu"),
        "CORALX_SEED": ("seed",),
        "CORALX_OUTPUT_DIR": ("execution", "output_dir"),
        "CORALX_ARTIFACTS_DIR": ("cache", "artifacts_dir"),
        "CORALX_BASE_CHECKPOINT": ("cache", "base_checkpoint"),
    }
    
    for env_var, config_path in env_mappings.items():
        if env_var in env:
            _set_nested_value(config, config_path, _parse_env_value(env[env_var]))
    
    return config


def _set_nested_value(config: Dict[str, Any], path: tuple, value: Any):
    """Set a nested value in a configuration dictionary."""
    current = config
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[path[-1]] = value


def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type."""
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


def _validate_config(config: Dict[str, Any]):
    """Validate configuration structure and values - FAIL-FAST."""
    required_sections = ['evo', 'execution', 'experiment', 'infra', 'cache', 'threshold', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"FAIL-FAST: Missing required configuration section: '{section}'")
    
    # Validate evolution configuration
    evo_config = config['evo']
    required_evo_fields = ['rank_candidates', 'alpha_candidates', 'dropout_candidates', 'target_modules']
    for field in required_evo_fields:
        if field not in evo_config:
            raise ValueError(f"FAIL-FAST: Missing '{field}' in evolution configuration")
    
    # Validate CA configuration within evo
    if 'ca' in evo_config:
        ca_config = evo_config['ca']
        required_ca_fields = ['grid_size', 'rule_range', 'steps_range', 'initial_density']
        for field in required_ca_fields:
            if field not in ca_config:
                raise ValueError(f"FAIL-FAST: Missing '{field}' in CA configuration")
    
    # Validate execution configuration
    exec_config = config['execution']
    required_exec_fields = ['population_size', 'generations', 'output_dir']
    for field in required_exec_fields:
        if field not in exec_config:
            raise ValueError(f"FAIL-FAST: Missing '{field}' in execution configuration")
    
    # Validate experiment configuration
    exp_config = config['experiment']
    required_exp_fields = ['name', 'target', 'dataset', 'model']
    for field in required_exp_fields:
        if field not in exp_config:
            raise ValueError(f"FAIL-FAST: Missing '{field}' in experiment configuration")
    
    # Validate dataset config
    dataset_config = exp_config.get('dataset', {})
    if 'path' not in dataset_config:
        raise ValueError(f"FAIL-FAST: Missing 'path' in dataset configuration")
    
    # Validate model config
    model_config = exp_config.get('model', {})
    if 'name' not in model_config:
        raise ValueError(f"FAIL-FAST: Missing 'name' in model configuration")
    
    # Validate infrastructure configuration
    infra_config = config['infra']
    if 'executor' not in infra_config:
        raise ValueError(f"FAIL-FAST: Missing 'executor' in infrastructure configuration")
    
    # Validate cache configuration
    cache_config = config['cache']
    required_cache_fields = ['artifacts_dir', 'base_checkpoint']
    for field in required_cache_fields:
        if field not in cache_config:
            raise ValueError(f"FAIL-FAST: Missing '{field}' in cache configuration")
    
    # Validate threshold configuration
    threshold_config = config['threshold']
    required_threshold_fields = ['base_thresholds', 'max_thresholds', 'schedule']
    for field in required_threshold_fields:
        if field not in threshold_config:
            raise ValueError(f"FAIL-FAST: Missing '{field}' in threshold configuration")
    
    # Validate evaluation configuration
    eval_config = config['evaluation']
    if 'fitness_weights' not in eval_config:
        raise ValueError(f"FAIL-FAST: Missing 'fitness_weights' in evaluation configuration")
    
    # Validate fitness weights
    fitness_weights = eval_config['fitness_weights']
    required_weights = ['bugfix', 'style', 'security', 'runtime', 'syntax']  # NEW: Include syntax
    for weight in required_weights:
        if weight not in fitness_weights:
            # Provide default for syntax if missing (backward compatibility)
            if weight == 'syntax':
                print(f"⚠️  Warning: Missing 'syntax' weight in fitness_weights, using default 0.2")
                fitness_weights['syntax'] = 0.2
            else:
                raise ValueError(f"FAIL-FAST: Missing fitness weight '{weight}' in evaluation configuration")
    
    # Validate adaptive testing if enabled
    if 'adaptive_testing' in eval_config:
        adaptive_config = eval_config['adaptive_testing']
        if adaptive_config.get('enable', False):
            if 'capability_thresholds' not in adaptive_config:
                raise ValueError(f"FAIL-FAST: Missing 'capability_thresholds' in adaptive_testing configuration")
    
    # Validate seed
    if 'seed' not in config:
        raise ValueError(f"FAIL-FAST: Missing 'seed' in configuration")


def _build_coral_config(raw_config: Dict[str, Any]) -> CoralConfig:
    """Build enhanced CoralConfig with raw data as source of truth."""
    # Create enhanced CoralConfig with raw data as source of truth
    return CoralConfig(raw_config)


def create_config_from_dict(config_dict: Dict[str, Any]) -> CoralConfig:
    """Create CoralConfig directly from dictionary (for Modal) - FAIL-FAST."""
    # Apply validation
    _validate_config(config_dict)
    
    # Build structured config
    return _build_coral_config(config_dict)


# ========== NEW: Monadic Configuration Loading Alternative ==========
# This demonstrates how the new categorical abstractions could replace FAIL-FAST

def load_config_monadic(path: str, env: Dict[str, str] = None) -> 'Result[CoralConfig, str]':
    """
    Monadic configuration loading using Result monad.
    Alternative to exception-based FAIL-FAST approach.
    Demonstrates compositional error handling.
    """
    # Import the new categorical types
    from coral.domain.categorical_result import (
        success, error, safe_call, ConfigValidation, compose_results
    )
    
    if env is None:
        env = os.environ
    
    def load_yaml_file(config_path: str):
        """Load YAML file safely."""
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path_obj) as f:
            raw_config = yaml.safe_load(f)
        
        if not raw_config:
            raise ValueError(f"Config file is empty or invalid: {config_path}")
        
        return raw_config
    
    def apply_env_overrides(config):
        """Apply environment overrides."""
        return _apply_env_overrides(config, env)
    
    def validate_config_monadic(config):
        """Validate configuration using monadic composition."""
        # Chain multiple validations
        return (
            ConfigValidation.validate_required_field(config, 'evo')
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'execution'))
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'experiment'))
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'infra'))
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'cache'))
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'threshold'))
            .bind(lambda _: ConfigValidation.validate_required_field(config, 'evaluation'))
            .bind(lambda _: success(config))
        )
    
    def build_config(config):
        """Build structured config."""
        return _build_coral_config(config)
    
    # Compose the entire pipeline using monadic composition
    pipeline = compose_results(
        lambda p: load_yaml_file(p),
        apply_env_overrides,
        lambda c: validate_config_monadic(c).unwrap(),  # Extract from Result
        build_config
    )
    
    # Execute pipeline with safe error handling
    return safe_call(lambda: pipeline(path), f"Failed to load config from {path}")


def create_default_config() -> Dict[str, Any]:
    """FAIL-FAST: No default configs - use real configuration files."""
    raise NotImplementedError(
        f"FAIL-FAST: Default configuration creation violates fail-fast principle. "
        f"Create a proper YAML configuration file instead of using defaults."
    ) 