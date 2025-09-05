"""
Adapter Configuration
LoRA/DoRA configuration management.
"""
from dataclasses import dataclass
from typing import Dict, Any, Literal
from pathlib import Path


@dataclass(frozen=True)
class AdapterParameters:
    """Immutable adapter parameters."""
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple  # Immutable tuple instead of list
    adapter_type: Literal["lora", "dora"]
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        """Validate parameters at creation time."""
        if not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError(f"  Invalid rank: {self.rank}. Must be positive integer.")

        if not isinstance(self.alpha, (int, float)) or self.alpha <= 0:
            raise ValueError(f"  Invalid alpha: {self.alpha}. Must be positive number.")

        if not isinstance(self.dropout, (int, float)) or not (0 <= self.dropout <= 1):
            raise ValueError(f"  Invalid dropout: {self.dropout}. Must be in [0, 1].")

        if not isinstance(self.target_modules, (list, tuple)) or len(self.target_modules) == 0:
            raise ValueError(f"  Invalid target_modules: {self.target_modules}. Must be non-empty list/tuple.")

        if self.adapter_type not in ["lora", "dora"]:
            raise ValueError(f"  Invalid adapter_type: {self.adapter_type}. Must be 'lora' or 'dora'.")


@dataclass(frozen=True)
class AdapterEnvironment:
    """Immutable adapter environment configuration."""
    base_model_name: str
    cache_path: str
    save_path: str
    training_data_count: int
    config_source: str  # Track where config came from

    def __post_init__(self):
        """Validate environment at creation time."""
        if not self.base_model_name or not isinstance(self.base_model_name, str):
            raise ValueError(f"  Invalid base_model_name: {self.base_model_name}")

        if not self.cache_path or not isinstance(self.cache_path, str):
            raise ValueError(f"  Invalid cache_path: {self.cache_path}")

        if not self.save_path or not isinstance(self.save_path, str):
            raise ValueError(f"  Invalid save_path: {self.save_path}")


@dataclass(frozen=True)
class AdapterConfiguration:
    """Complete immutable adapter configuration - Variable Orchestration."""
    parameters: AdapterParameters
    environment: AdapterEnvironment
    execution_context: str

    @property
    def config_hash(self) -> str:
        """Generate deterministic hash for caching."""
        import hashlib
        config_str = f"{self.parameters.rank}_{self.parameters.alpha}_{self.parameters.dropout}_{self.parameters.adapter_type}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    @property
    def display_name(self) -> str:
        """Human-readable configuration name."""
        return f"{self.parameters.adapter_type.upper()}_r{self.parameters.rank}_a{self.parameters.alpha}_d{self.parameters.dropout}"


# Pure Functors for Variable Orchestration

def create_adapter_parameters_from_genes(heavy_genes, config: Dict[str, Any]) -> AdapterParameters:
    """Create adapter parameters from heavy genes and config."""
    adapter_type = config.get('adapter_type', 'lora')

    # Extract from heavy genes with validation
    try:
        rank = int(heavy_genes.rank)
        alpha = float(heavy_genes.alpha)
        dropout = float(heavy_genes.dropout)
        target_modules = tuple(heavy_genes.target_modules)  # Ensure immutable
    except (AttributeError, ValueError, TypeError) as e:
        raise ValueError(
            f"  Cannot extract parameters from heavy_genes: {e}\n"
            f"Heavy genes type: {type(heavy_genes)}\n"
            f"Expected HeavyGenes object with rank, alpha, dropout, target_modules."
        )

    return AdapterParameters(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        adapter_type=adapter_type
    )


def create_adapter_environment_from_config(config: Dict[str, Any], save_path: str, training_data_count: int) -> AdapterEnvironment:
    """Create adapter environment from config and paths."""
    from core.config.path_utils import create_path_config_from_dict, get_model_cache_path

    # Get executor type and resolve paths
    executor_type = config.get('infra', {}).get('executor', 'modal')
    path_config = create_path_config_from_dict(config, executor_type)
    cache_path = get_model_cache_path(path_config)

    # Extract base model
    base_model_name = config.get('cache', {}).get('base_checkpoint', 'unknown')
    if base_model_name == 'unknown':
        base_model_name = config.get('experiment', {}).get('model', {}).get('name', 'unknown')

    if base_model_name == 'unknown':
        raise ValueError(
            "  Cannot determine base model name from config.\n"
            "Check config sections: cache.base_checkpoint or experiment.model.name"
        )

    return AdapterEnvironment(
        base_model_name=base_model_name,
        cache_path=cache_path,
        save_path=save_path,
        training_data_count=training_data_count,
        config_source=f"{executor_type}_config"
    )


def create_complete_adapter_config(heavy_genes, config: Dict[str, Any], save_path: str, training_data_count: int, execution_context: str) -> AdapterConfiguration:
    """Create complete adapter configuration."""

    # Compose pure functions
    parameters = create_adapter_parameters_from_genes(heavy_genes, config)
    environment = create_adapter_environment_from_config(config, save_path, training_data_count)

    return AdapterConfiguration(
        parameters=parameters,
        environment=environment,
        execution_context=execution_context
    )


def validate_adapter_compatibility(adapter_config: AdapterConfiguration) -> None:
    """Pure function: AdapterConfiguration → Validation Result"""

    # Check DoRA availability if requested
    if adapter_config.parameters.adapter_type == "dora":
        if not _check_dora_support_runtime():
            raise RuntimeError(
                f"  DoRA adapter requested but not available.\n"
                f"Configuration: {adapter_config.display_name}\n"
                f"Execution context: {adapter_config.execution_context}\n"
                f"Install peft>=0.10 to use DoRA adapters.\n"
                f"NO FALLBACKS - fix your environment or change adapter_type to 'lora'"
            )

    # Validate cache path accessibility
    cache_path = Path(adapter_config.environment.cache_path)
    if not cache_path.exists():
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"  Cannot create cache directory: {cache_path}\n"
                f"Configuration: {adapter_config.display_name}\n"
                f"Error: {e}\n"
                f"Ensure volume is properly mounted and writable."
            )

    # Validate save path directory
    save_path = Path(adapter_config.environment.save_path)
    save_dir = save_path.parent
    if not save_dir.exists():
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"  Cannot create save directory: {save_dir}\n"
                f"Configuration: {adapter_config.display_name}\n"
                f"Error: {e}"
            )


def log_adapter_configuration(adapter_config: AdapterConfiguration) -> None:
    """Pure function: AdapterConfiguration → Logging Side Effect"""
    print("🔧 ADAPTER CONFIGURATION ORCHESTRATION")
    print(f"   Display name: {adapter_config.display_name}")
    print(f"   Config hash: {adapter_config.config_hash}")
    print(f"   Execution context: {adapter_config.execution_context}")
    print("")
    print("📊 PARAMETERS:")
    print(f"   • Type: {adapter_config.parameters.adapter_type.upper()}")
    print(f"   • Rank: {adapter_config.parameters.rank}")
    print(f"   • Alpha: {adapter_config.parameters.alpha}")
    print(f"   • Dropout: {adapter_config.parameters.dropout}")
    print(f"   • Target modules: {adapter_config.parameters.target_modules}")
    print("")
    print("🌍 ENVIRONMENT:")
    print(f"   • Base model: {adapter_config.environment.base_model_name}")
    print(f"   • Cache path: {adapter_config.environment.cache_path}")
    print(f"   • Save path: {adapter_config.environment.save_path}")
    print(f"   • Training examples: {adapter_config.environment.training_data_count}")
    print(f"   • Config source: {adapter_config.environment.config_source}")


def _check_dora_support_runtime() -> bool:
    """Check DoRA support at runtime."""
    try:
        import inspect
        from peft import LoraConfig as PeftLoraConfig

        sig = inspect.signature(PeftLoraConfig.__init__)
        return 'use_dora' in sig.parameters
    except Exception:
        return False
