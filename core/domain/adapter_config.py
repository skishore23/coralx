"""Adapter configuration value objects."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.domain.stable_hash import stable_digest


@dataclass(frozen=True)
class AdapterParameters:
    """Immutable structural adapter parameters."""

    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    adapter_type: Literal["lora", "dora"]
    task_type: str = "CAUSAL_LM"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"Invalid rank: {self.rank}. Must be positive.")
        if self.alpha <= 0:
            raise ValueError(f"Invalid alpha: {self.alpha}. Must be positive.")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Invalid dropout: {self.dropout}. Must be in [0, 1].")
        if not self.target_modules:
            raise ValueError("target_modules must be non-empty.")
        if self.adapter_type not in ("lora", "dora"):
            raise ValueError("adapter_type must be 'lora' or 'dora'.")


@dataclass(frozen=True)
class AdapterEnvironment:
    """Local adapter artifact environment."""

    base_model_name: str
    cache_path: str
    save_path: str
    training_data_count: int
    config_source: str = "local_config"

    def __post_init__(self) -> None:
        if not self.base_model_name:
            raise ValueError("base_model_name is required.")
        if not self.cache_path:
            raise ValueError("cache_path is required.")
        if not self.save_path:
            raise ValueError("save_path is required.")


@dataclass(frozen=True)
class AdapterConfiguration:
    """Complete immutable local adapter configuration."""

    parameters: AdapterParameters
    environment: AdapterEnvironment
    execution_context: str

    @property
    def config_hash(self) -> str:
        """Generate a stable structural config hash."""
        return stable_digest(
            {
                "rank": self.parameters.rank,
                "alpha": self.parameters.alpha,
                "dropout": self.parameters.dropout,
                "target_modules": list(self.parameters.target_modules),
                "adapter_type": self.parameters.adapter_type,
            },
            16,
        )

    @property
    def display_name(self) -> str:
        """Human-readable configuration name."""
        return (
            f"{self.parameters.adapter_type.upper()}_r{self.parameters.rank}"
            f"_a{self.parameters.alpha}_d{self.parameters.dropout}"
        )


def create_adapter_parameters_from_genes(
    heavy_genes: Any, config: dict[str, Any]
) -> AdapterParameters:
    """Create adapter parameters from heavy genes and config."""
    return AdapterParameters(
        rank=int(heavy_genes.rank),
        alpha=float(heavy_genes.alpha),
        dropout=float(heavy_genes.dropout),
        target_modules=tuple(heavy_genes.target_modules),
        adapter_type=config.get("adapter_type", "lora"),
    )


def create_adapter_environment_from_config(
    config: dict[str, Any], save_path: str, training_data_count: int
) -> AdapterEnvironment:
    """Create a local adapter environment from config."""
    cache = config.get("cache", {})
    experiment_model = config.get("experiment", {}).get("model", {})
    base_model_name = cache.get("base_checkpoint") or experiment_model.get("name")
    if not base_model_name:
        raise ValueError("cache.base_checkpoint or experiment.model.name is required.")

    cache_path = str(Path(cache.get("artifacts_dir", "./artifacts/adapters")))
    return AdapterEnvironment(
        base_model_name=base_model_name,
        cache_path=cache_path,
        save_path=save_path,
        training_data_count=training_data_count,
    )


def create_complete_adapter_config(
    heavy_genes: Any,
    config: dict[str, Any],
    save_path: str,
    training_data_count: int,
    execution_context: str,
) -> AdapterConfiguration:
    """Create complete local adapter configuration."""
    return AdapterConfiguration(
        parameters=create_adapter_parameters_from_genes(heavy_genes, config),
        environment=create_adapter_environment_from_config(
            config, save_path, training_data_count
        ),
        execution_context=execution_context,
    )


def validate_adapter_compatibility(adapter_config: AdapterConfiguration) -> None:
    """Validate local artifact directories for an adapter config."""
    Path(adapter_config.environment.cache_path).mkdir(parents=True, exist_ok=True)
    Path(adapter_config.environment.save_path).parent.mkdir(parents=True, exist_ok=True)
