"""Local structural adapter cache primitives.

The cache key is part of the local architecture and deterministic test surface.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.domain.mapping import LoRAConfig
from core.domain.stable_hash import stable_digest


@dataclass(frozen=True)
class HeavyGenes:
    """Structural adapter genes that define a trained adapter artifact."""

    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    adapter_type: str = "lora"
    run_id: str | None = None

    def to_hash(self) -> str:
        """Return a stable structural cache key."""
        return stable_digest(self.to_dict(), 16)

    def to_dict(self) -> dict[str, Any]:
        """Serialize heavy genes into canonical key material."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "adapter_type": self.adapter_type,
            "run_id": self.run_id,
        }

    @classmethod
    def from_lora_config(
        cls, lora_cfg: LoRAConfig, run_id: str | None = None
    ) -> "HeavyGenes":
        """Extract structural genes from a LoRA config."""
        return cls(
            rank=lora_cfg.r,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            adapter_type=getattr(lora_cfg, "adapter_type", "lora"),
            run_id=run_id,
        )


@dataclass(frozen=True)
class CacheConfig:
    """Local cache configuration."""

    artifacts_dir: str
    base_checkpoint: str
    cache_metadata: bool = True
    cleanup_threshold: int = 100
    run_id: str | None = None


def create_cache_config_from_dict(config_dict: dict[str, Any]) -> CacheConfig:
    """Create local cache config from the app config dictionary."""
    if "cache" not in config_dict:
        raise ValueError("'cache' section missing from configuration")

    cache_config = config_dict["cache"]
    return CacheConfig(
        artifacts_dir=cache_config["artifacts_dir"],
        base_checkpoint=cache_config["base_checkpoint"],
        cache_metadata=cache_config.get("metadata", True),
        cleanup_threshold=cache_config.get("cleanup_threshold", 100),
        run_id=cache_config.get("run_id"),
    )


class AdapterCache:
    """Small local structural cache wrapper."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.root = Path(config.artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, heavy_genes: HeavyGenes) -> Path:
        """Return the local artifact directory for structural genes."""
        return self.root / f"adapter_{heavy_genes.to_hash()}"

    def get_or_train_adapter(
        self,
        heavy_genes: HeavyGenes,
        trainer_fn: Callable[[HeavyGenes, str], str],
        base_checkpoint: str | None = None,
    ) -> str:
        """Return cached adapter path or call the supplied trainer."""
        adapter_path = self.path_for(heavy_genes)
        if adapter_path.exists():
            return str(adapter_path)

        checkpoint = base_checkpoint or self.config.base_checkpoint
        return trainer_fn(heavy_genes, checkpoint)


_cache_instance: AdapterCache | None = None


def get_adapter_cache(config: CacheConfig | None = None) -> AdapterCache:
    """Return process-local adapter cache instance."""
    global _cache_instance
    if _cache_instance is None:
        if config is None:
            raise ValueError("AdapterCache requires configuration on first access.")
        _cache_instance = AdapterCache(config)
    return _cache_instance


def get_or_train_adapter(
    heavy_genes: HeavyGenes,
    trainer_fn: Callable[[HeavyGenes, str], str],
    cache_config: CacheConfig,
) -> str:
    """Convenience wrapper around the process-local adapter cache."""
    return get_adapter_cache(cache_config).get_or_train_adapter(
        heavy_genes, trainer_fn, cache_config.base_checkpoint
    )
