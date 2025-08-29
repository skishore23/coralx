"""
Modal LoRA Training Service - Clean Architecture Wrapper
"""
from typing import Dict, Any
from pathlib import Path
import shutil

def train_lora_adapter_modal(base_model: str, heavy_key, save_path: str, config: Dict[str, Any]) -> str:
    """Modal wrapper for LoRA adapter training.
    
    Args:
        base_model: Base model checkpoint
        heavy_key: Serialized heavy genes (list/tuple from Modal) or HeavyGenes object
        save_path: Path to save adapter
        config: Training configuration
        
    Returns:
        str: Path to saved adapter
    """
    # Import domain training logic and adapter cache
    from core.domain.lora_training import train_codellama_lora
    from infra.adapter_cache import HeavyGenes
    from core.domain.mapping import deserialize_heavy_genes_from_modal
    
    # Convert serialized heavy_key back to HeavyGenes object
    if isinstance(heavy_key, (list, tuple)):
        # Modal serialization format
        heavy_key_tuple = deserialize_heavy_genes_from_modal(heavy_key)
        heavy_genes = HeavyGenes(*heavy_key_tuple)
    elif isinstance(heavy_key, HeavyGenes):
        # Already a HeavyGenes object
        heavy_genes = heavy_key
    else:
        raise ValueError(
            f"  Invalid heavy_key format: {type(heavy_key)}. "
            f"Expected list/tuple (Modal format) or HeavyGenes object."
        )
    
    # Train using clean HeavyGenes object - returns adapter path string
    adapter_path = train_codellama_lora(base_model, heavy_genes, save_path, config)
    
    # ✅ COMMIT VOLUME CHANGES - Critical for container coordination
    # This ensures other containers (generation, evaluation) can see the trained adapter
    try:
        from infra.adapter_cache import is_modal_environment
        if is_modal_environment():
            import modal
            volume = modal.Volume.from_name("coral-x-clean-cache")
            volume.commit()
            print(f"✅ Modal volume committed - other containers can now see adapter")
    except Exception as commit_error:
        print(f"⚠️  Volume commit warning: {commit_error}")
    
    # Return the adapter path (function now returns string directly)
    return adapter_path 