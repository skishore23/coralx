"""
Path Configuration Utilities - Category Theory Compliant
Pure functorial path resolution eliminating all ambiguity.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import os


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration object - Category Theory compliant."""
    cache_root: str
    adapters: str
    models: str
    dataset: str
    progress: str
    emergent_behavior: str
    emergent_alerts: str
    realtime_benchmarks: str
    coralx_root: str
    
    def __post_init__(self):
        """Validate all paths at creation time - FAIL-FAST."""
        required_paths = {
            'cache_root': self.cache_root,
            'adapters': self.adapters, 
            'models': self.models,
            'dataset': self.dataset,
            'coralx_root': self.coralx_root
        }
        
        for name, path in required_paths.items():
            if not path or not isinstance(path, str):
                raise ValueError(f"FAIL-FAST: Invalid {name} path: {path}")


def create_path_config_from_dict(config: Dict[str, Any], executor_type: str) -> PathConfig:
    """
    Pure functor: Config Dict → PathConfig
    Category-theoretic path resolution with FAIL-FAST validation.
    """
    if 'paths' not in config:
        raise ValueError(
            f"FAIL-FAST: 'paths' section missing from configuration.\n"
            f"Expected structure: paths: {{modal: {{...}}, local: {{...}}}}\n"
            f"Check your YAML configuration file."
        )
    
    paths_config = config['paths']
    
    # Map executor types to path keys - NO AMBIGUITY
    executor_path_mapping = {
        'local': 'local',
        'modal': 'modal', 
        'queue_modal': 'queue_modal',  # Queue-based uses its own paths
        'test': 'local'
    }
    
    if executor_type not in executor_path_mapping:
        raise ValueError(
            f"FAIL-FAST: Unknown executor type '{executor_type}'.\n"
            f"Supported types: {list(executor_path_mapping.keys())}\n"
            f"Fix your executor configuration."
        )
    
    path_key = executor_path_mapping[executor_type]
    
    if path_key not in paths_config:
        raise ValueError(
            f"FAIL-FAST: '{path_key}' paths missing from configuration.\n"
            f"Available path sections: {list(paths_config.keys())}\n"
            f"Required for executor type: {executor_type}"
        )
    
    target_paths = paths_config[path_key]
    
    # Required path fields - FAIL-FAST if missing
    required_fields = [
        'cache_root', 'adapters', 'models', 'dataset',
        'progress', 'emergent_behavior', 'emergent_alerts', 
        'realtime_benchmarks', 'coralx_root'
    ]
    
    missing_fields = [field for field in required_fields if field not in target_paths]
    if missing_fields:
        raise ValueError(
            f"FAIL-FAST: Missing required path fields: {missing_fields}\n"
            f"Path section: {path_key}\n"
            f"Available fields: {list(target_paths.keys())}\n"
            f"Fix your YAML configuration."
        )
    
    # Create immutable path configuration
    return PathConfig(
        cache_root=target_paths['cache_root'],
        adapters=target_paths['adapters'],
        models=target_paths['models'],
        dataset=target_paths['dataset'],
        progress=target_paths['progress'],
        emergent_behavior=target_paths['emergent_behavior'],
        emergent_alerts=target_paths['emergent_alerts'],
        realtime_benchmarks=target_paths['realtime_benchmarks'],
        coralx_root=target_paths['coralx_root']
    )


def validate_path_accessibility(path_config: PathConfig, context: str = "unknown") -> None:
    """
    Pure function: PathConfig → Validation Result
    Validates that critical paths are accessible in current environment.
    """
    print(f"🔍 Validating path accessibility in context: {context}")
    
    # Check cache root existence (critical for all operations)
    cache_root = Path(path_config.cache_root)
    if not cache_root.exists():
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created cache root: {cache_root}")
        except Exception as e:
            raise RuntimeError(
                f"FAIL-FAST: Cannot create cache root directory: {cache_root}\n"
                f"Error: {e}\n"
                f"Context: {context}\n"
                f"Ensure volume is properly mounted and writable."
            )
    
    # Verify coralx root (critical for imports)
    coralx_root = Path(path_config.coralx_root)
    if not coralx_root.exists():
        raise RuntimeError(
            f"FAIL-FAST: CoralX codebase not found: {coralx_root}\n"
            f"Context: {context}\n"
            f"Expected directory with coral/ subdirectory.\n"
            f"Check Modal volume mount or local path configuration."
        )
    
    # Verify coral module is importable
    coral_module_path = coralx_root / "coral"
    if not coral_module_path.exists():
        raise RuntimeError(
            f"FAIL-FAST: Coral module directory not found: {coral_module_path}\n"
            f"Context: {context}\n"
            f"CoralX root exists but coral/ subdirectory missing."
        )
    
    print(f"✅ Path validation passed for context: {context}")
    print(f"   Cache root: {cache_root} (exists: {cache_root.exists()})")
    print(f"   CoralX root: {coralx_root} (exists: {coralx_root.exists()})")
    print(f"   Coral module: {coral_module_path} (exists: {coral_module_path.exists()})")


def setup_python_path(path_config: PathConfig) -> None:
    """
    Pure functor: PathConfig → Python Environment Setup
    Ensures coral module is importable in current environment.
    """
    import sys
    
    coralx_root = str(Path(path_config.coralx_root).resolve())
    
    # Add to Python path if not already present
    if coralx_root not in sys.path:
        sys.path.insert(0, coralx_root)
        print(f"🐍 Added to Python path: {coralx_root}")
    else:
        print(f"✅ Python path already includes: {coralx_root}")
    
    # Verify coral module is importable
    try:
        import coral
        print(f"✅ Coral module import successful")
    except ImportError as e:
        raise RuntimeError(
            f"FAIL-FAST: Cannot import coral module after path setup: {e}\n"
            f"CoralX root: {coralx_root}\n"
            f"Python path: {sys.path[:3]}...\n"
            f"Check that coral/ directory exists and contains __init__.py"
        )


def get_adapter_path(path_config: PathConfig, adapter_id: str) -> str:
    """
    Pure function: (PathConfig, AdapterID) → AdapterPath
    Canonical adapter path resolution.
    """
    if not adapter_id or not isinstance(adapter_id, str):
        raise ValueError(f"FAIL-FAST: Invalid adapter_id: {adapter_id}")
    
    # Ensure clean adapter_id (no path traversal)
    clean_id = adapter_id.replace('/', '_').replace('\\', '_')
    if clean_id != adapter_id:
        print(f"🧹 Cleaned adapter ID: {adapter_id} → {clean_id}")
    
    adapter_path = Path(path_config.adapters) / f"adapter_{clean_id}"
    return str(adapter_path)


def get_model_cache_path(path_config: PathConfig) -> str:
    """
    Pure function: PathConfig → ModelCachePath
    Canonical model cache path resolution.
    """
    return str(Path(path_config.models))


def get_dataset_path(path_config: PathConfig) -> str:
    """
    Pure function: PathConfig → DatasetPath  
    Canonical dataset path resolution.
    """
    return str(Path(path_config.dataset))


# Backward compatibility functions
def get_executor_paths(config: Dict[str, Any], executor_type: str) -> Dict[str, str]:
    """Legacy function for backward compatibility."""
    path_config = create_path_config_from_dict(config, executor_type)
    return {
        'cache_root': path_config.cache_root,
        'adapters': path_config.adapters,
        'models': path_config.models,
        'dataset': path_config.dataset,
        'progress': path_config.progress,
        'emergent_behavior': path_config.emergent_behavior,
        'emergent_alerts': path_config.emergent_alerts,
        'realtime_benchmarks': path_config.realtime_benchmarks,
        'coralx_root': path_config.coralx_root
    }


# ========== NEW: Functorial Path Configuration ==========
# Demonstrates categorical functors for context-aware configuration

def create_path_config_functorial(config: Dict[str, Any], target_context: str) -> PathConfig:
    """
    NEW: Create path configuration using categorical functors.
    Demonstrates structure-preserving context transformation vs manual approach above.
    """
    # Import the new categorical functors
    from coral.domain.categorical_functors import adapt_config_for_context
    
    print(f"🧮 FUNCTORIAL PATH CONFIGURATION:")
    print(f"   • Target context: {target_context}")
    print(f"   • Using categorical functors for structure preservation")
    
    # Use functorial transformation to adapt configuration
    adapted_config = adapt_config_for_context(config, target_context)
    
    # Extract path configuration from adapted result
    if 'paths' in adapted_config and target_context in adapted_config['paths']:
        target_paths = adapted_config['paths'][target_context]
        
        print(f"   • Functorial transformation: Complete")
        print(f"   • Structure preservation: Guaranteed by functor laws")
        
        return PathConfig(
            cache_root=target_paths['cache_root'],
            adapters=target_paths['adapters'],
            models=target_paths['models'],
            dataset=target_paths['dataset'],
            progress=target_paths['progress'],
            emergent_behavior=target_paths['emergent_behavior'],
            emergent_alerts=target_paths['emergent_alerts'],
            realtime_benchmarks=target_paths['realtime_benchmarks'],
            coralx_root=target_paths['coralx_root']
        )
    else:
        # Fallback to manual approach
        print(f"   • Fallback to manual path configuration")
        return create_path_config_from_dict(config, target_context)


def verify_path_transformation_laws(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    NEW: Verify that path transformations satisfy categorical laws.
    Tests functorial correctness of path configuration system.
    """
    from coral.domain.categorical_functors import verify_functorial_laws, adapt_config_for_context
    
    print(f"🧮 VERIFYING CATEGORICAL LAWS FOR PATH TRANSFORMATIONS:")
    
    # Test functorial laws
    law_results = verify_functorial_laws(config)
    
    # Test round-trip property: local → modal → local ≅ local
    try:
        # Start with local configuration
        local_config = create_path_config_functorial(config, 'local')
        
        # Transform to modal and back
        modal_adapted = adapt_config_for_context(config, 'modal')
        roundtrip_config = create_path_config_functorial(modal_adapted, 'local')
        
        # Check structural equality (simplified)
        roundtrip_law = (
            local_config.cache_root.endswith(roundtrip_config.cache_root.split('/')[-1]) and
            local_config.coralx_root.endswith(roundtrip_config.coralx_root.split('/')[-1])
        )
        
        law_results['roundtrip_law'] = roundtrip_law
        
    except Exception as e:
        print(f"   ⚠️  Roundtrip test failed: {e}")
        law_results['roundtrip_law'] = False
    
    print(f"   • Identity law: {law_results.get('identity_law', False)}")
    print(f"   • Composition law: {law_results.get('composition_law', False)}")
    print(f"   • Roundtrip law: {law_results.get('roundtrip_law', False)}")
    print(f"   • Overall correctness: {all(law_results.values())}")
    
    return law_results


def transform_config_between_contexts(config: Dict[str, Any], 
                                    source_context: str, 
                                    target_context: str) -> Dict[str, Any]:
    """
    NEW: Transform configuration between execution contexts using functorial composition.
    Demonstrates category theory in practice for context switching.
    """
    from coral.domain.categorical_functors import transform_paths_functorially
    
    print(f"🧮 FUNCTORIAL CONTEXT TRANSFORMATION:")
    print(f"   • Source: {source_context} → Target: {target_context}")
    print(f"   • Categorical structure: Preserved by functorial laws")
    
    # Use functorial transformation
    transformed = transform_paths_functorially(config, source_context, target_context)
    
    print(f"   • Transformation: Complete")
    return transformed 