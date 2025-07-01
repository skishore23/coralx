"""
Categorical Functors - Structure-Preserving Context Transformations
Proper functors for execution context switching preserving categorical laws.
"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Dict, Any, Optional, List
from abc import ABC, abstractmethod
from pathlib import Path

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


class Category(ABC):
    """
    Abstract category with objects and morphisms.
    Defines the basic structure for categorical reasoning.
    """
    
    @abstractmethod
    def objects(self) -> List[str]:
        """List of objects in this category."""
        pass
    
    @abstractmethod
    def morphisms(self, source: str, target: str) -> List[Callable]:
        """Morphisms between two objects."""
        pass
    
    @abstractmethod
    def identity(self, obj: str) -> Callable:
        """Identity morphism for object."""
        pass
    
    @abstractmethod
    def compose(self, f: Callable, g: Callable) -> Callable:
        """Compose morphisms f and g."""
        pass


@dataclass(frozen=True)
class ConfigurationObject:
    """
    Object in configuration category.
    Represents structured configuration with type safety.
    """
    config_type: str  # "local", "modal", "queue_modal"
    data: Dict[str, Any]
    validated: bool = False
    
    def __post_init__(self):
        """Validate configuration structure and convert CoralConfig to dict if needed."""
        # Handle CoralConfig objects by converting them to dictionaries
        if hasattr(self.data, '__dataclass_fields__') and hasattr(self.data, 'items'):
            # This is likely a CoralConfig object - convert to dictionary
            dict_data = dict(self.data.items())
            object.__setattr__(self, 'data', dict_data)
        elif not isinstance(self.data, dict):
            raise ValueError(f"Configuration data must be dict or CoralConfig, got {type(self.data)}")


class ConfigurationCategory(Category):
    """
    Category of configuration objects and their transformations.
    Objects: Local configs, Modal configs, Queue configs
    Morphisms: Structure-preserving transformations
    """
    
    def objects(self) -> List[str]:
        return ["local_config", "modal_config", "queue_modal_config"]
    
    def morphisms(self, source: str, target: str) -> List[Callable]:
        """Configuration transformation morphisms."""
        if source == "local_config" and target == "modal_config":
            return [self.local_to_modal_transform]
        elif source == "modal_config" and target == "local_config":
            return [self.modal_to_local_transform]
        elif source == target:
            return [self.identity(source)]
        else:
            return []
    
    def identity(self, obj: str) -> Callable:
        """Identity transformation for configuration."""
        return lambda config: config
    
    def compose(self, f: Callable, g: Callable) -> Callable:
        """Compose configuration transformations."""
        return lambda config: g(f(config))
    
    def local_to_modal_transform(self, local_config: ConfigurationObject) -> ConfigurationObject:
        """
        Functor morphism: Local Config → Modal Config
        Uses YAML-defined modal paths instead of hardcoded values.
        """
        modal_data = local_config.data.copy()
        
        # Use YAML-defined modal paths if available, otherwise use local paths as template
        if 'paths' in modal_data:
            if 'modal' in modal_data['paths']:
                # Modal paths already defined in YAML - use them directly
                pass  # Nothing to transform, YAML has the authoritative paths
            elif 'local' in modal_data['paths']:
                # Create modal paths based on YAML structure pattern if not defined
                # This should rarely happen as YAMLs should define both local and modal paths
                local_paths = modal_data['paths']['local']
                
                # Create modal section using the same key structure as local
                modal_paths = {}
                for key, local_value in local_paths.items():
                    # Transform local paths to modal pattern (this is a fallback)
                    if key == 'cache_root':
                        modal_paths[key] = '/cache'
                    elif key == 'coralx_root':
                        modal_paths[key] = '/root/coralx'
                    elif isinstance(local_value, str) and local_value.startswith('./'):
                        # Transform ./cache/something to /cache/something
                        modal_paths[key] = local_value.replace('./', '/cache/', 1)
                    elif isinstance(local_value, str) and local_value.startswith('./cache/'):
                        # Transform ./cache/something to /cache/something
                        modal_paths[key] = local_value.replace('./cache/', '/cache/')
                    else:
                        # Default: assume it should be under /cache
                        path_name = Path(local_value).name if isinstance(local_value, str) else str(local_value)
                        modal_paths[key] = f"/cache/{path_name}"
                
                modal_data['paths']['modal'] = modal_paths
        
        # Ensure Modal-specific configuration
        if 'infra' not in modal_data:
            modal_data['infra'] = {}
        
        modal_data['infra']['executor'] = 'modal'
        
        if 'modal' not in modal_data['infra']:
            modal_data['infra']['modal'] = {}
        
        # Set Modal-specific defaults
        modal_infra = modal_data['infra']['modal']
        if 'app_name' not in modal_infra:
            modal_infra['app_name'] = 'coral-x-production'
        if 'volume_name' not in modal_infra:
            modal_infra['volume_name'] = 'coral-x-clean-cache'
        
        return ConfigurationObject(
            config_type="modal_config",
            data=modal_data,
            validated=True
        )
    
    def modal_to_local_transform(self, modal_config: ConfigurationObject) -> ConfigurationObject:
        """
        Functor morphism: Modal Config → Local Config
        Uses YAML-defined local paths instead of hardcoded values.
        """
        local_data = modal_config.data.copy()
        
        # Use YAML-defined local paths if available, otherwise create from modal paths
        if 'paths' in local_data:
            if 'local' in local_data['paths']:
                # Local paths already defined in YAML - use them directly
                pass  # Nothing to transform, YAML has the authoritative paths
            elif 'modal' in local_data['paths']:
                # Create local paths based on modal paths pattern (fallback)
                modal_paths = local_data['paths']['modal']
                
                # Create local section using the same key structure as modal
                local_paths = {}
                for key, modal_value in modal_paths.items():
                    # Transform modal paths to local pattern (this is a fallback)
                    if key == 'cache_root':
                        local_paths[key] = './cache'
                    elif key == 'coralx_root':
                        local_paths[key] = '.'
                    elif isinstance(modal_value, str) and modal_value.startswith('/cache/'):
                        # Transform /cache/something to ./cache/something
                        local_paths[key] = modal_value.replace('/cache/', './cache/')
                    else:
                        # Default: assume it should be under ./cache
                        path_name = Path(modal_value).name if isinstance(modal_value, str) else str(modal_value)
                        local_paths[key] = f"./cache/{path_name}"
                
                local_data['paths']['local'] = local_paths
        
        # Set local executor
        if 'infra' not in local_data:
            local_data['infra'] = {}
        
        local_data['infra']['executor'] = 'local'
        
        return ConfigurationObject(
            config_type="local_config",
            data=local_data,
            validated=True
        )


class ExecutionContextFunctor(Generic[A, B]):
    """
    Proper functor for execution context transformations.
    
    Mathematical Properties:
    - Identity Preservation: fmap(id) = id
    - Composition Preservation: fmap(g ∘ f) = fmap(g) ∘ fmap(f)
    - Structure Preservation: Context semantics maintained across transformations
    """
    
    def __init__(self, source_category: Category, target_category: Category):
        self.source_category = source_category
        self.target_category = target_category
    
    def fmap(self, morphism: Callable[[A], B]) -> Callable[[ConfigurationObject], ConfigurationObject]:
        """
        Functorial mapping preserving categorical structure.
        Maps morphisms between contexts while preserving composition laws.
        """
        def mapped_morphism(config_obj: ConfigurationObject) -> ConfigurationObject:
            # Apply transformation while preserving categorical structure
            try:
                # Transform the configuration data using the morphism
                transformed_data = morphism(config_obj.data)
                
                # Preserve configuration object structure
                return ConfigurationObject(
                    config_type=config_obj.config_type,
                    data=transformed_data,
                    validated=config_obj.validated
                )
            except Exception as e:
                # Preserve error context for debugging
                raise RuntimeError(f"Functorial mapping failed: {e}")
        
        return mapped_morphism
    
    def verify_identity_law(self, obj: ConfigurationObject) -> bool:
        """
        Verify functor identity law: fmap(id) = id
        Tests that identity morphisms are preserved.
        """
        try:
            identity_fn = lambda x: x
            mapped_identity = self.fmap(identity_fn)
            result = mapped_identity(obj)
            
            # Check structural equality
            return (result.config_type == obj.config_type and 
                    result.data == obj.data and
                    result.validated == obj.validated)
        except:
            return False
    
    def verify_composition_law(self, 
                              f: Callable[[Dict], Dict],
                              g: Callable[[Dict], Dict],
                              obj: ConfigurationObject) -> bool:
        """
        Verify functor composition law: fmap(g ∘ f) = fmap(g) ∘ fmap(f)
        Tests that composition is preserved under functorial mapping.
        """
        try:
            # Direct composition
            composed = lambda x: g(f(x))
            direct_mapped = self.fmap(composed)
            direct_result = direct_mapped(obj)
            
            # Separate mapping then composition
            mapped_f = self.fmap(f)
            mapped_g = self.fmap(g)
            intermediate = mapped_f(obj)
            separate_result = mapped_g(intermediate)
            
            # Results should be equivalent
            return (direct_result.data == separate_result.data)
        except:
            return False


class PathConfigurationFunctor(ExecutionContextFunctor):
    """
    Specialized functor for path configuration transformations.
    Handles the specific case of local/Modal path mappings.
    """
    
    def __init__(self):
        config_category = ConfigurationCategory()
        super().__init__(config_category, config_category)
    
    def transform_local_to_modal(self, config: ConfigurationObject) -> ConfigurationObject:
        """Transform local configuration to Modal-compatible configuration."""
        return self.source_category.local_to_modal_transform(config)
    
    def transform_modal_to_local(self, config: ConfigurationObject) -> ConfigurationObject:
        """Transform Modal configuration to local-compatible configuration."""
        return self.source_category.modal_to_local_transform(config)
    
    def create_context_adaptive_config(self, 
                                     base_config: Dict[str, Any],
                                     target_context: str) -> ConfigurationObject:
        """
        Create configuration adapted for target execution context.
        Uses functorial transformations to preserve structure.
        """
        # Start with local configuration
        local_config = ConfigurationObject(
            config_type="local_config",
            data=base_config,
            validated=False
        )
        
        # Apply appropriate transformation functor
        if target_context == "modal":
            return self.transform_local_to_modal(local_config)
        elif target_context == "local":
            return local_config
        elif target_context == "queue_modal":
            # First transform to modal, then adapt for queue-based execution
            modal_config = self.transform_local_to_modal(local_config)
            queue_data = modal_config.data.copy()
            queue_data['infra']['executor'] = 'queue_modal'
            
            return ConfigurationObject(
                config_type="queue_modal_config",
                data=queue_data,
                validated=True
            )
        else:
            raise ValueError(f"Unknown target context: {target_context}")


class ResourceConfigurationFunctor:
    """
    Functor for resource configuration transformations.
    Maps between different resource allocation strategies while preserving semantics.
    """
    
    def __init__(self):
        self.resource_mappings = {
            "local": {
                "cpu_cores": 1,
                "memory_gb": 8,
                "gpu": None,
                "timeout_minutes": 30
            },
            "modal": {
                "cpu_cores": 4,
                "memory_gb": 32,
                "gpu": "A100-40GB",
                "timeout_minutes": 60
            },
            "queue_modal": {
                "cpu_cores": 2,
                "memory_gb": 16,
                "gpu": "A10G",
                "timeout_minutes": 45
            }
        }
    
    def fmap_resources(self, context_transform: Callable[[str], str]) -> Callable[[Dict], Dict]:
        """
        Map resource configurations through context transformations.
        Preserves resource allocation semantics across contexts.
        """
        def mapped_resources(resource_config: Dict[str, Any]) -> Dict[str, Any]:
            current_context = resource_config.get("context", "local")
            target_context = context_transform(current_context)
            
            if target_context in self.resource_mappings:
                target_resources = self.resource_mappings[target_context].copy()
                
                # Preserve any explicit overrides from original config
                for key, value in resource_config.items():
                    if key != "context" and value is not None:
                        target_resources[key] = value
                
                target_resources["context"] = target_context
                return target_resources
            else:
                return resource_config
        
        return mapped_resources


# Global instances for CoralX
config_category = ConfigurationCategory()
path_functor = PathConfigurationFunctor()
resource_functor = ResourceConfigurationFunctor()


# Convenience functions for use throughout CoralX
def create_modal_config(local_config: Dict[str, Any]) -> ConfigurationObject:
    """Create Modal-compatible configuration from local configuration."""
    local_obj = ConfigurationObject("local_config", local_config, False)
    return path_functor.transform_local_to_modal(local_obj)


def create_local_config(modal_config: Dict[str, Any]) -> ConfigurationObject:
    """Create local-compatible configuration from Modal configuration."""
    modal_obj = ConfigurationObject("modal_config", modal_config, False)
    return path_functor.transform_modal_to_local(modal_obj)


def adapt_config_for_context(config: Dict[str, Any], target_context: str) -> Dict[str, Any]:
    """
    Adapt configuration for target execution context using functorial transformations.
    Replaces ad-hoc context switching with proper categorical structure preservation.
    """
    adapted = path_functor.create_context_adaptive_config(config, target_context)
    return adapted.data


def transform_paths_functorially(config: Dict[str, Any], 
                                source_context: str, 
                                target_context: str) -> Dict[str, Any]:
    """
    Transform configuration paths using functorial composition.
    Ensures path transformations preserve categorical structure.
    """
    # Create source configuration object
    source_obj = ConfigurationObject(f"{source_context}_config", config, False)
    
    # Apply functorial transformation
    if source_context == "local" and target_context == "modal":
        result_obj = path_functor.transform_local_to_modal(source_obj)
    elif source_context == "modal" and target_context == "local":
        result_obj = path_functor.transform_modal_to_local(source_obj)
    else:
        # Identity transformation
        result_obj = source_obj
    
    return result_obj.data


def verify_functorial_laws(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Verify that functorial transformations satisfy categorical laws.
    Useful for testing and debugging categorical correctness.
    """
    test_obj = ConfigurationObject("local_config", config, False)
    
    # Test identity law
    identity_law = path_functor.verify_identity_law(test_obj)
    
    # Test composition law with simple transformations
    f = lambda x: {**x, "test_f": True}
    g = lambda x: {**x, "test_g": True}
    composition_law = path_functor.verify_composition_law(f, g, test_obj)
    
    return {
        "identity_law": identity_law,
        "composition_law": composition_law,
        "categorical_correctness": identity_law and composition_law
    } 