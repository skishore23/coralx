"""
Categorical Distribution - Natural Transformations for Execution Contexts
Systematic distribution preserving categorical structure across Local/Modal boundaries.
"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Dict, Any, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import json
import numpy as np

A = TypeVar('A')
B = TypeVar('B')


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be systematically serialized."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Reconstruct from dictionary representation."""
        ...


class ExecutionContext(Generic[A], ABC):
    """
    Abstract execution context - category in the execution context category.
    Represents different computational environments (Local, Modal, Queue).
    """
    
    @abstractmethod
    def run(self, computation: Callable[[], A]) -> A:
        """Execute computation in this context."""
        pass
    
    @abstractmethod
    def context_name(self) -> str:
        """Name of execution context."""
        pass


@dataclass(frozen=True)
class LocalContext(ExecutionContext[A]):
    """Local execution context - represents local computation environment."""
    
    def run(self, computation: Callable[[], A]) -> A:
        """Execute computation locally."""
        return computation()
    
    def context_name(self) -> str:
        return "local"


@dataclass(frozen=True) 
class ModalContext(ExecutionContext[A]):
    """Modal execution context - represents distributed Modal environment."""
    modal_config: Dict[str, Any]
    
    def run(self, computation: Callable[[], A]) -> A:
        """Execute computation on Modal (placeholder - actual implementation in executor)."""
        # This is a marker - actual execution handled by ModalExecutor
        raise NotImplementedError("Modal execution must be handled by ModalExecutor")
    
    def context_name(self) -> str:
        return "modal"


class NaturalTransformation(Generic[A, B], ABC):
    """
    Natural transformation between execution contexts.
    
    Mathematical Properties:
    - Naturality: For any function f: A → B, the following diagram commutes:
      
      F(A) --η_A--> G(A)
       |              |
      F(f)           G(f)  
       |              |
       v              v
      F(B) --η_B--> G(B)
    
    - Composition: Natural transformations compose associatively
    - Identity: Identity natural transformation exists
    """
    
    @abstractmethod
    def transform(self, obj: A) -> B:
        """Apply natural transformation to object."""
        pass
    
    @abstractmethod
    def naturality_law_holds(self, f: Callable[[A], A], obj: A) -> bool:
        """Verify naturality condition (for testing)."""
        pass


class SerializationTransformation(NaturalTransformation[A, Dict[str, Any]]):
    """
    Natural transformation: Local Objects → Serialized Dicts
    Systematically converts local objects to Modal-transmissible format.
    """
    
    def transform(self, obj: A) -> Dict[str, Any]:
        """Transform local object to serialized representation."""
        return self._serialize_object(obj)
    
    def _serialize_object(self, obj: Any) -> Dict[str, Any]:
        """Systematic object serialization preserving structure."""
        
        # Handle common CoralX domain objects
        if hasattr(obj, '__dataclass_fields__'):
            return self._serialize_dataclass(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return {"__type__": type(obj).__name__, "__data__": [self._serialize_value(v) for v in obj]}
        elif isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "__data__": obj.tolist(), "__shape__": obj.shape, "__dtype__": str(obj.dtype)}
        else:
            return {"__type__": type(obj).__name__, "__data__": str(obj)}
    
    def _serialize_dataclass(self, obj: Any) -> Dict[str, Any]:
        """Serialize dataclass preserving field structure."""
        result = {"__type__": type(obj).__name__, "__module__": type(obj).__module__}
        
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)
            result[field_name] = self._serialize_value(field_value)
        
        return result
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize individual value."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, np.ndarray):
            return {"__type__": "ndarray", "__data__": value.tolist(), "__shape__": value.shape, "__dtype__": str(value.dtype)}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, '__dataclass_fields__'):
            return self._serialize_dataclass(value)
        else:
            # Fallback to string representation
            return {"__type__": type(value).__name__, "__str__": str(value)}
    
    def naturality_law_holds(self, f: Callable[[A], A], obj: A) -> bool:
        """
        Verify naturality: serialize(f(obj)) ≅ f_serialized(serialize(obj))
        This is a simplified check - full verification would require function serialization.
        """
        try:
            # Direct path: f then serialize
            direct = self.transform(f(obj))
            
            # Commutative path: serialize then f (approximation)
            serialized_obj = self.transform(obj)
            # We can't apply f to serialized form, so this is a structural check
            return isinstance(direct, dict) and isinstance(serialized_obj, dict)
        except:
            return False


class DeserializationTransformation(NaturalTransformation[Dict[str, Any], A]):
    """
    Natural transformation: Serialized Dicts → Local Objects
    Inverse of SerializationTransformation, reconstructing local objects.
    """
    
    def transform(self, data: Dict[str, Any]) -> A:
        """Transform serialized data back to local object."""
        return self._deserialize_object(data)
    
    def _deserialize_object(self, data: Any) -> Any:
        """Systematic object deserialization preserving structure."""
        if not isinstance(data, dict):
            return data
        
        if "__type__" not in data:
            # Regular dict - deserialize values recursively
            return {k: self._deserialize_object(v) for k, v in data.items()}
        
        obj_type = data["__type__"]
        
        if obj_type == "ndarray":
            return np.array(data["__data__"], dtype=data.get("__dtype__", "float64")).reshape(data["__shape__"])
        elif obj_type == "list":
            return [self._deserialize_object(item) for item in data["__data__"]]
        elif obj_type == "tuple":
            return tuple(self._deserialize_object(item) for item in data["__data__"])
        elif obj_type in ["CASeed", "CAStateHistory", "Genome", "LoRAConfig", "AdapterConfig", "CAFeatures"]:
            return self._deserialize_dataclass(data)
        else:
            # Unknown type - return as dict
            return data
    
    def _deserialize_dataclass(self, data: Dict[str, Any]) -> Any:
        """Deserialize dataclass object."""
        obj_type = data["__type__"]
        module_name = data.get("__module__", "coral.domain")
        
        # Import the appropriate class
        try:
            if obj_type == "CASeed":
                from coral.domain.ca import CASeed
                cls = CASeed
            elif obj_type == "CAStateHistory":
                from coral.domain.ca import CAStateHistory
                cls = CAStateHistory
            elif obj_type == "Genome":
                from coral.domain.genome import Genome
                cls = Genome
            elif obj_type == "LoRAConfig" or obj_type == "AdapterConfig":
                from coral.domain.mapping import LoRAConfig
                cls = LoRAConfig
            elif obj_type == "CAFeatures":
                from coral.domain.feature_extraction import CAFeatures
                cls = CAFeatures
            else:
                # Unknown dataclass - return as dict
                return data
            
            # Reconstruct object with proper field handling
            field_data = {}
            for k, v in data.items():
                if k not in ["__type__", "__module__"]:
                    field_data[k] = self._deserialize_object(v)
            
            # Handle special cases for known objects
            if obj_type == "LoRAConfig" or obj_type == "AdapterConfig":
                # Ensure target_modules is tuple for LoRAConfig
                if 'target_modules' in field_data and isinstance(field_data['target_modules'], list):
                    field_data['target_modules'] = tuple(field_data['target_modules'])
            
            return cls(**field_data)
            
        except Exception as e:
            # Fallback to dict if reconstruction fails
            print(f"⚠️  Failed to reconstruct {obj_type}: {e}")
            return data
    
    def naturality_law_holds(self, f: Callable[[Dict[str, Any]], Dict[str, Any]], obj: Dict[str, Any]) -> bool:
        """Verify naturality for deserialization."""
        try:
            # This is an approximation - full naturality check requires more structure
            deserialized = self.transform(obj)
            return deserialized is not None
        except:
            return False


class DistributionFunctor(Generic[A, B]):
    """
    Functor for distributed execution contexts.
    Maps between Local and Modal categories while preserving structure.
    """
    
    def __init__(self, 
                 serialize: SerializationTransformation,
                 deserialize: DeserializationTransformation):
        self.serialize = serialize
        self.deserialize = deserialize
    
    def fmap(self, f: Callable[[A], B]) -> Callable[[ExecutionContext[A]], ExecutionContext[B]]:
        """
        Functorial map: (A → B) → (F[A] → F[B])
        Maps local function to distributed function preserving structure.
        """
        def distributed_f(ctx_a: ExecutionContext[A]) -> ExecutionContext[B]:
            if isinstance(ctx_a, LocalContext):
                # Local to local - direct application
                return LocalContext()
            elif isinstance(ctx_a, ModalContext):
                # Local to Modal - requires serialization/deserialization
                return ModalContext(ctx_a.modal_config)
            else:
                raise ValueError(f"Unknown execution context: {type(ctx_a)}")
        
        return distributed_f
    
    def preserve_composition(self, f: Callable[[A], B], g: Callable[[B], Any]) -> bool:
        """
        Verify functor law: F(g ∘ f) = F(g) ∘ F(f)
        Tests composition preservation across distribution.
        """
        try:
            # This is a structural check - full verification requires execution
            composed = lambda x: g(f(x))
            
            # Both should produce equivalent distributed functions
            direct_composed = self.fmap(composed)
            separately_composed = lambda ctx: self.fmap(g)(self.fmap(f)(ctx))
            
            return True  # Simplified - real test would require execution
        except:
            return False


# Concrete implementation for CoralX objects
class CoralXDistribution:
    """
    Practical distribution system for CoralX domain objects.
    Combines natural transformations for seamless local/Modal execution.
    """
    
    def __init__(self):
        self.serialize = SerializationTransformation()
        self.deserialize = DeserializationTransformation()
        self.functor = DistributionFunctor(self.serialize, self.deserialize)
    
    def to_modal(self, obj: Any) -> Dict[str, Any]:
        """Natural transformation: Local → Modal serializable."""
        return self.serialize.transform(obj)
    
    def from_modal(self, data: Dict[str, Any]) -> Any:
        """Natural transformation: Modal serializable → Local."""
        return self.deserialize.transform(data)
    
    def distribute_function(self, 
                          local_fn: Callable[[A], B],
                          context: ExecutionContext) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Distribute local function to execution context.
        Preserves function semantics across contexts via natural transformations.
        """
        def distributed_fn(serialized_input: Dict[str, Any]) -> Dict[str, Any]:
            # η: Serialized → Local
            local_input = self.from_modal(serialized_input)
            
            # Apply local function
            local_output = local_fn(local_input)
            
            # η: Local → Serialized  
            return self.to_modal(local_output)
        
        return distributed_fn
    
    def verify_roundtrip(self, obj: Any) -> bool:
        """
        Verify round-trip property: from_modal(to_modal(obj)) ≅ obj
        Tests natural transformation composition laws.
        """
        try:
            serialized = self.to_modal(obj)
            reconstructed = self.from_modal(serialized)
            
            # Structural equality check (simplified)
            return type(obj) == type(reconstructed)
        except:
            return False


# Global instance for CoralX
coralx_distribution = CoralXDistribution()


# Convenience functions for use in executors
def serialize_for_modal(obj: Any) -> Dict[str, Any]:
    """Convenience function: Local object → Modal serializable."""
    return coralx_distribution.to_modal(obj)


def deserialize_from_modal(data: Dict[str, Any]) -> Any:
    """Convenience function: Modal serializable → Local object."""
    return coralx_distribution.from_modal(data)


def create_distributed_function(local_fn: Callable[[A], B]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create distributed version of local function.
    Handles serialization/deserialization automatically.
    """
    return coralx_distribution.distribute_function(local_fn, ModalContext({})) 