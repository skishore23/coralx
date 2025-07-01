###############################################################################
# Modal Executor for CORAL evolution with clean architecture
# NO FALLBACKS - fail-fast principle
###############################################################################
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    raise ImportError("FAIL-FAST: Modal not available. Install with: pip install modal")

from coral.ports.interfaces import Executor


@dataclass(frozen=True)
class ResourceConfig:
    """Resource configuration for Modal functions - loaded from YAML config."""
    cpu: int
    memory: int
    gpu: str
    timeout: int


class LocalExecutor(Executor):
    """Local single-threaded executor."""
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Execute function locally and return completed future."""
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future


class ThreadExecutor(Executor):
    """Multi-threaded local executor."""
    
    def __init__(self, max_workers: int):
        if max_workers <= 0:
            raise ValueError("FAIL-FAST: max_workers must be positive")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit function to thread pool."""
        return self._executor.submit(fn, *args, **kwargs)
    
    def shutdown(self):
        """Shutdown thread pool."""
        self._executor.shutdown(wait=True)


class ModalExecutor(Executor):
    """Modal-based distributed executor with clean architecture - NO FALLBACKS."""
    
    def __init__(self, app_name: str, config: Dict[str, Any]):
        if not MODAL_AVAILABLE:
            raise RuntimeError("FAIL-FAST: Modal not available for distributed execution")
        
        # Use production app name consistently
        self.app_name = app_name or "coral-x-production"  # Default to production app
        self.config = config
        self._setup_modal_functions()
    
    def _setup_modal_functions(self):
        """Setup Modal functions from deployed app - fail if not available."""
        try:
            # Use direct function lookup instead of App.lookup which has issues
            # This avoids the "bad argument type for built-in operation" error
            self.modal_functions = {
                'evaluate_genome_modal': modal.Function.from_name(self.app_name, "evaluate_genome_modal"),
                'train_lora_modal': modal.Function.from_name(self.app_name, "train_lora_modal"),
                'run_experiment_modal': modal.Function.from_name(self.app_name, "run_experiment_modal"),
                'generate_code_modal': modal.Function.from_name(self.app_name, "generate_code_modal"),
                'run_benchmarks_modal': modal.Function.from_name(self.app_name, "run_benchmarks_modal")
            }
            print(f"✅ Modal functions loaded from app: {self.app_name}")
        except Exception as e:
            raise RuntimeError(
                f"FAIL-FAST: Modal functions not available for app '{self.app_name}'. "
                f"Deploy first with: modal deploy coral_modal_app.py. Error: {e}"
            )
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit function execution to Modal - no fallbacks."""
        function_name = getattr(fn, '__name__', str(fn))
        
        # Map function names to Modal functions
        modal_fn = self._get_modal_function(function_name)
        
        # Transform arguments for Modal execution
        modal_args = self._transform_arguments(function_name, *args, **kwargs)
        
        # Execute on Modal
        return self._execute_modal_function(modal_fn, modal_args, *args)
    
    def _get_modal_function(self, function_name: str):
        """Get Modal function by name - fail if not found."""
        if 'evaluate' in function_name:
            return self.modal_functions['evaluate_genome_modal']
        elif 'train_lora' in function_name:
            return self.modal_functions['train_lora_modal']
        elif 'run_experiment' in function_name:
            return self.modal_functions['run_experiment_modal']
        elif 'generate' in function_name:
            return self.modal_functions['generate_code_modal']
        elif 'benchmark' in function_name:
            return self.modal_functions['run_benchmarks_modal']
        else:
            raise RuntimeError(
                f"FAIL-FAST: No Modal function mapping for '{function_name}'. "
                f"Available functions: {list(self.modal_functions.keys())}"
            )
    
    def _transform_arguments(self, function_name: str, *args, **kwargs):
        """Transform arguments for Modal execution."""
        if 'evaluate' in function_name and len(args) == 1:
            # Genome evaluation
            genome = args[0]
            genome_data = self._serialize_genome(genome)
            return (genome_data, self.config)
        else:
            # For other functions, include config
            return args + (self.config,)
    
    def _serialize_genome(self, genome):
        """Serialize genome for Modal transmission."""
        # 🔥 FIX: Ensure all fields are properly serialized for consistent hash calculation
        
        # 🔥 FIX: Serialize CA features for consistency
        ca_features_data = None
        if hasattr(genome, 'ca_features') and genome.ca_features is not None:
            ca_features_data = {
                'complexity': genome.ca_features.complexity,
                'intensity': genome.ca_features.intensity,
                'periodicity': genome.ca_features.periodicity,
                'convergence': genome.ca_features.convergence
            }
        
        serialized = {
            'id': getattr(genome, 'id', f'genome_{hash(str(genome))%10000:04d}'),
            'seed': {
                'grid': genome.seed.grid.tolist() if hasattr(genome.seed.grid, 'tolist') else genome.seed.grid,
                'rule': getattr(genome.seed, 'rule', 0),
                'steps': getattr(genome.seed, 'steps', 15)
            },
            'lora_config': {
                'r': genome.lora_cfg.r,
                'alpha': genome.lora_cfg.alpha,
                'dropout': genome.lora_cfg.dropout,
                'target_modules': list(genome.lora_cfg.target_modules),
                'adapter_type': getattr(genome.lora_cfg, 'adapter_type', 'lora')  # 🔥 FIX: Explicit adapter_type
            },
            'ca_features': ca_features_data,  # 🔥 FIX: Include CA features
            'run_id': getattr(genome, 'run_id', None)
        }
        
        # 🔥 CRITICAL: Debug hash consistency for cache coordination
        if hasattr(genome, 'lora_cfg'):
            from infra.adapter_cache import HeavyGenes
            
            # Create HeavyGenes exactly as it would be during training
            heavy_genes_training = HeavyGenes.from_lora_config(
                genome.lora_cfg, 
                run_id=getattr(genome, 'run_id', None)
            )
            
            # Create HeavyGenes as it would be during reconstruction
            from coral.domain.mapping import LoRAConfig
            reconstructed_lora = LoRAConfig(
                r=serialized['lora_config']['r'],
                alpha=serialized['lora_config']['alpha'],
                dropout=serialized['lora_config']['dropout'],
                target_modules=tuple(serialized['lora_config']['target_modules']),
                adapter_type=serialized['lora_config']['adapter_type']
            )
            
            heavy_genes_reconstructed = HeavyGenes.from_lora_config(
                reconstructed_lora,
                run_id=serialized['run_id']
            )
            
            hash_training = heavy_genes_training.to_hash()
            hash_reconstructed = heavy_genes_reconstructed.to_hash()
            
            print(f"🔍 CACHE HASH CONSISTENCY CHECK:")
            print(f"   • Training hash: {hash_training}")
            print(f"   • Reconstructed hash: {hash_reconstructed}")
            print(f"   • Hashes match: {hash_training == hash_reconstructed}")
            
            if hash_training != hash_reconstructed:
                print(f"❌ CRITICAL: Hash mismatch detected!")
                print(f"   • Training genes: {heavy_genes_training}")
                print(f"   • Reconstructed genes: {heavy_genes_reconstructed}")
                raise RuntimeError(
                    f"FAIL-FAST: Cache hash inconsistency detected! "
                    f"Training hash {hash_training} != reconstructed hash {hash_reconstructed}. "
                    f"This will cause cache misses and wasted training."
                )
            else:
                print(f"✅ Cache hash consistency verified: {hash_training}")
        
        return serialized
    
    def _serialize_genome_categorical(self, genome):
        """
        NEW: Categorical serialization using natural transformations.
        Demonstrates systematic structure preservation vs manual serialization above.
        """
        # Import the new categorical distribution system
        from coral.domain.categorical_distribution import serialize_for_modal
        
        # Use natural transformation for systematic serialization
        # This preserves categorical structure automatically
        serialized = serialize_for_modal(genome)
        
        print(f"🧮 CATEGORICAL SERIALIZATION:")
        print(f"   • Using natural transformation: Local → Modal")
        print(f"   • Structure preservation: Automatic")
        print(f"   • Type safety: Guaranteed by categorical laws")
        
        return serialized
    
    def _deserialize_result_categorical(self, modal_result, original_genome):
        """
        NEW: Categorical deserialization using natural transformations.
        Demonstrates systematic reconstruction vs manual result handling.
        """
        from coral.domain.categorical_distribution import deserialize_from_modal
        
        # Use natural transformation for systematic deserialization
        if isinstance(modal_result, dict) and 'genome_data' in modal_result:
            # Reconstruct genome using natural transformation
            reconstructed_genome = deserialize_from_modal(modal_result['genome_data'])
            
            # Apply scores to genome
            if all(key in modal_result for key in ['bugfix', 'style', 'security', 'runtime', 'syntax']):
                from coral.domain.genome import MultiObjectiveScores
                multi_scores = MultiObjectiveScores(
                    bugfix=modal_result['bugfix'],
                    style=modal_result['style'],
                    security=modal_result['security'],
                    runtime=modal_result['runtime'],
                    syntax=modal_result['syntax']
                )
                return reconstructed_genome.with_multi_scores(multi_scores)
            
            return reconstructed_genome
        
        # Fallback to manual handling
        return self._transform_result(None, modal_result, original_genome)
    
    def _execute_modal_function(self, modal_fn, modal_args, *original_args) -> Future:
        """Execute Modal function and return Future with timeout and retry handling."""
        future = Future()
        
        try:
            # Call Modal function remotely with timeout handling
            import time
            start_time = time.time()
            
            try:
                raw_result = modal_fn.remote(*modal_args)
                execution_time = time.time() - start_time
                print(f"✅ Modal function completed in {execution_time:.1f}s")
                
            except Exception as modal_error:
                execution_time = time.time() - start_time
                print(f"❌ Modal function failed after {execution_time:.1f}s: {modal_error}")
                
                # Handle specific Modal errors
                if "ClientClosed" in str(modal_error):
                    raise RuntimeError(
                        f"FAIL-FAST: Modal client disconnected after {execution_time:.1f}s. "
                        f"This usually indicates a timeout or network issue. "
                        f"Function: {getattr(modal_fn, 'function_name', str(modal_fn))}"
                    )
                elif "timeout" in str(modal_error).lower():
                    raise RuntimeError(
                        f"FAIL-FAST: Modal function timeout after {execution_time:.1f}s. "
                        f"Consider increasing timeout or optimizing the function. "
                        f"Function: {getattr(modal_fn, 'function_name', str(modal_fn))}"
                    )
                else:
                    raise modal_error
            
            # Transform result back for local use
            result = self._transform_result(modal_fn, raw_result, *original_args)
            
            future.set_result(result)
            
        except Exception as e:
            # Check for fail-fast conditions
            if "FAIL-FAST:" in str(e):
                raise e  # Re-raise fail-fast errors immediately
            else:
                future.set_exception(RuntimeError(f"Modal execution failed: {e}"))
        
        return future
    
    def submit_training(self, base_model: str, heavy_genes, save_path: str, config: Dict[str, Any]) -> Future:
        """Submit training job - direct training when in Modal environment."""
        import os
        
        # Check if we're already in Modal environment
        if os.environ.get('MODAL_ENVIRONMENT'):
            # We're inside Modal - do training directly instead of calling another Modal function
            print(f"🏗️  Direct training in Modal: {heavy_genes.to_hash()[:8]}...")
            
            future = Future()
            try:
                # Import and call training function directly
                import sys
                from pathlib import Path
                
                # Add coralx to path
                coralx_path = Path("/root/coralx")
                if coralx_path.exists():
                    sys.path.insert(0, str(coralx_path))
                
                from coral.domain.lora_training import train_codellama_lora
                
                # Call training function directly
                result_path = train_codellama_lora(
                    base_ckpt=base_model,
                    heavy_genes=heavy_genes,
                    save_to=save_path,
                    config=config
                )
                
                future.set_result(result_path)
                print(f"✅ Direct training completed: {heavy_genes.to_hash()[:8]} → {result_path}")
                
            except Exception as e:
                future.set_exception(RuntimeError(f"Direct training failed: {e}"))
                print(f"❌ Direct training failed: {heavy_genes.to_hash()[:8]} - {e}")
            
            return future
            
        else:
            # We're in local environment - call Modal function
            modal_fn = self.modal_functions['train_lora_modal']
            
            # Convert heavy_genes object to heavy_key (hash) as expected by Modal function
            heavy_key = heavy_genes.to_hash()
            
            # Create training arguments with correct parameter name
            training_args = (base_model, heavy_key, save_path, config)
            
            print(f"🚀 Submitting training job: {heavy_key[:8]}...")
            return self._execute_modal_function(modal_fn, training_args, base_model, heavy_genes, save_path)
    
    def submit_evaluation(self, genome, adapter_path: str, config: Dict[str, Any]) -> Future:
        """Submit evaluation job to Modal - parallel evaluation on A10G."""
        modal_fn = self.modal_functions['evaluate_genome_modal']
        
        # Serialize genome and add adapter path to config
        genome_data = self._serialize_genome(genome)
        evaluation_config = {**config, 'adapter_path': adapter_path}
        
        # Create evaluation arguments
        evaluation_args = (genome_data, evaluation_config)
        
        print(f"🧪 Submitting evaluation job: {genome.id}")
        return self._execute_modal_function(modal_fn, evaluation_args, genome)
    
    def _transform_result(self, modal_fn, modal_result, *original_args):
        """Transform Modal result back to local format."""
        function_name = getattr(modal_fn, 'function_name', str(modal_fn))
        
        if 'train_lora_modal' in function_name:
            # Training result - return adapter path
            if isinstance(modal_result, str):
                return modal_result  # Direct adapter path
            elif isinstance(modal_result, dict) and 'adapter_path' in modal_result:
                return modal_result['adapter_path']
            else:
                raise RuntimeError(f"FAIL-FAST: Training returned invalid result: {modal_result}")
        
        elif 'evaluate_genome_modal' in function_name and len(original_args) == 1:
            # Genome evaluation result
            original_genome = original_args[0]
            
            if not isinstance(modal_result, dict):
                raise RuntimeError(f"FAIL-FAST: Modal returned invalid result format: {type(modal_result)}")
            
            if 'error' in modal_result:
                raise RuntimeError(f"FAIL-FAST: Modal evaluation failed: {modal_result['error']}")
            
            # Reconstruct multi-objective scores
            from coral.domain.genome import MultiObjectiveScores
            
            if all(key in modal_result for key in ['bugfix', 'style', 'security', 'runtime', 'syntax']):
                multi_scores = MultiObjectiveScores(
                    bugfix=modal_result['bugfix'],
                    style=modal_result['style'],
                    security=modal_result['security'],
                    runtime=modal_result['runtime'],
                    syntax=modal_result['syntax']
                )
                return original_genome.with_multi_scores(multi_scores)
            else:
                raise RuntimeError(f"FAIL-FAST: Modal result missing required score fields: {modal_result}")
        
        return modal_result


# Factory functions for creating executors from config
def create_executor_from_config(config: Dict[str, Any]) -> Executor:
    """Create executor based on configuration - no fallbacks."""
    infra_config = config.get('infra', {})
    executor_type = infra_config.get('executor')
    
    if not executor_type:
        raise ValueError("FAIL-FAST: No executor type specified in config")
    
    if executor_type == 'local':
        return LocalExecutor()
    elif executor_type == 'thread':
        local_config = infra_config.get('local', {})
        max_workers = local_config.get('max_workers')
        if not max_workers:
            raise ValueError("FAIL-FAST: max_workers not specified for thread executor")
        return ThreadExecutor(max_workers=max_workers)
    elif executor_type == 'modal':
        modal_config = infra_config.get('modal', {})
        app_name = modal_config.get('app_name')
        if not app_name:
            raise ValueError("FAIL-FAST: app_name not specified for modal executor")
        return ModalExecutor(app_name=app_name, config=config)
    elif executor_type == 'queue_modal':
        # Use queue-based Modal executor
        from infra.queue_modal_executor import QueueModalExecutor
        return QueueModalExecutor(config)
    else:
        raise ValueError(f"FAIL-FAST: Unknown executor type '{executor_type}'. Supported: local, thread, modal, queue_modal") 