###############################################################################
# Modal Executor for CORAL evolution with clean architecture
###############################################################################
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict
from dataclasses import dataclass

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    raise ImportError("  Modal not available. Install with: pip install modal")

from core.ports.interfaces import Executor
from core.domain.genome import MultiObjectiveScores


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

    def submit_training(self, base_model: str, heavy_genes, save_path: str, config: Dict[str, Any]) -> Future:
        """Submit LoRA training job for local execution."""
        future = Future()
        try:
            print(f"   🏗️  Local LoRA training: {heavy_genes.to_hash()[:8]}...")
            print(f"      Model: {base_model}")
            print(f"      LoRA config: r={heavy_genes.rank}, α={heavy_genes.alpha}, dropout={heavy_genes.dropout}")

            # Import training function
            from core.domain.lora_training import train_lora_adapter_local

            # Train the adapter locally
            result_path = train_lora_adapter_local(
                base_model=base_model,
                heavy_genes=heavy_genes,
                save_path=save_path,
                config=config
            )

            future.set_result(result_path)
        except Exception as e:
            print(f"   ❌ Local training failed: {str(e)}")
            future.set_exception(e)
        return future

    def submit_evaluation(self, genome, adapter_path, config) -> Future:
        """Submit evaluation job for local execution."""
        future = Future()
        try:
            print(f"   🧪 Local evaluation: {genome.id[:8]}...")
            print(f"      Adapter: {adapter_path}")

            # Use plugin-based evaluation (model-agnostic)
            # This should be handled by the fitness function, not hardcoded here
            # For now, return a placeholder that should be replaced by proper plugin evaluation
            scores = genome.scores or MultiObjectiveScores(
                bugfix=0.5, style=0.5, security=0.5, runtime=0.5, syntax=0.5
            )

            # Create updated genome with evaluation results
            # The system expects an evaluated genome, not just scores
            evaluated_genome = genome.with_multi_scores(scores)

            future.set_result(evaluated_genome)
        except Exception as e:
            print(f"   ❌ Local evaluation failed: {str(e)}")
            future.set_exception(e)
        return future


class ThreadExecutor(Executor):
    """Multi-threaded local executor."""

    def __init__(self, max_workers: int):
        if max_workers <= 0:
            raise ValueError("  max_workers must be positive")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit function to thread pool."""
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self):
        """Shutdown thread pool."""
        self._executor.shutdown(wait=True)


class ModalExecutor(Executor):
    """Modal-based distributed executor with clean architecture."""

    def __init__(self, app_name: str, config: Dict[str, Any]):
        if not MODAL_AVAILABLE:
            raise RuntimeError("  Modal not available for distributed execution")

        # Use production app name consistently
        self.app_name = app_name or "coral-x-production"  # Default to production app
        self.config = config
        self._setup_modal_functions()

    def _setup_modal_functions(self):
        """Setup Modal functions from deployed app."""
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
                f"  Modal functions not available for app '{self.app_name}'. "
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
        """Get Modal function by name."""
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
                f"  No Modal function mapping for '{function_name}'. "
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
            from core.domain.mapping import LoRAConfig
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

            print("🔍 CACHE HASH CONSISTENCY CHECK:")
            print(f"   • Training hash: {hash_training}")
            print(f"   • Reconstructed hash: {hash_reconstructed}")
            print(f"   • Hashes match: {hash_training == hash_reconstructed}")

            if hash_training != hash_reconstructed:
                print("❌ CRITICAL: Hash mismatch detected!")
                print(f"   • Training genes: {heavy_genes_training}")
                print(f"   • Reconstructed genes: {heavy_genes_reconstructed}")
                raise RuntimeError(
                    f"  Cache hash inconsistency detected! "
                    f"Training hash {hash_training} != reconstructed hash {hash_reconstructed}. "
                    f"This will cause cache misses and wasted training."
                )
            else:
                print(f"✅ Cache hash consistency verified: {hash_training}")

        return serialized


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

                raise modal_error

            # Transform result back for local use
            result = self._transform_result(modal_fn, raw_result, *original_args)

            future.set_result(result)

        except Exception as e:
            # Check for configuration errors
            if " " in str(e):
                raise e  # Re-raise configuration errors immediately
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

                from core.domain.lora_training import train_codellama_lora

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
                raise RuntimeError(f"  Training returned invalid result: {modal_result}")

        elif 'evaluate_genome_modal' in function_name and len(original_args) == 1:
            # Genome evaluation result
            original_genome = original_args[0]

            if not isinstance(modal_result, dict):
                raise RuntimeError(f"  Modal returned invalid result format: {type(modal_result)}")

            if 'error' in modal_result:
                raise RuntimeError(f"  Modal evaluation failed: {modal_result['error']}")

            # Reconstruct multi-objective scores
            from core.domain.genome import MultiObjectiveScores

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
                raise RuntimeError(f"  Modal result missing required score fields: {modal_result}")

        return modal_result


# Factory functions for creating executors from config
def create_executor_from_config(config: Dict[str, Any]) -> Executor:
    """Create executor based on configuration."""
    infra_config = config.get('infra', {})
    executor_type = infra_config.get('executor')

    if not executor_type:
        raise ValueError("  No executor type specified in config")

    if executor_type == 'local':
        return LocalExecutor()
    elif executor_type == 'thread':
        local_config = infra_config.get('local', {})
        max_workers = local_config.get('max_workers')
        if not max_workers:
            raise ValueError("  max_workers not specified for thread executor")
        return ThreadExecutor(max_workers=max_workers)
    elif executor_type == 'modal':
        modal_config = infra_config.get('modal', {})
        app_name = modal_config.get('app_name')
        if not app_name:
            raise ValueError("  app_name not specified for modal executor")
        return ModalExecutor(app_name=app_name, config=config)
    elif executor_type == 'queue_modal':
        from infra.queue_modal_executor import QueueModalExecutor
        return QueueModalExecutor(config)
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")
