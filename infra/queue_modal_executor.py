"""
Queue-Based Modal Executor
Clean architecture using Modal Queues for coordination
Based on CORAL_X_QUEUE_REFACTORING_PLAN.md
"""

import modal
import time
import uuid
from concurrent.futures import Future
from typing import Any, Callable, Dict
from dataclasses import dataclass

from core.ports.interfaces import Executor


@dataclass
class QueueJob:
    """Queue job structure."""
    job_id: str
    job_type: str
    parameters: Dict[str, Any]
    timestamp: float


class QueueModalExecutor(Executor):
    """
    Category theory compliant executor via Modal queues.
    Implements proper natural transformations η: Local → Queue → Result.
    
    Mathematical Properties:
    - Functoriality: Queue[g ∘ f] = Queue[g] ∘ Queue[f]
    - Natural Transformations: Preserves categorical structure  
    - Composition Laws: Associativity guaranteed by queue operations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app_name = config.get('infra', {}).get('modal', {}).get('app_name', 'coral-x-queues')

        # 🧮 CATEGORY THEORY: Reference the SAME global queue objects as workers
        # These MUST match the queue names in coral_queue_modal_app.py
        self.training_queue = modal.Queue.from_name("coral-training", create_if_missing=True)
        self.test_queue = modal.Queue.from_name("coral-test", create_if_missing=True)
        self.generation_queue = modal.Queue.from_name("coral-generation", create_if_missing=True)
        self.results_queue = modal.Queue.from_name("coral-results", create_if_missing=True)

        # Cache as categorical limit object (matches workers)
        self.cache_index = modal.Dict.from_name("coral-cache-index", create_if_missing=True)

        # Job tracking for result collection
        self.pending_jobs: Dict[str, Dict[str, Any]] = {}

        print("🧮 QueueModalExecutor initialized with category theory compliance")
        print(f"   App: {self.app_name}")
        print("   Global queues: coral-training, coral-test, coral-generation, coral-results")

    def start_workers(self, training_workers: int = 2, test_workers: int = 3):
        """
        Start category theory compliant workers that reference global queues.
        Natural transformation: Workers automatically find global queue objects.
        """
        try:
            # Start training workers (natural functors for training category)
            training_fn = modal.Function.from_name(self.app_name, "training_worker")
            for i in range(training_workers):
                training_fn.spawn()
                print(f"🏗️ Started training worker {i+1}/{training_workers}")

            # Start test workers (natural functors for evaluation category)
            test_fn = modal.Function.from_name(self.app_name, "test_worker")
            for i in range(test_workers):
                test_fn.spawn()
                print(f"🧪 Started test worker {i+1}/{test_workers}")

            print(f"✅ Started {training_workers + test_workers} category theory compliant workers")

        except Exception as e:
            print(f"⚠️  Failed to start workers: {e}")
            print("Workers can be started manually if needed")

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit function execution to appropriate queue."""
        function_name = getattr(fn, '__name__', str(fn))

        if 'train' in function_name.lower():
            return self._submit_training_job(*args, **kwargs)
        elif 'generate' in function_name.lower():
            return self._submit_generation_job(*args, **kwargs)
        elif 'evaluate' in function_name.lower():
            return self._submit_test_job(*args, **kwargs)
        else:
            raise RuntimeError(f"  Unknown function type for queue routing: {function_name}")

    def submit_training(self, base_model: str, heavy_genes, save_path: str, config: Dict[str, Any]) -> Future:
        """Submit training job to training queue."""
        job_id = f"train_{uuid.uuid4().hex[:8]}"

        # Create training job
        job = {
            'job_id': job_id,
            'job_type': 'training',
            'base_model': base_model,
            'heavy_genes': heavy_genes,
            'save_path': save_path,
            'config': config,
            'timestamp': time.time()
        }

        # Submit to training queue
        self.training_queue.put(job)
        print(f"🚀 Submitted training job to queue: {job_id}")

        # Create future for result
        future = Future()
        self.pending_jobs[job_id] = {'future': future}

        # Start result collector if not already running
        self._ensure_result_collector_running()

        return future

    def submit_evaluation(self, genome, adapter_path: str, config: Dict[str, Any]) -> Future:
        """Submit evaluation job to test queue."""
        job_id = f"eval_{uuid.uuid4().hex[:8]}"

        # Serialize genome
        genome_data = self._serialize_genome(genome)

        # Create evaluation job
        job = {
            'job_id': job_id,
            'job_type': 'evaluation',
            'genome_data': genome_data,
            'adapter_path': adapter_path,
            'config': config,
            'timestamp': time.time()
        }

        # Submit to test queue
        self.test_queue.put(job)
        print(f"🧪 Submitted evaluation job to queue: {job_id}")
        print(f"📊 Pending jobs count: {len(self.pending_jobs)} → {len(self.pending_jobs) + 1}")

        # Create future for result
        future = Future()
        self.pending_jobs[job_id] = {'future': future}

        # Start result collector if not already running
        self._ensure_result_collector_running()

        return future

    def _submit_training_job(self, *args, **kwargs) -> Future:
        """Route training job submission."""
        if len(args) >= 4:
            return self.submit_training(args[0], args[1], args[2], args[3])
        else:
            raise RuntimeError("  Invalid training job arguments")

    def _submit_generation_job(self, *args, **kwargs) -> Future:
        """Submit generation job to generation queue."""
        job_id = f"gen_{uuid.uuid4().hex[:8]}"

        # Create generation job
        job = {
            'job_id': job_id,
            'job_type': 'generation',
            'parameters': args,
            'config': kwargs.get('config', self.config),
            'timestamp': time.time()
        }

        # Submit to generation queue
        self.generation_queue.put(job)
        print(f"🤖 Submitted generation job to queue: {job_id}")

        # Create future for result
        future = Future()
        self.pending_jobs[job_id] = {'future': future}

        return future

    def _submit_test_job(self, *args, **kwargs) -> Future:
        """Route test job submission."""
        if len(args) >= 2:
            return self.submit_evaluation(args[0], args[1], args[2] if len(args) > 2 else self.config)
        else:
            raise RuntimeError("  Invalid test job arguments")

    def _serialize_genome(self, genome):
        """
        Serialize genome for queue transmission.
        Category theory compliant: preserves structural information.
        🔥 FIX: Now includes CA features for consistency.
        """
        # Handle both old AdapterConfig and new AdapterParameters
        lora_cfg = genome.lora_cfg

        # Extract adapter parameters (handle both naming conventions)
        rank = getattr(lora_cfg, 'rank', getattr(lora_cfg, 'r', 8))
        alpha = getattr(lora_cfg, 'alpha', 16.0)
        dropout = getattr(lora_cfg, 'dropout', 0.1)
        target_modules = getattr(lora_cfg, 'target_modules', ['q_proj', 'v_proj'])
        adapter_type = getattr(lora_cfg, 'adapter_type', 'lora')

        # 🔥 FIX: Serialize CA features for consistency
        ca_features_data = None
        if hasattr(genome, 'ca_features') and genome.ca_features is not None:
            ca_features_data = {
                'complexity': genome.ca_features.complexity,
                'intensity': genome.ca_features.intensity,
                'periodicity': genome.ca_features.periodicity,
                'convergence': genome.ca_features.convergence
            }

        return {
            'id': getattr(genome, 'id', f'genome_{hash(str(genome))%10000:04d}'),
            'seed': {
                'grid': genome.seed.grid.tolist() if hasattr(genome.seed.grid, 'tolist') else genome.seed.grid,
                'rule': getattr(genome.seed, 'rule', 0),
                'steps': getattr(genome.seed, 'steps', 15)
            },
            'lora_config': {
                'r': rank,           # Keep 'r' for backward compatibility
                'rank': rank,        # New naming convention
                'alpha': alpha,
                'dropout': dropout,
                'target_modules': list(target_modules) if isinstance(target_modules, tuple) else target_modules,
                'adapter_type': adapter_type
            },
            'ca_features': ca_features_data,  # 🔥 FIX: Include CA features
            'run_id': getattr(genome, 'run_id', None)
        }

    def _ensure_result_collector_running(self):
        """Ensure result collector is running."""
        if not hasattr(self, '_result_collector_running'):
            self._result_collector_running = True
            # Start background result collection
            import threading
            thread = threading.Thread(target=self._collect_results_continuously, daemon=True)
            thread.start()

    def _collect_results_continuously(self):
        """
        Category theory compliant result collection with orphan handling.
        Natural transformation: μ: Queue[Result] → Local[Result]
        ENHANCED: Handles orphaned results from multiple evaluation processes.
        """
        timeout = 30
        orphan_count = 0

        while True:
            try:
                # η^(-1): Natural transformation from queue to local result
                result = self.results_queue.get(timeout=timeout)
                if result is None:
                    continue

                job_id = result.get('job_id')
                job_type = result.get('job_type', 'unknown')
                print(f"📡 Natural transformation μ: Queue[{job_type}] → Local[{job_id}]")

                # F(complete_job): Functorial completion of pending job
                if job_id in self.pending_jobs:
                    job_data = self.pending_jobs[job_id]
                    future = job_data['future']

                    # Update categorical metadata
                    job_data['completed_at'] = time.time()
                    job_data['result_type'] = job_type

                    if 'error' in result:
                        # Error preserves categorical structure
                        error = RuntimeError(f"Categorical composition failed: {result['error']}")
                        future.set_exception(error)
                        print(f"❌ Composition broken: {job_id} - {result['error']}")
                    else:
                        # Successful composition: Domain → Queue → Result
                        job_result = self._process_queue_result(result)
                        future.set_result(job_result)
                        print(f"✅ Categorical composition: {job_id} (type: {job_type})")

                    # Remove from pending (composition complete)
                    del self.pending_jobs[job_id]
                else:
                    # ENHANCED: Handle orphaned results more gracefully
                    orphan_count += 1
                    print(f"⚠️  Orphaned result #{orphan_count}: {job_id}")
                    print(f"     Result type: {job_type}, Success: {result.get('result', {}).get('success', 'unknown')}")
                    print(f"     Pending jobs: {list(self.pending_jobs.keys())}")

                    # If too many orphaned results, warn about dual evaluation systems
                    if orphan_count >= 3:
                        print(f"🚨 WARNING: {orphan_count} orphaned results detected!")
                        print("     This suggests multiple evaluation processes are running.")
                        print("     Consider restarting the experiment to ensure clean state.")

                    # Log successful orphaned results for debugging
                    if result.get('result', {}).get('success', False):
                        genome_id = result.get('result', {}).get('genome_id', 'unknown')
                        print(f"     ✅ Orphaned but successful: {genome_id}")

            except Exception as e:
                if 'timeout' in str(e).lower() or 'empty' in str(e).lower():
                    continue
                elif 'cancel' in str(e).lower() or 'shutdown' in str(e).lower():
                    break
                else:
                    time.sleep(2)

    def _process_queue_result(self, result):
        """Process result from queue based on job type."""
        job_type = result.get('job_type', 'unknown')
        raw_result = result.get('result')

        if job_type == 'training':
            # Training result - return adapter path
            return raw_result

        elif job_type == 'evaluation':
            # Evaluation result - reconstruct genome with scores
            if not isinstance(raw_result, dict):
                raise RuntimeError(f"  Invalid evaluation result format: {type(raw_result)}")

            # FIX: Properly reconstruct Genome object from evaluation result
            return self._reconstruct_genome_from_result(raw_result)

        elif job_type == 'generation':
            # Generation result - return generated code
            return raw_result

        else:
            return raw_result

    def _reconstruct_genome_from_result(self, result_dict: Dict[str, Any]):
        """
        Reconstruct Genome object from Modal evaluation result.
        Category theory compliant: Inverse of _serialize_genome.
        🔥 FIX: Now reconstructs CA features for consistency.
        """
        from core.domain.genome import Genome
        from core.domain.ca import CASeed
        from core.domain.mapping import LoRAConfig
        from core.domain.genome import MultiObjectiveScores
        from core.domain.feature_extraction import CAFeatures
        import numpy as np

        # Extract genome data from result
        genome_data = result_dict.get('genome_data', {})
        scores_data = result_dict.get('scores', {})

        # Reconstruct CA seed
        seed_data = genome_data.get('seed', {})
        ca_seed = CASeed(
            grid=np.array(seed_data.get('grid', [[1, 0, 1]])),
            rule=seed_data.get('rule', 30),
            steps=seed_data.get('steps', 15)
        )

        # Reconstruct LoRA config
        lora_data = genome_data.get('lora_config', {})
        lora_config = LoRAConfig(
            r=lora_data.get('r', lora_data.get('rank', 8)),
            alpha=lora_data.get('alpha', 16.0),
            dropout=lora_data.get('dropout', 0.1),
            target_modules=tuple(lora_data.get('target_modules', ['q_proj', 'v_proj'])),
            adapter_type=lora_data.get('adapter_type', 'lora')
        )

        # 🔥 FIX: Reconstruct CA features for consistency
        ca_features = None
        ca_features_data = genome_data.get('ca_features')
        if ca_features_data is not None:
            ca_features = CAFeatures(
                complexity=ca_features_data.get('complexity', 0.5),
                intensity=ca_features_data.get('intensity', 0.5),
                periodicity=ca_features_data.get('periodicity', 0.5),
                convergence=ca_features_data.get('convergence', 0.5)
            )

        # Reconstruct multi-objective scores
        multi_scores = None
        if scores_data:
            multi_scores = MultiObjectiveScores(
                bugfix=scores_data.get('bugfix', 0.0),
                style=scores_data.get('style', 0.0),
                security=scores_data.get('security', 0.0),
                runtime=scores_data.get('runtime', 0.0),
                syntax=scores_data.get('syntax', 0.0)
            )

        # Create genome with evaluation results
        genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_config,
            id=genome_data.get('id', 'unknown'),
            ca_features=ca_features,  # 🔥 FIX: Include CA features
            fitness=scores_data.get('fitness', None),
            run_id=genome_data.get('run_id', None),
            multi_scores=multi_scores
        )

        return genome

    def get_queue_status(self):
        """Get status of all queues."""
        try:
            return {
                'training_queue': self.training_queue.len() if self.training_queue else 0,
                'test_queue': self.test_queue.len() if self.test_queue else 0,
                'generation_queue': self.generation_queue.len() if self.generation_queue else 0,
                'results_queue': self.results_queue.len() if self.results_queue else 0,
                'pending_jobs': len(self.pending_jobs),
                'timestamp': time.time()
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': time.time()}


# Factory function for creating queue-based executor
def create_queue_executor_from_config(config: Dict[str, Any]) -> QueueModalExecutor:
    """Create queue-based Modal executor."""
    return QueueModalExecutor(config)
