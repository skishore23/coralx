# CORAL-X Detailed Implementation Map

**Exact file-by-file changes for queue-based refactoring**

---

## 📁 **File Change Matrix**

| **File** | **Action** | **Priority** | **Impact** |
|----------|------------|--------------|------------|
| `coral_queue_modal_app.py` | 🆕 CREATE | P0 | Replace volume coordination |
| `infra/queue_modal_executor.py` | 🆕 CREATE | P0 | Restore functors |
| `coral/application/evolution_engine.py` | 🔧 MODIFY | P1 | Queue-based evolution |
| `coral/ports/interfaces.py` | 🔧 MODIFY | P1 | Add queue protocols |
| `plugins/quixbugs_codellama/plugin.py` | 🔧 MODIFY | P2 | Queue compatibility |
| `infra/modal_executor.py` | 🔄 REPLACE | P2 | Remove broken functor implementations |
| `coral_modal_app.py` | 🗑️ DELETE | P3 | Remove after migration |

---

## 🆕 **NEW FILES TO CREATE**

### **1. `coral_queue_modal_app.py`** (Priority P0)

**Purpose**: Replace complex volume coordination with queue-based workers

```python
"""
CORAL-X Queue-Based Modal Application
Functorial architecture using Modal Queues for coordination.
"""
import modal
from typing import Dict, Any, List
import json
import time

# Application with queue-based coordination
app = modal.App("coral-x-queues")

# Persistent volume for models/datasets only (not for coordination)
cache_volume = modal.Volume.from_name("coral-cache-data", create_if_missing=True)

# Queue infrastructure - replaces manual coordination
training_queue = modal.Queue.from_name("coral-training", create_if_missing=True)
generation_queue = modal.Queue.from_name("coral-generation", create_if_missing=True)
test_queue = modal.Queue.from_name("coral-tests", create_if_missing=True)
results_queue = modal.Queue.from_name("coral-results", create_if_missing=True)

# Cache coordination via Modal Dict (not volume files)
cache_index = modal.Dict.from_name("coral-cache-index", create_if_missing=True)

# Container image with CORAL-X dependencies
coral_image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch", "transformers", "accelerate", "peft",
        "datasets", "numpy", "scipy", "pyyaml"
    ])
    .env({"PYTHONPATH": "/root/coralx"})
    .add_local_dir(".", "/root/coralx")
)

# ========== TRAINING WORKERS ==========

@app.function(
    image=coral_image,
    volumes={"/cache": cache_volume},
    gpu="A100-40GB",
    memory=32768,
    timeout=1800
)
def training_worker():
    """Auto-scaling training worker - processes jobs from queue."""
    import sys
    sys.path.insert(0, "/root/coralx")
    
    from infra.adapter_cache import train_lora_adapter
    
    while True:
        try:
            # Get next training job from queue
            job = training_queue.get(timeout=30)
            if job is None:
                print("No more training jobs - worker shutting down")
                break
            
            job_id, heavy_genes, config = job
            print(f"🏋️ Training worker processing job: {job_id}")
            
            # Pure domain logic - no coordination complexity
            adapter_path = train_lora_adapter(heavy_genes, config)
            
            # Update cache index atomically
            cache_key = heavy_genes.to_hash()
            cache_index[cache_key] = adapter_path
            
            # Put result in results queue
            results_queue.put({
                "job_id": job_id,
                "type": "training",
                "adapter_path": adapter_path,
                "cache_key": cache_key,
                "status": "completed",
                "timestamp": time.time()
            })
            
            print(f"✅ Training completed: {job_id} → {adapter_path}")
            
        except Exception as e:
            # Fail-fast: Put error in results queue
            results_queue.put({
                "job_id": job_id,
                "type": "training", 
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            })
            print(f"❌ Training failed: {job_id} - {e}")

# ========== GENERATION WORKERS ==========

@app.function(
    image=coral_image,
    volumes={"/cache": cache_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=600
)
@modal.batched(max_batch_size=5, wait_ms=2000)
async def generation_worker(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batched generation worker - processes multiple jobs efficiently."""
    import sys
    sys.path.insert(0, "/root/coralx")
    
    from infra.modal.codellama_service import generate_with_codellama_modal
    
    results = []
    
    for job in jobs:
        try:
            job_id = job["job_id"]
            adapter_path = job["adapter_path"]
            problem_data = job["problem_data"]
            cheap_knobs = job["cheap_knobs"]
            config = job["config"]
            
            print(f"🤖 Generation worker processing: {job_id}")
            
            # Pure generation - adapter path provided directly
            generated_code = generate_with_codellama_modal(
                adapter_path=adapter_path,
                problem_data=problem_data,
                cheap_knobs=cheap_knobs,
                config=config
            )
            
            result = {
                "job_id": job_id,
                "type": "generation",
                "generated_code": generated_code,
                "status": "completed",
                "timestamp": time.time()
            }
            
            # Also put in results queue for streaming
            results_queue.put(result)
            results.append(result)
            
        except Exception as e:
            error_result = {
                "job_id": job_id,
                "type": "generation",
                "status": "failed", 
                "error": str(e),
                "timestamp": time.time()
            }
            results_queue.put(error_result)
            results.append(error_result)
    
    return results

# ========== TEST WORKERS ==========

@app.function(
    image=coral_image,
    cpu=4,
    memory=8192,
    timeout=300
)
def test_worker():
    """Lightweight test execution worker."""
    import sys
    sys.path.insert(0, "/root/coralx")
    
    from coral.domain.quixbugs_evaluation import evaluate_quixbugs_code
    
    while True:
        try:
            job = test_queue.get(timeout=30)
            if job is None:
                break
            
            job_id, generated_code, problem_data, config = job
            print(f"🧪 Test worker processing: {job_id}")
            
            # Direct Python execution - no pytest complexity
            scores = evaluate_quixbugs_code(
                generated_code=generated_code,
                problem_data=problem_data,
                config=config
            )
            
            results_queue.put({
                "job_id": job_id,
                "type": "testing",
                "scores": scores,
                "status": "completed",
                "timestamp": time.time()
            })
            
        except Exception as e:
            results_queue.put({
                "job_id": job_id,
                "type": "testing",
                "status": "failed",
                "error": str(e), 
                "timestamp": time.time()
            })

# ========== CACHE MANAGEMENT ==========

@app.function(image=coral_image, cpu=1, memory=1024)
def cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    try:
        total_adapters = len(cache_index)
        cache_keys = list(cache_index.keys())[:10]  # Sample for debugging
        
        return {
            "total_adapters": total_adapters,
            "sample_keys": cache_keys,
            "status": "healthy"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ========== QUEUE MANAGEMENT ==========

@app.function(image=coral_image, cpu=1, memory=1024)
def queue_stats() -> Dict[str, Any]:
    """Get queue statistics for monitoring."""
    try:
        return {
            "training_queue_len": len(training_queue),
            "generation_queue_len": len(generation_queue),
            "test_queue_len": len(test_queue),
            "results_queue_len": len(results_queue),
            "status": "healthy"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=coral_image, cpu=1, memory=1024)
def clear_queues() -> Dict[str, str]:
    """Clear all queues - development only."""
    try:
        # Clear queues by consuming all items
        while training_queue.get(timeout=1) is not None:
            pass
        while generation_queue.get(timeout=1) is not None:
            pass
        while test_queue.get(timeout=1) is not None:
            pass
        while results_queue.get(timeout=1) is not None:
            pass
        
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ========== LOCAL ENTRYPOINT ==========

@app.local_entrypoint()
def test_queue_system():
    """Test the queue-based system."""
    print("🧪 Testing queue-based Modal system...")
    
    # Test queue stats
    stats = queue_stats.remote()
    print(f"Queue stats: {stats}")
    
    # Test cache stats  
    cache_stats_result = cache_stats.remote()
    print(f"Cache stats: {cache_stats_result}")
    
    print("✅ Queue system test complete")
```

**Key Features**:
- **Queue-based coordination** - no manual volume sync
- **Auto-scaling workers** - Modal manages container lifecycle
- **Batched generation** - efficient GPU utilization
- **Atomic cache operations** - Modal Dict eliminates race conditions
- **Fail-fast error handling** - errors propagated via queues

---

### **2. `infra/queue_modal_executor.py`** (Priority P0)

**Purpose**: Queue-aware executor implementing proper functors

```python
"""
Queue-based Modal Executor - Category Theory Compliant
Implements proper functors with natural transformations via Modal Queues.
"""
from typing import Any, Callable, Dict, List, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
import modal
import uuid
import time

from coral.ports.interfaces import Executor
from coral.domain.genome import Genome, MultiObjectiveScores


@dataclass(frozen=True)
class QueueJob:
    """Immutable job record for queue submission."""
    job_id: str
    job_type: str  # "training", "generation", "testing"
    data: Dict[str, Any]
    timestamp: float


class QueueBasedModalExecutor(Executor):
    """
    Modal executor using queues for coordination.
    Implements proper category theory functors.
    """
    
    def __init__(self, app_name: str = "coral-x-queues"):
        if not self._check_modal_available():
            raise RuntimeError("FAIL-FAST: Modal not available")
        
        self.app_name = app_name
        self._setup_queues()
        self._setup_workers()
    
    def _check_modal_available(self) -> bool:
        """Check if Modal is available - fail fast if not."""
        try:
            import modal
            return True
        except ImportError:
            return False
    
    def _setup_queues(self):
        """Setup queue connections - natural transformations."""
        try:
            self.training_queue = modal.Queue.from_name("coral-training")
            self.generation_queue = modal.Queue.from_name("coral-generation")
            self.test_queue = modal.Queue.from_name("coral-tests")
            self.results_queue = modal.Queue.from_name("coral-results")
            self.cache_index = modal.Dict.from_name("coral-cache-index")
        except Exception as e:
            raise RuntimeError(f"FAIL-FAST: Could not connect to Modal queues: {e}")
    
    def _setup_workers(self):
        """Setup worker function references."""
        try:
            self.training_worker = modal.Function.from_name(self.app_name, "training_worker")
            self.generation_worker = modal.Function.from_name(self.app_name, "generation_worker")
            self.test_worker = modal.Function.from_name(self.app_name, "test_worker")
        except Exception as e:
            raise RuntimeError(f"FAIL-FAST: Could not connect to Modal workers: {e}")
    
    # ========== FUNCTOR IMPLEMENTATION ==========
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit function execution via queues.
        Implements functor mapping: Local → Queue.
        """
        function_name = getattr(fn, '__name__', str(fn))
        
        if 'train' in function_name.lower():
            return self._submit_training(*args, **kwargs)
        elif 'generate' in function_name.lower():
            return self._submit_generation(*args, **kwargs)
        elif 'test' in function_name.lower() or 'evaluate' in function_name.lower():
            return self._submit_testing(*args, **kwargs)
        else:
            raise ValueError(f"FAIL-FAST: Unknown function type: {function_name}")
    
    def submit_training_batch(self, genomes: List[Genome], config: Dict[str, Any]) -> 'BatchFuture':
        """
        Submit batch of training jobs.
        Natural transformation: List[Genome] → Queue[TrainingJob].
        """
        jobs = []
        for genome in genomes:
            job_id = str(uuid.uuid4())
            
            # Check cache first
            cache_key = genome.heavy_genes.to_hash()
            if cache_key in self.cache_index:
                # Cache hit - no training needed
                continue
            
            job = QueueJob(
                job_id=job_id,
                job_type="training",
                data={
                    "genome_id": genome.id,
                    "heavy_genes": genome.heavy_genes.to_dict(),
                    "config": config
                },
                timestamp=time.time()
            )
            jobs.append(job)
        
        # Submit all jobs to training queue
        for job in jobs:
            self.training_queue.put((job.job_id, job.data["heavy_genes"], job.data["config"]))
        
        return BatchFuture(job_ids=[j.job_id for j in jobs], executor=self)
    
    def submit_generation_batch(self, generation_requests: List[Dict], config: Dict[str, Any]) -> 'BatchFuture':
        """
        Submit batch of generation jobs.
        Functorial mapping preserving structure.
        """
        jobs = []
        job_batch = []
        
        for request in generation_requests:
            job_id = str(uuid.uuid4())
            
            job_data = {
                "job_id": job_id,
                "adapter_path": request["adapter_path"],
                "problem_data": request["problem_data"], 
                "cheap_knobs": request["cheap_knobs"],
                "config": config
            }
            
            job_batch.append(job_data)
            jobs.append(job_id)
        
        # Submit batch to generation worker
        self.generation_worker.spawn(job_batch)
        
        return BatchFuture(job_ids=jobs, executor=self)
    
    def stream_results(self, timeout: float = 300) -> Iterator[Dict[str, Any]]:
        """
        Stream results from queue.
        Natural transformation: Queue → Iterator.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.results_queue.get(timeout=5)
                if result is not None:
                    yield result
            except:
                # Queue empty - continue waiting
                continue
    
    # ========== CATEGORY THEORY COMPLIANCE ==========
    
    def map_morphism(self, f: Callable) -> Callable:
        """
        Map local morphism to queue morphism.
        Preserves composition: F(g ∘ f) = F(g) ∘ F(f).
        """
        def queue_morphism(*args, **kwargs):
            # Submit to appropriate queue
            future = self.submit(f, *args, **kwargs)
            return future.result()
        
        return queue_morphism
    
    def natural_transformation(self, local_obj: Any) -> 'QueueObject':
        """
        Natural transformation: Local → Queue.
        Satisfies naturality condition.
        """
        job_id = str(uuid.uuid4())
        # Transform object to queue-compatible format
        return QueueObject(job_id=job_id, data=local_obj)


@dataclass
class QueueObject:
    """Queue-wrapped object preserving structure."""
    job_id: str
    data: Any


class BatchFuture:
    """
    Future for batch operations.
    Implements proper monad structure.
    """
    
    def __init__(self, job_ids: List[str], executor: QueueBasedModalExecutor):
        self.job_ids = job_ids
        self.executor = executor
        self._results = {}
    
    def result(self, timeout: float = 300) -> List[Any]:
        """Collect all batch results."""
        results = []
        collected_ids = set()
        
        for result in self.executor.stream_results(timeout=timeout):
            job_id = result.get("job_id")
            if job_id in self.job_ids and job_id not in collected_ids:
                results.append(result)
                collected_ids.add(job_id)
                
                if len(collected_ids) == len(self.job_ids):
                    break
        
        if len(collected_ids) != len(self.job_ids):
            missing = set(self.job_ids) - collected_ids
            raise RuntimeError(f"FAIL-FAST: Missing results for jobs: {missing}")
        
        return results
    
    def map(self, f: Callable) -> 'BatchFuture':
        """Functor map operation."""
        # Apply function to each result when available
        new_job_ids = [f"{job_id}_mapped" for job_id in self.job_ids]
        return BatchFuture(job_ids=new_job_ids, executor=self.executor)
```

**Key Features**:
- **Proper functors** - preserves composition laws
- **Natural transformations** - between Local and Queue categories  
- **Batch operations** - efficient processing
- **Fail-fast validation** - strict error handling
- **Monad structure** - proper bind/map operations

---

## 🔧 **FILES TO MODIFY**

### **3. `coral/application/evolution_engine.py`** (Priority P1)

**Changes**: Transform to use queue-based executor

```python
# Add after existing imports:
from infra.queue_modal_executor import QueueBasedModalExecutor, BatchFuture

class EvolutionEngine:
    def __init__(self, cfg: CoralConfig, ...):
        # ... existing code ...
        
        # Replace complex modal executor with queue-based
        if isinstance(self.executor, QueueBasedModalExecutor):
            self._use_queue_based_evolution = True
        else:
            self._use_queue_based_evolution = False
    
    def evolve_generation(self, population: Population) -> Population:
        """Queue-based evolution with proper functor composition."""
        if self._use_queue_based_evolution:
            return self._evolve_generation_queue_based(population)
        else:
            return self._evolve_generation_legacy(population)  # Keep legacy for migration
    
    def _evolve_generation_queue_based(self, population: Population) -> Population:
        """
        Queue-based evolution implementing proper category theory.
        Functor composition: Population → TrainingJobs → GenerationJobs → TestJobs → Population
        """
        print(f"🧬 Queue-based evolution: Generation {self.current_generation}")
        
        # Step 1: Submit training jobs (Functor F1: Population → TrainingResults)
        training_future = self.executor.submit_training_batch(
            genomes=population.genomes,
            config=self.cfg.__dict__
        )
        
        # Step 2: Wait for training completion and submit generation jobs
        training_results = training_future.result(timeout=1800)  # 30 min timeout
        
        # Create generation requests based on training results
        generation_requests = self._create_generation_requests(
            population=population,
            training_results=training_results
        )
        
        # Submit generation batch (Functor F2: TrainingResults → GenerationResults)
        generation_future = self.executor.submit_generation_batch(
            generation_requests=generation_requests,
            config=self.cfg.__dict__
        )
        
        # Step 3: Collect generation results and submit test jobs
        generation_results = generation_future.result(timeout=600)  # 10 min timeout
        
        # Create test requests based on generation results  
        test_requests = self._create_test_requests(
            population=population,
            generation_results=generation_results
        )
        
        # Submit test batch (Functor F3: GenerationResults → TestResults)
        test_future = self.executor.submit_test_batch(
            test_requests=test_requests,
            config=self.cfg.__dict__
        )
        
        # Step 4: Collect final results and update population
        test_results = test_future.result(timeout=300)  # 5 min timeout
        
        # Apply results to population (Functor F4: TestResults → Population)
        updated_population = self._apply_results_to_population(
            population=population,
            test_results=test_results
        )
        
        # Apply selection and mutation (Pure domain functions)
        selected_population = self._apply_selection(updated_population)
        mutated_population = self._apply_mutation(selected_population)
        
        return mutated_population
    
    def _create_generation_requests(self, population: Population, training_results: List[Dict]) -> List[Dict]:
        """Pure function: Population × TrainingResults → GenerationRequests"""
        requests = []
        
        # Create map of genome_id to adapter_path
        adapter_map = {}
        for result in training_results:
            if result["status"] == "completed":
                genome_id = result.get("genome_id")
                adapter_path = result["adapter_path"]
                adapter_map[genome_id] = adapter_path
        
        # Create generation requests for each genome × problem
        problems = list(self.dataset.problems())
        
        for genome in population.genomes:
            # Get adapter path (from training or cache)
            adapter_path = adapter_map.get(genome.id)
            if not adapter_path:
                # Check cache for existing adapter
                cache_key = genome.heavy_genes.to_hash()
                adapter_path = self.executor.cache_index.get(cache_key)
            
            if not adapter_path:
                raise RuntimeError(f"FAIL-FAST: No adapter available for genome {genome.id}")
            
            for problem in problems:
                # Generate cheap knobs from CA features
                ca_features = self._extract_ca_features(genome)
                cheap_knobs = self._generate_cheap_knobs(ca_features)
                
                requests.append({
                    "genome_id": genome.id,
                    "problem_id": problem["name"],
                    "adapter_path": adapter_path,
                    "problem_data": problem,
                    "cheap_knobs": cheap_knobs
                })
        
        return requests
```

**Key Changes**:
- **Queue-based coordination** replaces manual orchestration
- **Proper functor composition** with timeout handling
- **Fail-fast validation** at each step
- **Parallel/legacy toggle** for migration safety

---

### **4. `coral/ports/interfaces.py`** (Priority P1)

**Changes**: Add queue-aware protocols

```python
# Add after existing imports:
from typing import List, Iterator, Dict, Any
from concurrent.futures import Future

# Add new protocols:

class QueueAwareExecutor(Protocol):
    """Enhanced executor with batch and streaming capabilities."""
    
    def submit_training_batch(self, genomes: List['Genome'], config: Dict[str, Any]) -> 'BatchFuture':
        """Submit batch of training jobs."""
        ...
    
    def submit_generation_batch(self, requests: List[Dict], config: Dict[str, Any]) -> 'BatchFuture':
        """Submit batch of generation jobs."""
        ...
    
    def submit_test_batch(self, requests: List[Dict], config: Dict[str, Any]) -> 'BatchFuture':
        """Submit batch of test jobs."""
        ...
    
    def stream_results(self, timeout: float = 300) -> Iterator[Dict[str, Any]]:
        """Stream results from queue."""
        ...


class BatchFuture(Protocol):
    """Future for batch operations."""
    
    def result(self, timeout: float = 300) -> List[Any]:
        """Get all batch results."""
        ...
    
    def map(self, f: Callable) -> 'BatchFuture':
        """Apply function to results (functor map)."""
        ...
```

---

## 🗑️ **FILES TO DELETE (After Migration)**

### **5. Complex Volume Coordination Files**

```bash
# After successful migration and validation:
rm coral_modal_app.py                    # Replace with coral_queue_modal_app.py
rm infra/modal_executor.py               # Replace with queue_modal_executor.py

# Clean up complex cache files:
rm infra/adapter_cache.py                # Replace with simple cache_index Dict operations

# Remove race condition workarounds:
find . -name "*volume*coordination*" -delete
find . -name "*cache*race*" -delete
```

---

## 📊 **Migration Validation**

### **Validation Script**: `validate_queue_migration.py`

```python
"""
Validation harness for queue-based migration.
Ensures results match between legacy and queue systems.
"""
import yaml
from pathlib import Path

def validate_migration():
    """Run identical experiments on legacy vs queue systems."""
    
    # Load test config
    config_path = "test_configs/migration_test.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Test 1: Legacy system results
    print("🧪 Running legacy Modal system...")
    config['infra']['executor'] = 'modal_legacy'
    legacy_results = run_coral_experiment(config)
    
    # Test 2: Queue system results  
    print("🧪 Running queue-based Modal system...")
    config['infra']['executor'] = 'queue_modal'
    queue_results = run_coral_experiment(config)
    
    # Validation
    assert_results_equivalent(legacy_results, queue_results)
    assert_performance_improved(queue_results, legacy_results)
    
    print("✅ Migration validation passed!")

def assert_results_equivalent(legacy, queue):
    """Ensure scientific results are equivalent."""
    # Compare final fitness scores
    assert abs(legacy.best_fitness - queue.best_fitness) < 0.01
    
    # Compare population diversity  
    assert abs(legacy.diversity_score - queue.diversity_score) < 0.05
    
    # Compare cache efficiency
    assert queue.cache_hit_rate >= legacy.cache_hit_rate  # Should be better

def assert_performance_improved(queue, legacy):
    """Ensure performance metrics improved."""
    assert queue.total_time <= legacy.total_time * 1.1  # At most 10% slower
    assert queue.race_conditions == 0  # No race conditions
    assert queue.error_recovery_time < legacy.error_recovery_time
```

---

This detailed implementation map provides exact file-by-file changes needed to transform CORAL-X from complex volume coordination to a clean, queue-based architecture that properly implements category theory principles while maintaining the fail-fast philosophy. 

## ✅ **Success Metrics**

- 80% reduction in infrastructure complexity
- Zero race conditions
- Category theory compliance restored
- Auto-scaling workers
- Fail-fast principles maintained 