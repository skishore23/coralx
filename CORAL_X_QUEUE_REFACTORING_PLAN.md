# CORAL-X Queue-Based Refactoring Plan

**Transform complex Modal volume coordination → Simple queue-based functorial architecture**

---

## 🎯 **Executive Summary**

**Current State**: Complex 794-line Modal infrastructure with manual volume coordination, race conditions, and violated category theory principles.

**Target State**: Clean queue-based architecture where Modal Queues handle coordination, restoring proper functors and eliminating 80% of infrastructure complexity.

**Key Principle**: **Fail-Fast + Category Theory + Onion Architecture**

---

## 📋 **Current Architecture Analysis**

### **❌ Problems to Solve**

| **Component** | **Current Issue** | **Impact** |
|---------------|------------------|------------|
| **Volume Coordination** | Manual `volume.commit()`/`reload()` | Race conditions, complexity |
| **Container Orchestration** | Manual function routing via string names | Broken functor composition |
| **Cache Management** | File-based with hash inconsistencies | Cache misses, wasted training |
| **Error Recovery** | Complex retry logic across containers | Unpredictable failures |
| **Monitoring** | Custom logging scattered across functions | Poor observability |
| **Scaling** | Manual resource management | Inefficient GPU usage |

### **✅ What's Working (Keep)**

| **Component** | **Why Keep** | **Integration Plan** |
|---------------|--------------|-------------------|
| **Domain Layer** | Pure functions, proper functors | Wrap in queue workers |
| **Plugin System** | Clean interfaces (`DatasetProvider`, `ModelRunner`, `FitnessFn`) | Queue-compatible adapters |
| **Configuration System** | Type-safe YAML → dataclass pipeline | Add queue configurations |
| **CLI Interface** | Clean Typer-based commands | Add queue management commands |

---

## 🚀 **Target Queue-Based Architecture**

### **Core Transformation**

```mermaid
graph TB
    subgraph "NEW: Queue-Based Functorial Architecture"
        LocalEngine[Local Evolution Engine<br/>Category Theory Functors]
        
        subgraph "Modal Queue Layer (Natural Transformations)"
            TrainingQueue[Training Jobs<br/>modal.Queue]
            GenerationQueue[Generation Jobs<br/>modal.Queue]
            TestQueue[Test Jobs<br/>modal.Queue]
            ResultsQueue[Results<br/>modal.Queue]
            CacheDict[Cache Index<br/>modal.Dict]
        end
        
        subgraph "Auto-scaling Workers (Kleisli Arrows)"
            TrainingWorkers[Training Workers<br/>@app.function + queue.get()]
            GenerationWorkers[Generation Workers<br/>@modal.batched + queue]
            TestWorkers[Test Workers<br/>@app.function + queue.get()]
        end
        
        LocalEngine -->|Submit Jobs| TrainingQueue
        LocalEngine -->|Submit Jobs| GenerationQueue
        LocalEngine -->|Submit Jobs| TestQueue
        
        TrainingQueue -->|Auto-scale| TrainingWorkers
        GenerationQueue -->|Auto-scale| GenerationWorkers
        TestQueue -->|Auto-scale| TestWorkers
        
        TrainingWorkers -->|Results| ResultsQueue
        GenerationWorkers -->|Results| ResultsQueue
        TestWorkers -->|Results| ResultsQueue
        
        ResultsQueue -->|Stream Results| LocalEngine
    end
    
    style TrainingQueue fill:#99ff99
    style GenerationQueue fill:#99ccff
    style ResultsQueue fill:#ffcc99
    style CacheDict fill:#ff9999
```

---

## 📁 **File-by-File Refactoring Plan**

### **Phase 1: Core Infrastructure (Week 1)**

#### **🔄 NEW: `coral_queue_modal_app.py`**
```python
# Replace coral_modal_app.py with queue-based architecture
import modal

app = modal.App("coral-x-queues")

# Queue infrastructure
training_queue = modal.Queue.from_name("coral-training", create_if_missing=True)
generation_queue = modal.Queue.from_name("coral-generation", create_if_missing=True)
test_queue = modal.Queue.from_name("coral-tests", create_if_missing=True)
results_queue = modal.Queue.from_name("coral-results", create_if_missing=True)
cache_index = modal.Dict.from_name("coral-cache", create_if_missing=True)

# Auto-scaling workers replace manual function routing
@app.function(gpu="A100-40GB")
def training_worker():
    while True:
        job = training_queue.get(timeout=30)
        if job is None: break
        process_training_job(job)

@app.function(gpu="A100-40GB")
@modal.batched(max_batch_size=5, wait_ms=2000)
async def generation_worker(jobs: list):
    return [await process_generation_job(job) for job in jobs]
```

#### **🔧 MODIFY: `infra/modal_executor.py`**
```python
# Replace complex string-based routing with queue submission
class QueueBasedModalExecutor(Executor):
    def __init__(self, config: Dict[str, Any]):
        self.training_queue = modal.Queue.from_name("coral-training")
        self.generation_queue = modal.Queue.from_name("coral-generation") 
        self.results_queue = modal.Queue.from_name("coral-results")
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit to appropriate queue based on function type."""
        if 'train' in fn.__name__:
            return self._submit_training(*args, **kwargs)
        elif 'generate' in fn.__name__:
            return self._submit_generation(*args, **kwargs)
        # ... queue-based routing
```

#### **🗑️ DELETE: Complex Volume Logic**
- Remove `volume.commit()` / `volume.reload()` workarounds
- Remove `_volume_coordination.py` if exists
- Remove manual cache synchronization code

### **Phase 2: Domain Layer Integration (Week 2)**

#### **🔧 MODIFY: `coral/application/evolution_engine.py`**
```python
# Transform to queue-based evolution
class QueueBasedEvolutionEngine:
    def evolve_generation(self, population: Population) -> Population:
        # Functor composition via queues
        jobs = self._create_training_jobs(population)
        self.executor.submit_training_batch(jobs)
        
        # Collect results via queue streaming
        results = self._collect_results_stream()
        
        return self._apply_selection(results)
    
    def _create_training_jobs(self, population: Population) -> List[TrainingJob]:
        """Pure function: Population → List[TrainingJob]"""
        return [TrainingJob(g.id, g.heavy_genes) for g in population.genomes]
```

#### **🔧 MODIFY: `coral/domain/mapping.py`**
```python
# Enhance for queue compatibility
def serialize_for_queue(obj: Any) -> QueueSerializable:
    """Natural transformation: Local → Queue"""
    # Category theory compliant serialization
    
def deserialize_from_queue(data: QueueSerializable) -> Any:
    """Natural transformation: Queue → Local"""
    # Inverse transformation preserving structure
```

### **Phase 3: Plugin System Enhancement (Week 3)**

#### **🔧 MODIFY: `plugins/quixbugs_codellama/plugin.py`**
```python
# Make plugins queue-aware
class QuixBugsCodeLlamaQueuePlugin:
    def __init__(self, config: Dict[str, Any]):
        # Queue-compatible initialization
        self.generation_queue = modal.Queue.from_name("coral-generation")
        self.test_queue = modal.Queue.from_name("coral-tests")
    
    def fitness_fn(self) -> QueueAwareFitnessFn:
        """Return queue-compatible fitness function."""
        return QueueAwareFitnessFn(
            queue_submit=self._submit_evaluation_jobs,
            queue_collect=self._collect_evaluation_results
        )
```

#### **🔧 MODIFY: `coral/ports/interfaces.py`**
```python
# Add queue-aware protocols
class QueueAwareExecutor(Protocol):
    def submit_batch(self, jobs: List[Job]) -> BatchFuture: ...
    def stream_results(self) -> Iterator[Result]: ...

class QueueAwareFitnessFn(Protocol):
    def evaluate_batch(self, genomes: List[Genome]) -> List[Score]: ...
```

### **Phase 4: Cache Management Simplification (Week 4)**

#### **🔄 NEW: `infra/queue_cache.py`**
```python
# Replace complex volume cache with Modal Dict
class QueueBasedCache:
    def __init__(self):
        self.cache_index = modal.Dict.from_name("coral-cache")
        self.cache_queue = modal.Queue.from_name("coral-cache-ops")
    
    def get_or_train(self, heavy_genes: HeavyGenes) -> AdapterPath:
        """Simplified cache coordination via queues."""
        cache_key = heavy_genes.to_hash()
        
        if cache_key in self.cache_index:
            return self.cache_index[cache_key]  # Cache hit
        
        # Submit training job to queue
        self.cache_queue.put(("train", cache_key, heavy_genes))
        return self._wait_for_training_result(cache_key)
```

#### **🗑️ DELETE: Complex Cache Files**
- Remove `infra/adapter_cache.py` complex volume logic
- Remove manual hash coordination
- Remove cache consistency checks

---

## 🎭 **Category Theory Compliance Restoration**

### **Functor Laws Verification**

#### **Before (Broken)**
```python
# ❌ Violates functor laws due to side effects
def broken_modal_functor(f: LocalMorphism) -> ModalMorphism:
    # Manual serialization breaks F(id) = id
    # Volume coordination breaks F(g ∘ f) = F(g) ∘ F(f)
    pass
```

#### **After (Fixed)**
```python
# ✅ Proper functor preserving composition
class QueueFunctor:
    def map_object(self, obj: A) -> modal.Queue[A]:
        """Natural transformation preserving structure."""
        return self.queue.put(obj)
    
    def map_morphism(self, f: A → B) -> modal.Queue[A] → modal.Queue[B]:
        """Morphism mapping preserving composition."""
        @app.function()
        def queue_f(queue_a):
            return f(queue_a.get())
        return queue_f
```

### **Monad Structure**
```python
# Queue operations form proper monad
class QueueMonad:
    def unit(self, x: A) -> modal.Queue[A]:
        """η: A → Queue[A]"""
        return modal.Queue.ephemeral().put(x)
    
    def join(self, qq: modal.Queue[modal.Queue[A]]) -> modal.Queue[A]:
        """μ: Queue[Queue[A]] → Queue[A]"""
        # Flatten nested queues naturally
    
    def bind(self, qa: modal.Queue[A], f: A → modal.Queue[B]) -> modal.Queue[B]:
        """Kleisli composition via queue chaining"""
        return self.join(qa.map(f))
```

---

## 🔧 **CLI and Configuration Changes**

### **🔧 MODIFY: `cli/coral.py`**
```python
# Add queue management commands
@app.command()
def queue_status():
    """Show queue status and worker health."""
    
@app.command() 
def deploy_queues():
    """Deploy queue-based Modal app."""
    
@app.command()
def clear_queues():
    """Clear all queues (development only)."""
```

### **🔧 MODIFY: Configuration Files**
```yaml
# Add queue configurations to existing configs
infra:
  executor: "queue_modal"  # New executor type
  modal:
    app_name: "coral-x-queues"
    queues:
      training:
        timeout: 300
        max_batch_size: 10
      generation:
        timeout: 60
        batch_size: 5
        wait_ms: 2000
      results:
        ttl: 3600  # 1 hour TTL
```

---

## 📊 **Migration Strategy**

### **Parallel Migration Approach**

1. **Week 1**: Create queue-based app alongside existing app
2. **Week 2**: Add queue toggle in config (`executor: modal_legacy` vs `executor: queue_modal`)
3. **Week 3**: Test queue system with subset of workloads
4. **Week 4**: Full migration + remove legacy code

### **Validation Strategy**

```python
# Validation harness for migration
def validate_queue_migration():
    """Ensure queue-based results match legacy results."""
    legacy_results = run_with_legacy_modal()
    queue_results = run_with_queue_modal()
    
    assert_results_equivalent(legacy_results, queue_results)
    assert_performance_improved(queue_results, legacy_results)
```

---

## 🎯 **Expected Benefits**

| **Metric** | **Before** | **After** | **Improvement** |
|------------|------------|-----------|-----------------|
| **Infrastructure Code** | 794 lines | ~200 lines | 75% reduction |
| **Race Conditions** | Frequent | Eliminated | 100% elimination |
| **Cache Hit Rate** | 30-40% | 60-80% | 2x improvement |
| **Error Recovery** | Manual | Automatic | Built-in resilience |
| **Scaling** | Manual | Auto-scaling | Modal handles |
| **Monitoring** | Custom | Native Modal | Better observability |
| **Category Theory** | Broken | Compliant | Restored functors |

---

## 🚨 **Risk Mitigation**

### **Fail-Fast Checkpoints**

1. **Queue Connectivity**: Verify Modal queues accessible before migration
2. **Serialization**: Ensure all data types queue-compatible 
3. **Worker Health**: Monitor auto-scaling worker startup times
4. **Result Consistency**: Validate queue results match legacy results

### **Rollback Plan**

- Keep legacy Modal app deployed during migration
- Config toggle for instant rollback
- Automated tests comparing legacy vs queue results

---

## 🎉 **Success Criteria**

1. **✅ 80% reduction** in infrastructure complexity
2. **✅ Zero race conditions** in cache coordination
3. **✅ Category theory compliance** restored
4. **✅ Auto-scaling** workers handling load
5. **✅ Fail-fast** principles maintained
6. **✅ Plugin compatibility** preserved
7. **✅ Performance improvement** over legacy system

---

This plan transforms CORAL-X from a complex, manually-coordinated system into a clean, queue-based architecture that properly implements category theory principles while dramatically simplifying the infrastructure. 