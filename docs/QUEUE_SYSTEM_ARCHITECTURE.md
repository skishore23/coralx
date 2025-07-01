# CORAL-X Queue-Based Architecture Documentation

## Overview

The CORAL-X queue-based system replaces direct Modal function calls with a distributed queue architecture for better scalability, cost efficiency, and fault tolerance.

## Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local Client  │    │  Modal Queues   │    │ Modal Workers   │
│                 │    │                 │    │                 │
│ Evolution       │───▶│ Training Queue  │───▶│ Training        │
│ Engine          │    │ Test Queue      │    │ Workers (2x)    │
│                 │    │ Results Queue   │    │                 │
│                 │◀───│                 │◀───│ Test Workers    │
│                 │    │                 │    │ (3x)            │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Queue Types

### 1. Training Queue
- **Purpose**: DoRA/LoRA adapter training jobs
- **Workers**: 2x `training_worker_with_queues` (A100-40GB)
- **Job Type**: `training`
- **Input**: Heavy genes, base model, save path, config
- **Output**: Trained adapter path

### 2. Test Queue  
- **Purpose**: Genome evaluation jobs
- **Workers**: 3x `test_worker_with_queues` (A10G)
- **Job Type**: `evaluation` or `test`
- **Input**: Genome data, adapter path, config
- **Output**: Multi-objective scores

### 3. Results Queue
- **Purpose**: Collecting completed job results
- **Consumer**: Queue executor result collector thread
- **Job Types**: All result types (training, evaluation, test)

## Job Flow

### Training Flow
```
1. Evolution Engine → Queue Executor
2. Queue Executor → HeavyGenes → Training Queue
3. Training Worker → Gets job → Calls train_codellama_lora()
4. Training Worker → Puts result → Results Queue
5. Result Collector → Gets result → Returns to Evolution Engine
```

### Evaluation Flow  
```
1. Evolution Engine → Queue Executor  
2. Queue Executor → Genome data → Test Queue
3. Test Worker → Gets job → Routes to process_evaluation_job()
4. process_evaluation_job() → Calls evaluate_genome_modal()
5. Test Worker → Puts result → Results Queue
6. Result Collector → Gets result → Returns to Evolution Engine
```

## Worker Functions

### training_worker_with_queues
- **Resources**: A100-40GB, 16GB RAM, 30min timeout
- **Function**: Processes training jobs from training_queue
- **Calls**: `process_training_job()` → `train_codellama_lora()`
- **Output**: Adapter paths to results_queue

### test_worker_with_queues  
- **Resources**: A10G, 4GB RAM, 5min timeout
- **Function**: Processes evaluation jobs from test_queue
- **Routes**: 
  - `job_type == 'evaluation'` → `process_evaluation_job()`
  - `job_type == 'test'` → `process_test_job()`
- **Output**: Evaluation results to results_queue

## Job Processing Functions

### process_training_job()
- Updates progress tracking during training
- Calls `train_codellama_lora()` from domain layer
- Handles idempotency caching
- Includes cost circuit-breaker checks

### process_evaluation_job()  
- **NEW**: Added to handle genome evaluation
- Updates progress tracking during evaluation
- Calls `evaluate_genome_modal()` from experiment service
- Updates best scores in progress file

### process_test_job()
- **LEGACY**: Handles old-style test jobs
- For backward compatibility
- Uses QuixBugs adapter for code testing

## Queue Executor Architecture

### QueueBasedModalExecutor
```python
class QueueBasedModalExecutor:
    - training_queue: Modal.Queue  
    - test_queue: Modal.Queue
    - results_queue: Modal.Queue
    - _pending_jobs: Dict[str, Future]  # Track submitted jobs
    - _result_collector_running: bool   # Background thread status
```

### Key Methods
- `submit_training()`: Submit training job to training_queue
- `submit_evaluation()`: Submit evaluation job to test_queue  
- `_collect_results_continuously()`: Background thread collecting from results_queue
- `_process_queue_result()`: Convert queue results back to local format

## Configuration

### Queue-Based Config
```yaml
infra:
  executor: queue_modal           # Use queue-based system
  modal:
    app_name: coral-x-queues     # Queue-based Modal app
```

### Standard Config  
```yaml
infra:
  executor: modal                 # Direct function calls
  modal:
    app_name: coral-x-production # Standard Modal app
```

## Current Known Issues

### 1. Job Routing Problem (FIXED)
- **Issue**: Evaluation jobs routed to `process_test_job` instead of `process_evaluation_job`
- **Fix**: Added job_type routing in `test_worker_with_queues`

### 2. Result Collection Timeout 
- **Issue**: Results queue timeout causing "Result collection error"
- **Symptoms**: Jobs submitted but results never received
- **Location**: `_collect_results_continuously()` in queue executor

### 3. Progress Tracking Gaps
- **Issue**: Progress file not updated during queue execution
- **Fix**: Added progress updates in `process_training_job()` and `process_evaluation_job()`

## Debugging Queue Issues

### Check Queue Status
```bash
modal run coral_queue_modal_app.py::queue_status
```

### Check Worker Status
```bash
modal app logs coral-x-queues --tail 50
```

### Check Volume Contents
```bash  
modal volume ls coral-x-clean-cache adapters/
modal volume get coral-x-clean-cache evolution_progress.json /tmp/progress.json
```

### Manual Queue Testing
```python
# Test job submission
from infra.queue_modal_executor import QueueBasedModalExecutor
executor = QueueBasedModalExecutor(config)
future = executor.submit_evaluation(genome, adapter_path, config)
result = future.result(timeout=300)  # 5 min timeout
```

## Performance Characteristics

### Advantages
- **Auto-scaling**: Workers scale up/down based on queue depth
- **Fault tolerance**: Jobs survive worker crashes via queue persistence  
- **Cost efficiency**: Pay only for active workers, not idle functions
- **Resource isolation**: Training (A100) vs evaluation (A10G) workers

### Disadvantages  
- **Complexity**: More moving parts than direct function calls
- **Latency**: Queue overhead vs direct function invocation
- **Debugging**: Harder to trace job flow through multiple components

## Migration Path

### From Direct → Queue
1. Change `executor: modal` → `executor: queue_modal`
2. Change `app_name: coral-x-production` → `app_name: coral-x-queues`  
3. Deploy queue-based app: `modal deploy coral_queue_modal_app.py`
4. No code changes needed - same Evolution Engine interface

### From Queue → Direct
1. Change `executor: queue_modal` → `executor: modal`
2. Change `app_name: coral-x-queues` → `app_name: coral-x-production`
3. Deploy standard app: `modal deploy coral_modal_app.py`

## Monitoring & Observability

### Key Metrics
- Queue depths (training_queue, test_queue, results_queue)
- Job completion rates 
- Worker utilization
- Result collection errors
- Average job latency

### Health Checks
- Queue connectivity
- Worker responsiveness  
- Result collection thread status
- Progress file updates 