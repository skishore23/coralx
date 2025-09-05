#!/usr/bin/env python3
"""
CORAL-X Queue-Based Modal App
Clean architecture using Modal Queues for coordination
Based on CORAL_X_QUEUE_REFACTORING_PLAN.md
"""

import modal
import time
import os
from pathlib import Path

# ========================================
# Modal app and infrastructure
# ========================================

app = modal.App("coral-x-queues")

# Build images
coral_gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "safetensors>=0.3.1"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(".", remote_path="/root/coralx")
)

coral_cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "requests>=2.28.0"
    ])
    .add_local_dir(".", remote_path="/root/coralx")
)

# Create volume
coral_volume = modal.Volume.from_name("coral-x-clean-cache", create_if_missing=True)

# ========================================
# 🚀 QUEUE INFRASTRUCTURE - GLOBAL CATEGORY OBJECTS
# ========================================

# 🧮 CATEGORY THEORY: Global queue category - persistent across function calls
# These MUST be defined BEFORE the worker functions that reference them
training_queue = modal.Queue.from_name("coral-training", create_if_missing=True)
test_queue = modal.Queue.from_name("coral-test", create_if_missing=True)
generation_queue = modal.Queue.from_name("coral-generation", create_if_missing=True)
results_queue = modal.Queue.from_name("coral-results", create_if_missing=True)
cache_index = modal.Dict.from_name("coral-cache-index", create_if_missing=True)

print("🧮 Global queue category objects initialized")
print("   Queues: coral-training, coral-test, coral-generation, coral-results")
print("   Cache: coral-cache-index")

# ========================================
# 🏗️ AUTO-SCALING WORKERS
# ========================================

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")]
)
def training_worker():
    """
    Category theory compliant training worker.
    Natural transformation: η: Local → Queue via global queue reference.
    """
    """Natural transformation: Local → Queue → Result via global queue category."""
    import sys
    import time

    # 🧮 CATEGORY THEORY: Reference global queue objects (natural transformation)
    global training_queue, results_queue, cache_index

    # 🔧 Environment setup
    print("🏗️ Training worker started (category theory compliant)")

    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        print(f"❌ CoralX codebase not found: {coralx_path}")
        return

    sys.path.insert(0, str(coralx_path))
    print(f"🐍 Added to Python path: {coralx_path}")
    print("🏗️ Training worker ready - processing from global training queue...")

    # 🧮 CATEGORY THEORY: Continuous natural transformation η: Queue[Job] → Queue[Result]
    while True:
        job = None
        try:
            # η: Get job from global queue (natural transformation)
            job = training_queue.get(timeout=60)  # Longer timeout for stability
            if job is None:
                continue  # Keep worker alive for auto-scaling

            print(f"🚀 Processing training job: {job['job_id']}")

            # F(process_training_job): Pure functorial mapping
            result = process_training_job(job, idempotency_cache=cache_index)

            # μ: Put result in global results queue (natural transformation)
            result_data = {
                'job_id': job['job_id'],
                'job_type': 'training',
                'result': result,
                'timestamp': time.time()
            }
            results_queue.put(result_data)

            print(f"✅ Training completed: {job['job_id']}")

        except Exception as e:
            # Handle errors while preserving categorical structure
            error_str = str(e).lower()
            exception_type = type(e).__name__.lower()

            # Check for cancellation/shutdown conditions (  Exit gracefully on shutdown)
            is_cancellation = (
                exception_type in ['clientclosed', 'cancelled', 'asynciocancellederror'] or
                'client' in error_str and 'closed' in error_str or
                'cancell' in error_str or
                'shutdown' in error_str
            )

            if is_cancellation:
                print(f"🛑 Training worker graceful shutdown: {type(e).__name__}")
                break  # Exit worker loop gracefully

            # Check for normal timeout conditions (queue empty, timeouts, etc.)
            is_timeout = (
                any(timeout_word in error_str for timeout_word in ['timeout', 'empty', 'no message']) or
                exception_type in ['empty', 'queueempty'] or
                'queue.empty' in str(e).lower()
            )

            if is_timeout:
                # Normal timeout - worker stays alive for auto-scaling
                continue
            else:
                # ENHANCED ERROR REPORTING: Show detailed error information
                import traceback
                error_details = f"{type(e).__name__}: {str(e)}"
                stack_trace = traceback.format_exc()
                print(f"❌ Training worker error: {error_details}")
                print(f"📋 Stack trace: {stack_trace}")
                if job:
                    print(f"🔍 Failed job details: {job.get('job_id', 'unknown')} - {job.get('job_type', 'training')}")
                    # Send error result through proper channel (with cancellation check)
                    try:
                        error_result = {
                            'job_id': job.get('job_id', 'unknown'),
                            'job_type': 'training',
                            'error': error_details,
                            'stack_trace': stack_trace,
                            'timestamp': time.time()
                        }
                        results_queue.put(error_result)
                    except Exception as put_error:
                        # If we can't put error result (e.g., client shutdown), just log and exit
                        print(f"⚠️  Unable to report error result (likely shutdown): {put_error}")
                        break

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A10G",
    memory=8192,
    timeout=900
)
@modal.batched(max_batch_size=5, wait_ms=2000)
def generation_worker(jobs: list):
    """Auto-scaling generation worker - batched for efficiency."""
    import sys

    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        return [{"error": "CoralX codebase not found"} for _ in jobs]
    sys.path.insert(0, str(coralx_path))

    print(f"🧪 Processing {len(jobs)} generation jobs in batch")

    results = []
    for job in jobs:
        try:
            result = process_generation_job(job)
            results.append({
                'job_id': job['job_id'],
                'result': result,
                'timestamp': time.time()
            })
        except Exception as e:
            results.append({
                'job_id': job['job_id'],
                'error': str(e),
                'timestamp': time.time()
            })

    return results

@app.function(
    image=coral_gpu_image,  # Use GPU image (has scipy) instead of CPU image
    volumes={"/cache": coral_volume},
    gpu="A10G",       # 🔥 FIX: Add GPU for fast inference
    cpu=4,
    memory=8192,      # 🔥 FIX: Increased memory for GPU operations
    timeout=900,      # 🔥 FIX: Longer timeout for GPU inference
    secrets=[modal.Secret.from_name("huggingface")]
)
def test_worker():
    """
    Category theory compliant test/evaluation worker.
    Natural transformation: η: Local → Queue → Result via global queue category.
    """
    import sys
    import time

    # 🧮 CATEGORY THEORY: Reference global queue objects (natural transformation)
    global test_queue, results_queue, cache_index

    # 🔧 Environment setup
    print("🧪 Test worker started (category theory compliant)")

    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        print(f"❌ CoralX codebase not found: {coralx_path}")
        return

    sys.path.insert(0, str(coralx_path))
    print(f"🐍 Added to Python path: {coralx_path}")
    print("🧪 Test worker ready - processing from global test queue...")

    # 🧮 CATEGORY THEORY: Continuous natural transformation η: Queue[Job] → Queue[Result]
    while True:
        job = None
        try:
            # η: Get job from global queue (natural transformation)
            job = test_queue.get(timeout=60)  # Longer timeout for stability
            if job is None:
                continue  # Keep worker alive for auto-scaling

            print(f"🧪 Processing job: {job['job_id']} (type: {job.get('job_type', 'unknown')})")

            # F(process_job): Pure functorial mapping based on job type
            job_type = job.get('job_type', 'test')
            if job_type == 'evaluation':
                result = process_evaluation_job(job)
            else:
                result = process_test_job(job)

            # μ: Put result in global results queue (natural transformation)
            result_data = {
                'job_id': job['job_id'],
                'job_type': job.get('job_type', 'test'),
                'result': result,
                'timestamp': time.time()
            }
            results_queue.put(result_data)

            print(f"✅ Test completed: {job['job_id']}")

        except Exception as e:
            # Handle errors while preserving categorical structure
            error_str = str(e).lower()
            exception_type = type(e).__name__.lower()

            # Check for cancellation/shutdown conditions (  Exit gracefully on shutdown)
            is_cancellation = (
                exception_type in ['clientclosed', 'cancelled', 'asynciocancellederror'] or
                'client' in error_str and 'closed' in error_str or
                'cancell' in error_str or
                'shutdown' in error_str
            )

            if is_cancellation:
                print(f"🛑 Test worker graceful shutdown: {type(e).__name__}")
                break  # Exit worker loop gracefully

            # Check for normal timeout conditions (queue empty, timeouts, etc.)
            is_timeout = (
                any(timeout_word in error_str for timeout_word in ['timeout', 'empty', 'no message']) or
                exception_type in ['empty', 'queueempty'] or
                'queue.empty' in str(e).lower()
            )

            if is_timeout:
                # Normal timeout - worker stays alive for auto-scaling
                continue
            else:
                # ENHANCED ERROR REPORTING: Show detailed error information
                import traceback
                error_details = f"{type(e).__name__}: {str(e)}"
                stack_trace = traceback.format_exc()
                print(f"❌ Test worker error: {error_details}")
                print(f"📋 Stack trace: {stack_trace}")
                if job:
                    print(f"🔍 Failed job details: {job.get('job_id', 'unknown')} - {job.get('job_type', 'test')}")
                    # Send error result through proper channel (with cancellation check)
                    try:
                        error_result = {
                            'job_id': job.get('job_id', 'unknown'),
                            'job_type': job.get('job_type', 'test'),
                            'error': error_details,
                            'stack_trace': stack_trace,
                            'timestamp': time.time()
                        }
                        results_queue.put(error_result)
                    except Exception as put_error:
                        # If we can't put error result (e.g., client shutdown), just log and exit
                        print(f"⚠️  Unable to report error result (likely shutdown): {put_error}")
                        break

# ========================================
# 🔧 JOB PROCESSING FUNCTIONS
# ========================================

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,
    timeout=60
)
def check_cost_budget():
    """Check if we're within cost budget for the day."""
    from datetime import datetime

    # Simple budget check - can be made more sophisticated
    budget_limit_gpu_hours = int(os.environ.get('CORAL_DAILY_GPU_BUDGET', '10'))  # Default 10 GPU hours/day

    print(f"💰 Checking cost budget (limit: {budget_limit_gpu_hours} GPU hours/day)")

    # For now, return a simple status - can be enhanced with actual Modal billing API
    budget_status = {
        'within_budget': True,  # TODO: Implement actual budget checking
        'gpu_hours_used_today': 0,  # TODO: Calculate from Modal usage
        'gpu_hours_limit': budget_limit_gpu_hours,
        'timestamp': datetime.now().isoformat()
    }

    print(f"✅ Budget check: {budget_status}")
    return budget_status

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    cpu=2,
    memory=4096,
    timeout=300
)
@modal.fastapi_endpoint(method="GET")
def check_cache_volume():
    """Check what's in the cache volume via HTTP endpoint."""

    cache_path = Path("/cache")
    print(f"📁 Cache volume contents at: {cache_path}")

    if not cache_path.exists():
        return {"error": "Cache path doesn't exist"}

    def scan_directory(path, max_depth=3, current_depth=0):
        items = []
        if current_depth >= max_depth:
            return items

        try:
            for item in sorted(path.iterdir()):
                if item.is_file():
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
                elif item.is_dir():
                    items.append(f"📁 {item.name}/")
                    if current_depth < max_depth - 1:
                        sub_items = scan_directory(item, max_depth, current_depth + 1)
                        for sub_item in sub_items[:10]:  # Limit sub-items
                            items.append(f"    {sub_item}")
                        if len(sub_items) > 10:
                            items.append(f"    ... and {len(sub_items) - 10} more items")
        except PermissionError:
            items.append("❌ Permission denied")
        except Exception as e:
            items.append(f"❌ Error: {e}")

        return items

    contents = scan_directory(cache_path)
    return {
        "path": str(cache_path),
        "contents": contents[:50],  # Limit output
        "total_items": len(contents)
    }

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")]
)
def setup_model_cache(model_name: str = "codellama/CodeLlama-7b-Python-hf"):
    """Pre-download and cache the model in the volume."""
    import os

    print(f"📥 Setting up model cache for: {model_name}")

    model_cache_dir = "/cache/models"
    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
            local_files_only=False  # Allow download
        )

        print("📥 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
            local_files_only=False,  # Allow download
            torch_dtype="auto",
            device_map="cpu"  # Keep on CPU for caching
        )

        print(f"✅ Model cached successfully at: {model_cache_dir}")

        # Check what was cached
        cached_files = []
        for root, dirs, files in os.walk(model_cache_dir):
            for file in files[:10]:  # Limit output
                file_path = Path(root) / file
                size = file_path.stat().st_size
                cached_files.append(f"{file} ({size} bytes)")

        return {
            "success": True,
            "model_name": model_name,
            "cache_dir": model_cache_dir,
            "cached_files": cached_files
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name
        }

def process_training_job(job, idempotency_cache=None):
    """Process a training job with cost budget check and idempotency."""
    from core.domain.lora_training import train_codellama_lora
    import json
    import time

    job_id = job['job_id']
    heavy_genes = job['heavy_genes']

    # 🔄 IDEMPOTENCY CHECK: Avoid duplicate processing
    if idempotency_cache and job_id in idempotency_cache:
        cached_result = idempotency_cache[job_id]
        print(f"🔄 Idempotency hit: {job_id} → {cached_result}")
        return cached_result

    # 💰 COST CIRCUIT-BREAKER: Check budget before expensive training
    try:
        budget_status = check_cost_budget.remote()
        if not budget_status['within_budget']:
            raise RuntimeError(f"COST-CIRCUIT-BREAKER: Daily GPU budget exceeded ({budget_status['gpu_hours_used_today']}/{budget_status['gpu_hours_limit']} hours)")
    except Exception as budget_error:
        print(f"⚠️  Budget check failed, proceeding cautiously: {budget_error}")

    # Extract job parameters
    base_model = job['base_model']
    save_path = job['save_path']
    config = job['config']

    print(f"🏗️ Training: {heavy_genes.to_hash()[:8]} → {save_path}")

    # 📊 UPDATE PROGRESS: Training started
    try:
        progress_file = Path("/cache/evolution_progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            # Update training stats
            progress['status'] = 'training'
            progress['message'] = f'Training adapter {heavy_genes.to_hash()[:8]}...'
            progress['training_stats']['current_adapter'] = heavy_genes.to_hash()[:8]
            progress['training_stats']['adapters_trained'] += 1
            progress['last_update'] = time.time()

            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️  Progress update failed: {e}")

    # Call training function directly
    result_path = train_codellama_lora(
        base_ckpt=base_model,
        heavy_genes=heavy_genes,
        save_to=save_path,
        config=config
    )

    # 📊 UPDATE PROGRESS: Training completed
    try:
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            # Update training completion stats
            progress['training_stats']['training_rate'] = progress['training_stats']['adapters_trained'] / max(1, progress.get('population_size', 1))
            progress['last_update'] = time.time()

            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️  Progress update failed: {e}")

    # 🔄 CACHE RESULT: Store for idempotency
    if idempotency_cache:
        idempotency_cache[job_id] = result_path
        print(f"🔄 Cached result: {job_id} → {result_path}")

    return result_path

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",  # Faster GPU
    memory=16384,     # More memory
    timeout=600,      # Shorter timeout
    min_containers=1, # Keep 1 container warm
    secrets=[modal.Secret.from_name("huggingface")]
)
def generate_code_modal(model_name: str, adapter_path: str, problem_name: str, buggy_code: str, config: dict, cheap_knobs: dict):
    """🎯 Generate code using CodeLlama with LoRA adapter and CA-derived cheap knobs."""
    import sys

    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("  CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))

    print(f"🤖 Modal generation: {problem_name} with {model_name}")
    print(f"   📁 Adapter: {adapter_path}")
    print(f"   🎛️  CA knobs: T={cheap_knobs.get('temperature', 0.7):.3f}, p={cheap_knobs.get('top_p', 0.9):.3f}")

    # Create GenerationRequest object and delegate to existing service
    from plugins.quixbugs_codellama.codellama_generation import GenerationRequest
    from infra.modal.codellama_service import generate_with_codellama_modal

    # Convert cheap_knobs dict to GenerationRequest parameters
    request = GenerationRequest(
        problem_name=problem_name,
        buggy_code=buggy_code,
        model_name=model_name,
        adapter_path=adapter_path,
        max_tokens=cheap_knobs.get('max_new_tokens', 512),
        temperature=cheap_knobs.get('temperature', 0.7),
        top_p=cheap_knobs.get('top_p', 0.9),
        top_k=cheap_knobs.get('top_k', 50),
        repetition_penalty=cheap_knobs.get('repetition_penalty', 1.0),
        do_sample=cheap_knobs.get('do_sample', True)
    )

    # Call the actual generation service
    result = generate_with_codellama_modal(request)

    # Return just the generated code string (not the full GenerationResult object)
    return result.generated_code


def process_generation_job(job):
    """Process a code generation job."""
    from plugins.quixbugs_codellama.codellama_generation import generate_with_codellama

    # Extract job parameters
    request = job['request']
    config = job['config']

    print(f"🤖 Generating code for: {request.problem_name}")

    # Call generation function
    result = generate_with_codellama(request, config)

    return result.generated_code

def process_evaluation_job(job):
    """Process a genome evaluation job."""
    import sys
    import json
    import time

    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("  CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))

    # Import clean Modal service for evaluation
    from infra.modal.experiment_service import evaluate_genome_modal

    # 🔥 CRITICAL FIX: Extract adapter_path from job (was missing!)
    genome_data = job['genome_data']
    adapter_path = job['adapter_path']  # ✅ This was the missing piece!
    config = job['config']

    print(f"🧬 Evaluating genome: {genome_data['id']}")
    print(f"📁 Using pre-trained adapter: {adapter_path}")

    # 📊 UPDATE PROGRESS: Evaluation started
    try:
        progress_file = Path("/cache/evolution_progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            # Update evaluation stats
            progress['status'] = 'evaluating'
            progress['message'] = f'Evaluating genome {genome_data["id"][:8]}...'
            progress['last_update'] = time.time()

            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️  Progress update failed: {e}")

    # 🔥 CRITICAL FIX: Pass adapter_path to evaluation service
    # This tells the evaluation to use the pre-trained adapter instead of training a new one
    config_with_adapter = {**config, 'adapter_path': adapter_path}
    result = evaluate_genome_modal(genome_data, config_with_adapter)

    # 📊 UPDATE PROGRESS: Update best scores if better
    try:
        if progress_file.exists() and isinstance(result, dict):
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            # Update best scores if this genome is better
            if 'bugfix' in result:
                current_best = progress.get('best_scores', {})
                for metric in ['bugfix', 'style', 'security', 'runtime', 'syntax']:
                    if metric in result:
                        current_score = result[metric]
                        if current_score > current_best.get(metric, 0):
                            current_best[metric] = current_score

                progress['best_scores'] = current_best

                # Calculate overall fitness
                total_fitness = sum(current_best.values()) / len(current_best)
                if total_fitness > progress.get('best_fitness', 0):
                    progress['best_fitness'] = total_fitness

            progress['last_update'] = time.time()

            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️  Progress update failed: {e}")

    return result

def process_test_job(job):
    """Process a code testing job."""
    from adapters.quixbugs_real import QuixBugsRealAdapter

    # Extract job parameters
    generated_code = job['generated_code']
    problem_name = job['problem_name']
    problem_data = job['problem_data']
    config = job['config']

    print(f"🧪 Testing code for: {problem_name}")

    # Create adapter and evaluate
    adapter = QuixBugsRealAdapter(config=config)
    result = adapter.evaluate_code(generated_code, problem_name, problem_data)

    return result

# ========================================
# 🛡️ DEAD LETTER QUEUE & ERROR HANDLING
# ========================================

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,
    timeout=60
)
def handle_failed_job(job, error_message: str, attempt_count: int = 1):
    """Handle failed jobs - send to DLQ after multiple failures."""
    import json
    from datetime import datetime

    job_id = job.get('job_id', 'unknown')
    job_type = job.get('job_type', 'unknown')

    print(f"💀 Job failure #{attempt_count}: {job_id} ({job_type}) - {error_message}")

    # DLQ threshold - send to DLQ after 3 failures
    max_attempts = 3

    if attempt_count >= max_attempts:
        # Send to Dead Letter Queue (file-based for simplicity)
        dlq_path = Path("/cache/dead_letter_queue")
        dlq_path.mkdir(exist_ok=True)

        dlq_entry = {
            'job_id': job_id,
            'job_type': job_type,
            'job': job,
            'error_message': error_message,
            'attempt_count': attempt_count,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed_permanently'
        }

        dlq_file = dlq_path / f"{job_id}.json"
        with open(dlq_file, 'w') as f:
            json.dump(dlq_entry, f, indent=2, default=str)

        print(f"💀 Job sent to DLQ: {dlq_file}")

        # TODO: Send alert (email, Slack, etc.)
        print(f"🚨 ALERT: Job {job_id} failed permanently after {attempt_count} attempts")

        return {'status': 'sent_to_dlq', 'path': str(dlq_file)}
    else:
        # TODO: Implement retry logic if needed
        print(f"🔄 Job could be retried (attempt {attempt_count}/{max_attempts})")
        return {'status': 'could_retry', 'attempt_count': attempt_count}

# ========================================
# 🎮 QUEUE MANAGEMENT FUNCTIONS
# ========================================

@app.function()
@modal.fastapi_endpoint(method="GET")
def queue_status():
    """Get status of all queues via HTTP endpoint."""
    try:
        cache_size = len(list(cache_index.keys())) if hasattr(cache_index, 'keys') else 0
    except:
        cache_size = 0

    return {
        'training_queue': training_queue.len(),
        'test_queue': test_queue.len(),
        'generation_queue': generation_queue.len(),
        'results_queue': results_queue.len(),
        'cache_index_size': cache_size,
        'timestamp': time.time(),
        'status': 'healthy'
    }

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,
    timeout=60
)
@modal.fastapi_endpoint(method="GET")
def check_dlq_status():
    """Check Dead Letter Queue status for monitoring via HTTP endpoint."""
    import json

    dlq_path = Path("/cache/dead_letter_queue")

    if not dlq_path.exists():
        return {
            'dlq_count': 0,
            'status': 'healthy',
            'message': 'No DLQ directory (no failures yet)',
            'timestamp': time.time()
        }

    # Count DLQ entries
    dlq_files = list(dlq_path.glob("*.json"))
    dlq_count = len(dlq_files)

    # Get recent failures (last 5)
    recent_failures = []
    for dlq_file in sorted(dlq_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        try:
            with open(dlq_file) as f:
                failure_data = json.load(f)
                recent_failures.append({
                    'job_id': failure_data.get('job_id', 'unknown'),
                    'job_type': failure_data.get('job_type', 'unknown'),
                    'error': failure_data.get('error_message', 'unknown'),
                    'timestamp': failure_data.get('timestamp', 'unknown')
                })
        except Exception as e:
            recent_failures.append({'error': f'Failed to read {dlq_file.name}: {e}'})

    status = 'healthy' if dlq_count == 0 else 'has_failures'

    return {
        'dlq_count': dlq_count,
        'status': status,
        'recent_failures': recent_failures,
        'message': f'Found {dlq_count} failed jobs in DLQ',
        'timestamp': time.time()
    }

@app.function()
@modal.fastapi_endpoint(method="POST")  # POST for destructive operation
def clear_queues():
    """Clear all queues (development only) via HTTP endpoint."""
    training_queue.clear()
    test_queue.clear()
    generation_queue.clear()
    results_queue.clear()

    return {
        'status': 'cleared',
        'timestamp': time.time()
    }

@app.function()
@modal.fastapi_endpoint(method="GET")
def health_check():
    """Complete system health check via HTTP endpoint."""
    try:
        # Check queue lengths
        queue_lengths = {
            'training_queue': training_queue.len(),
            'test_queue': test_queue.len(),
            'generation_queue': generation_queue.len(),
            'results_queue': results_queue.len()
        }

        # Check cache
        try:
            cache_size = len(list(cache_index.keys()))
        except:
            cache_size = 0

        # Overall health
        total_queued = sum(queue_lengths.values())
        is_healthy = total_queued < 1000  # Arbitrary threshold

        return {
            'status': 'healthy' if is_healthy else 'busy',
            'queues': queue_lengths,
            'cache_size': cache_size,
            'total_queued_jobs': total_queued,
            'timestamp': time.time(),
            'version': '1.0.0-queues'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,
    timeout=60
)
def start_workers_for_run(run_id: str, training_workers: int = 2, test_workers: int = 3):
    """
    Start workers for a specific run using predictable queue names.
    """
    import modal

    print(f"🚀 Starting workers for run: {run_id}")

    # Create/lookup queues using predictable names
    queue_prefix = f"coral-{run_id}"
    training_queue = modal.Queue.from_name(f"{queue_prefix}-training", create_if_missing=True)
    test_queue = modal.Queue.from_name(f"{queue_prefix}-test", create_if_missing=True)
    results_queue = modal.Queue.from_name(f"{queue_prefix}-results", create_if_missing=True)

    print(f"📡 Using queues with prefix: {queue_prefix}")

    # Shared cache index
    cache_index = {}

    # Spawn training workers
    training_futures = []
    for i in range(training_workers):
        future = training_worker_with_queues.spawn(training_queue, results_queue, cache_index)
        training_futures.append(future)
        print(f"🏗️  Started training worker {i + 1}")

    # Spawn test workers
    test_futures = []
    for i in range(test_workers):
        future = test_worker_with_queues.spawn(test_queue, results_queue, cache_index)
        test_futures.append(future)
        print(f"🧪 Started test worker {i + 1}")

    return {
        'run_id': run_id,
        'queue_prefix': queue_prefix,
        'training_workers': len(training_futures),
        'test_workers': len(test_futures),
        'status': 'started'
    }

def spawn_workers_with_queues(training_queue, test_queue, results_queue, cache_index, training_workers: int = 2, test_workers: int = 1):
    """Start worker instances with global queue references (category theory compliant)."""
    # Spawn training workers (category theory compliant)
    training_futures = []
    for i in range(training_workers):
        future = training_worker.spawn()  # No parameters - uses global queues
        training_futures.append(future)

    # Spawn test workers (category theory compliant)
    test_futures = []
    for i in range(test_workers):
        future = test_worker.spawn()  # No parameters - uses global queues
        test_futures.append(future)

    print(f"🏗️ Started {len(training_futures)} training workers, {len(test_futures)} test workers")
    print("🧮 Category theory: Workers automatically reference global queue objects")

    return {
        'training_workers': len(training_futures),
        'test_workers': len(test_futures),
        'status': 'started'
    }

# ========================================
# 🧪 EXPERIMENT ORCHESTRATION
# ========================================

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    cpu=4,
    memory=8192,
    timeout=21600  # 6 hours
)
def run_experiment_modal(config_dict):
    """🎯 Modal experiment runner that delegates to queue-based implementation."""
    return run_experiment_with_queues(config_dict)


def run_experiment_with_queues(config_dict):
    """Run experiment using queue-based coordination."""
    import sys
    import json

    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("  CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))

    print("🚀 Starting queue-based CORAL-X experiment")

    # Parse config
    if isinstance(config_dict, str):
        config_dict = json.loads(config_dict)

    # Create ephemeral queues for this experiment
    with modal.Queue.ephemeral() as training_queue, \
         modal.Queue.ephemeral() as generation_queue, \
         modal.Queue.ephemeral() as test_queue, \
         modal.Queue.ephemeral() as results_queue, \
         modal.Dict.ephemeral() as cache_index:

        # Import evolution infrastructure
        from core.config.loader import create_config_from_dict
        from core.domain.experiment import create_experiment_config, create_initial_population
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        from infra.queue_modal_executor import QueueModalExecutor

        # Create structured configs
        coral_config = create_config_from_dict(config_dict)
        exp_config = create_experiment_config(config_dict)

        # Load plugin
        plugin = QuixBugsCodeLlamaRealPlugin(config_dict)

        # Create category theory compliant queue executor (global queues)
        executor = QueueModalExecutor(config_dict)

        # Create initial population
        diversity_strength = 0.4
        run_id = config_dict.get('cache', {}).get('run_id', None)
        init_pop = create_initial_population(exp_config, diversity_strength, raw_config=config_dict, run_id=run_id)

        # Start workers with queue references
        spawn_workers_with_queues(training_queue, test_queue, results_queue, cache_index)

        # Create evolution engine with queue executor
        from core.application.evolution_engine import EvolutionEngine
        engine = EvolutionEngine(
            cfg=coral_config,
            fitness_fn=plugin.fitness_fn(),
            executor=executor,
            model_factory=plugin.model_factory(),
            dataset=plugin.dataset(),
            run_id=run_id,
            raw_config=config_dict
        )

        # Run evolution
        winners = engine.run(init_pop)

        # Create result
        best_genome = winners.best() if winners.size() > 0 else None
        result = {
            'success': True,
            'best_fitness': best_genome.fitness if best_genome else 0.0,
            'generations': exp_config.generations,
            'population_size': exp_config.population_size,
            'final_population_size': winners.size(),
            'run_location': 'modal-queues'
        }

        print("✅ Queue-based evolution completed successfully")
        return result

# ========================================
# 📊 PROGRESS TRACKING FUNCTIONS
# ========================================

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=512,
    timeout=30
)
@modal.fastapi_endpoint(method="GET")
def get_evolution_progress_modal(config_dict=None):
    """Get evolution progress from cache volume via HTTP endpoint."""
    import json

    progress_file = Path("/cache/evolution_progress.json")

    if not progress_file.exists():
        return {
            "status": "not_started",
            "message": "Evolution not started yet",
            "current_generation": 0,
            "max_generations": 0,
            "best_fitness": 0.0
        }

    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return progress
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read progress: {e}",
            "current_generation": 0,
            "max_generations": 0,
            "best_fitness": 0.0
        }

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=512,
    timeout=30
)
def update_evolution_progress_modal(progress_dict):
    """Update evolution progress in cache volume."""
    import json
    import time

    progress_file = Path("/cache/evolution_progress.json")

    # Add timestamp
    progress_dict['last_update'] = time.time()

    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_dict, f, indent=2)
        return {"success": True, "message": "Progress updated"}
    except Exception as e:
        return {"success": False, "message": f"Failed to update progress: {e}"}

# ========================================
# 🧮 CATEGORY THEORY RESTORATION COMPLETE
# Queue-based natural transformations defined above
# ========================================

if __name__ == "__main__":
    print("🪸 CORAL-X Queue-Based Modal App")
    print("Deploy with: modal deploy coral_queue_modal_app.py")
