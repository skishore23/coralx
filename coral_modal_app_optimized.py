"""
CORAL-X COST-OPTIMIZED Modal Application
Proper resource allocation to reduce costs by 80%+ while maintaining performance.
"""
import json
from pathlib import Path
import modal
import os
import time

# SINGLE PERSISTENT APP - Container-like architecture
APP_NAME = "coral-x-production-optimized"
app = modal.App(APP_NAME)

# Volume for caching models and data - persistent across deployments
coral_volume = modal.Volume.from_name("coral-x-clean-cache", create_if_missing=True)

# Base CPU image for lightweight operations
coral_cpu_image = (
    modal.Image.debian_slim()
    .pip_install([
        "numpy",
        "pyyaml", 
        "pytest"
    ])
    .env({
        "PYTHONPATH": "/root/coralx:/root",
        "WANDB_DISABLED": "true"
    })
    .add_local_dir(".", "/root/coralx")
)

# GPU image for ML operations only
coral_gpu_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install([
        "torch",
        "transformers", 
        "accelerate",
        "peft",
        "datasets",
        "numpy",
        "scipy",
        "scikit-learn",
        "pyyaml",
        "pytest",
        "huggingface_hub",
        "wandb"
    ])
    .env({
        "PYTHONPATH": "/root/coralx:/root",
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled"
    })
    .add_local_dir(".", "/root/coralx")
)

# ========================================
# 🔥 CPU-ONLY FUNCTIONS (MASSIVE SAVINGS)
# ========================================

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=512,  # 512MB is enough for JSON operations
    timeout=30   # 30 seconds max
)
def get_evolution_progress_modal() -> dict:
    """Read evolution progress - CPU only, minimal resources."""
    import json
    from pathlib import Path
    
    try:
        progress_file = Path("/cache/evolution_progress.json")
        
        if not progress_file.exists():
            return {
                'status': 'no_progress_file',
                'message': 'Evolution not started yet'
            }
        
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        # Add computed fields for display
        if 'start_time' in progress_data and 'last_update' in progress_data:
            progress_data['runtime_minutes'] = (progress_data['last_update'] - progress_data['start_time']) / 60
        
        return progress_data
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=512,  # 512MB is enough for JSON operations
    timeout=30   # 30 seconds max
)
def get_emergent_alerts_modal() -> list:
    """Read emergent behavior alerts - CPU only, minimal resources."""
    import json
    from pathlib import Path
    
    try:
        alerts_file = Path("/cache/emergent_alerts.json")
        
        if not alerts_file.exists():
            return []
        
        with open(alerts_file, 'r') as f:
            alerts = json.load(f)
        
        # Return most recent 50 alerts
        return alerts[-50:] if len(alerts) > 50 else alerts
        
    except Exception as e:
        return [{"error": str(e), "timestamp": time.time()}]

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,  # 1GB for file operations
    timeout=300   # 5 minutes max
)
def save_realtime_results_modal(results: list):
    """Save benchmark results - CPU only, minimal resources."""
    import json
    from pathlib import Path
    
    try:
        results_file = Path("/cache/realtime_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to existing results
        existing_results = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        
        existing_results.extend(results)
        
        with open(results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
            
        print(f"✅ Saved {len(results)} new benchmark results")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        raise RuntimeError(f"FAIL-FAST: Could not save results: {e}")

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=2,
    memory=2048,  # 2GB for dataset operations
    timeout=600   # 10 minutes max
)
def setup_quixbugs_dataset_modal() -> dict:
    """Setup QuixBugs dataset - CPU only, optimized resources."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import dataset service
    from infra.modal.dataset_service import setup_dataset_modal
    
    return setup_dataset_modal()

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,  # 1GB for file operations
    timeout=300   # 5 minutes max
)
def cleanup_corrupted_adapters():
    """Clean up corrupted adapters - CPU only, minimal resources."""
    import shutil
    from pathlib import Path
    
    adapter_dir = Path("/cache/adapters")
    if not adapter_dir.exists():
        return {"cleaned": 0, "message": "No adapter directory found"}
    
    cleaned_count = 0
    for adapter_path in adapter_dir.glob("adapter_*"):
        if adapter_path.is_dir():
            # Check for required files
            required_files = ['adapter_config.json', 'adapter_model.safetensors']
            missing_files = [f for f in required_files if not (adapter_path / f).exists()]
            
            if missing_files:
                print(f"🗑️  Cleaning corrupted adapter: {adapter_path.name}")
                shutil.rmtree(adapter_path)
                cleaned_count += 1
    
    return {"cleaned": cleaned_count, "message": f"Cleaned {cleaned_count} corrupted adapters"}

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume}, 
    cpu=1,
    memory=1024,  # 1GB for file operations
    timeout=300   # 5 minutes max
)
def clear_all_adapters():
    """Clear all adapters - CPU only, minimal resources."""
    import shutil
    from pathlib import Path
    
    adapter_dir = Path("/cache/adapters")
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "cleared", "message": "All adapters cleared"}
    
    return {"status": "no_adapters", "message": "No adapter directory found"}

@app.function(
    image=coral_cpu_image,
    volumes={"/cache": coral_volume},
    cpu=2,
    memory=2048,  # 2GB for dependency checks
    timeout=600   # 10 minutes max
)
def ensure_dependencies_modal() -> dict:
    """Check dependencies - CPU only, optimized resources."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    start_time = time.time()
    results = {}
    
    # Check model cache
    model_cache = Path("/cache/models")
    results["model_cache"] = model_cache.exists()
    
    # Check dataset
    dataset_cache = Path("/cache/quixbugs_dataset")
    results["dataset"] = dataset_cache.exists()
    
    # Check adapters directory
    adapters_dir = Path("/cache/adapters")
    results["adapters_dir"] = adapters_dir.exists()
    
    results["total_time"] = time.time() - start_time
    results["all_ready"] = all(results[key] for key in ["model_cache", "dataset", "adapters_dir"])
    
    return results

# ========================================
# 💻 RIGHT-SIZED GPU FUNCTIONS
# ========================================

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A10G",  # Cheaper than A100 for inference
    memory=8192,  # 8GB instead of 16GB
    timeout=600   # 10 minutes instead of 30
)
def generate_code_modal(model_name: str, adapter_path: str, problem_name: str, buggy_code: str, config: dict, cheap_knobs: dict = None) -> str:
    """Generate code - A10G GPU, optimized memory."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import clean generation service
    from infra.modal.codellama_service import generate_with_codellama_modal
    from coral.domain.codellama_generation import GenerationRequest
    
    # Get generation parameters
    gen_config = config.get('generation', {})
    
    # Handle cheap knobs or use defaults
    if cheap_knobs is None:
        temperature = 0.7
        top_p = 0.9  
        top_k = 50
        max_tokens = gen_config.get('max_tokens', 512)
        repetition_penalty = 1.0
        do_sample = True
    else:
        temperature = cheap_knobs['temperature']
        top_p = cheap_knobs['top_p']
        top_k = cheap_knobs['top_k']
        max_tokens = cheap_knobs['max_new_tokens']
        repetition_penalty = cheap_knobs['repetition_penalty']
        do_sample = cheap_knobs['do_sample']
    
    # Create generation request
    request = GenerationRequest(
        problem_name=problem_name,
        buggy_code=buggy_code,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        adapter_path=adapter_path,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample
    )
    
    # Generate using clean service
    result = generate_with_codellama_modal(request)
    return result.generated_code

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",  # Keep A100 for training
    memory=16384,     # 16GB for training
    timeout=1800      # 30 minutes for training
)
def train_lora_modal(base_model: str, heavy_key, save_path: str, config: dict) -> str:
    """Train LoRA adapter - A100 for training efficiency."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import LoRA training service
    from infra.modal.lora_service import train_lora_adapter_modal
    
    return train_lora_adapter_modal(base_model, heavy_key, save_path, config)

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",  # Keep A100 for full evaluation
    memory=16384,     # 16GB for evaluation
    timeout=1800      # 30 minutes for evaluation
)
def evaluate_genome_modal(genome_data: dict, config: dict) -> dict:
    """Evaluate genome - A100 for comprehensive evaluation."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import clean Modal service
    from infra.modal.experiment_service import evaluate_genome_modal
    
    return evaluate_genome_modal(genome_data, config)

@app.function(
    image=coral_gpu_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",  # Keep for experiment orchestration
    memory=16384,     # 16GB instead of 32GB
    timeout=21600,    # 6 hours instead of 12
    secrets=[modal.Secret.from_name("huggingface")]
)
def run_experiment_modal(config_dict: dict) -> dict:
    """Run complete experiment - optimized resources."""
    import sys
    import time
    import json
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))

    print(f"🚀 [MODAL-OPTIMIZED] CORAL-X Evolution Experiment Starting")
    start_time = time.time()
    
    # Initialize progress tracking
    progress_file = Path("/cache/evolution_progress.json")
    
    try:
        # Import evolution infrastructure
        from coral.config.loader import create_config_from_dict
        from coral.domain.experiment import create_experiment_config, create_initial_population
        from coral.application.evolution_engine import EvolutionEngine
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        from infra.modal_executor import create_executor_from_config
        
        # Create structured configs
        coral_config = create_config_from_dict(config_dict)
        exp_config = create_experiment_config(config_dict)
        
        # Load plugin
        plugin = QuixBugsCodeLlamaRealPlugin(config_dict)
        
        # Create executor from configuration 
        executor = create_executor_from_config(config_dict)
        
        # Create initial population
        diversity_strength = 0.4
        run_id = config_dict.get('cache', {}).get('run_id', None)
        init_pop = create_initial_population(exp_config, diversity_strength, raw_config=config_dict, run_id=run_id)
        
        # Create evolution engine
        engine = EvolutionEngine(
            cfg=coral_config,
            fitness_fn=plugin.fitness_fn(),
            executor=executor,
            model_factory=plugin.model_factory(),
            dataset=plugin.dataset(),
            run_id=run_id
        )
        
        # Run evolution
        winners = engine.run(init_pop)
        
        evolution_time = time.time() - start_time
        
        # Create result
        best_genome = winners.best() if winners.size() > 0 else None
        best_scores = {}
        
        if best_genome and hasattr(best_genome, 'multi_scores') and best_genome.multi_scores:
            best_scores = {
                'bugfix': best_genome.multi_scores.bugfix,
                'style': best_genome.multi_scores.style,
                'security': best_genome.multi_scores.security,
                'runtime': best_genome.multi_scores.runtime,
                'syntax': best_genome.multi_scores.syntax
            }
        
        result = {
            'success': True,
            'best_fitness': best_genome.fitness if best_genome else 0.0,
            'best_scores': best_scores,
            'generations': exp_config.generations,
            'population_size': exp_config.population_size,
            'experiment_time': evolution_time,
            'final_population_size': winners.size(),
            'run_location': 'modal-optimized'
        }
        
        print(f"✅ [MODAL-OPTIMIZED] Evolution completed in {evolution_time:.1f}s")
        return result
        
    except Exception as e:
        print(f"❌ [MODAL-OPTIMIZED] Evolution failed: {e}")
        raise RuntimeError(f"FAIL-FAST: Modal experiment failed: {e}")

# ========================================
# 🔧 UTILITY FUNCTIONS
# ========================================

@app.function(
    image=coral_cpu_image,
    cpu=4,  # CPU for code evaluation with timeout protection
    memory=4096,  # 4GB for code execution
    timeout=300   # 5 minutes max
)
def evaluate_code_modal(generated_code: str, problem_name: str, problem_data: dict) -> dict:
    """Evaluate generated code - CPU with timeout protection."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import evaluation service
    from infra.modal.codellama_service import evaluate_code_modal
    
    return evaluate_code_modal(generated_code, problem_name, problem_data)

@app.local_entrypoint()
def test_optimized_functions():
    """Test optimized Modal functions locally."""
    
    print("🧪 Testing optimized Modal functions...")
    
    # Test CPU functions
    progress = get_evolution_progress_modal.remote()
    print(f"✅ Progress check: {progress}")
    
    alerts = get_emergent_alerts_modal.remote()
    print(f"✅ Alerts check: {len(alerts)} alerts")
    
    deps = ensure_dependencies_modal.remote()
    print(f"✅ Dependencies: {deps}")
    
    print("🎯 Optimized functions tested successfully!")

if __name__ == "__main__":
    test_optimized_functions() 