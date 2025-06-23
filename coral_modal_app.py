"""
CORAL-X Clean Modal Application
Production Modal deployment using clean architecture - SINGLE PERSISTENT APP.
"""
import json
from pathlib import Path
import modal
import os
import time

# SINGLE PERSISTENT APP - Container-like architecture
# App name should be unique and persistent
APP_NAME = "coral-x-production"
app = modal.App(APP_NAME)

# Volume for caching models and data - persistent across deployments
coral_volume = modal.Volume.from_name("coral-x-clean-cache", create_if_missing=True)

# Image with dependencies - packages from config
coral_image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Add git for dataset cloning
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
        "WANDB_DISABLED": "true",  # Disable wandb completely
        "WANDB_MODE": "disabled"   # Alternative disable method
    })
    .add_local_dir(".", "/root/coralx")  # Add entire project as package (must be last)
    # Dataset is cached in volume at /cache/quixbugs_dataset
)

# CONTAINER 1: Experiment Orchestration
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=2048,
    timeout=300
)
def start_realtime_monitoring_modal(config_dict: dict):
    """Start real-time benchmark monitoring in background on Modal."""
    print("🚀 [MODAL] Starting real-time benchmark monitoring...")
    
    # Start the monitor in background (async)
    realtime_benchmark_monitor_modal.spawn(config_dict)
    
    print("✅ [MODAL] Real-time monitoring started in background")
    return {"status": "started", "monitor": "running"}


@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=32768,
    timeout=43200,  # 12 hours for full 20-generation evolution cycles (increased from 4 hours)
    secrets=[modal.Secret.from_name("huggingface")]
)
def run_experiment_modal(config_dict: dict) -> dict:
    """Run complete CORAL-X evolution experiment on Modal with real-time progress updates."""
    import sys
    import time
    import json
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))

    print(f"🚀 [MODAL] CORAL-X Evolution Experiment Starting")
    start_time = time.time()
    
    # Initialize progress tracking
    progress_file = Path("/cache/evolution_progress.json")
    _write_progress_update(progress_file, {
        'status': 'starting',
        'start_time': start_time,
        'current_generation': 0,
        'max_generations': config_dict['execution']['generations'],
        'best_fitness': 0.0,
        'best_scores': {},
        'cache_stats': {'hits': 0, 'misses': 0, 'hit_rate': 0.0},
        'diversity_stats': {},
        'training_stats': {'adapters_trained': 0, 'total_gpu_hours': 0.0}
    })
    
    try:
        # Import evolution infrastructure
        from coral.config.loader import create_config_from_dict
        from coral.domain.experiment import create_experiment_config, create_initial_population
        from coral.application.evolution_engine import EvolutionEngine
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        from infra.modal_executor import create_executor_from_config
        
        # Update progress
        _write_progress_update(progress_file, {
            'status': 'initializing',
            'message': 'Loading plugins and configurations...'
        })
        
        # Create structured configs
        coral_config = create_config_from_dict(config_dict)
        exp_config = create_experiment_config(config_dict)
        
        # Load plugin
        plugin = QuixBugsCodeLlamaRealPlugin(config_dict)
        
        # Create executor from configuration 
        executor = create_executor_from_config(config_dict)
        
        # Update progress
        _write_progress_update(progress_file, {
            'status': 'creating_population',
            'message': 'Creating initial population...'
        })
        
        # Create initial population with balanced cache-clone strategy  
        diversity_strength = 0.4  # Target 3-8x cache efficiency
        run_id = config_dict.get('cache', {}).get('run_id', None)
        
        init_pop = create_initial_population(exp_config, diversity_strength, raw_config=config_dict, run_id=run_id)
        
        # Update progress
        _write_progress_update(progress_file, {
            'status': 'evolution_starting',
            'message': f'Starting evolution with {init_pop.size()} genomes...',
            'population_size': init_pop.size()
        })
        
        # Create evolution engine with progress callback
        engine = EvolutionEngine(
            cfg=coral_config,
            fitness_fn=plugin.fitness_fn(),
            executor=executor,
            model_factory=plugin.model_factory(),
            dataset=plugin.dataset(),
            run_id=run_id
        )
        
        # Add progress callback to evolution engine
        def progress_callback(generation, population, best_genome, cache_stats=None):
            """Callback to update evolution progress."""
            try:
                scores = {}
                if best_genome and hasattr(best_genome, 'multi_scores') and best_genome.multi_scores:
                    scores = {
                        'bugfix': best_genome.multi_scores.bugfix,
                        'style': best_genome.multi_scores.style,
                        'security': best_genome.multi_scores.security,
                        'runtime': best_genome.multi_scores.runtime,
                        'syntax': best_genome.multi_scores.syntax
                    }
                
                # Calculate diversity stats
                diversity_stats = {
                    'unique_genomes': len(set(str(g.id) for g in population.genomes if hasattr(g, 'id'))),
                    'total_genomes': population.size(),
                    'avg_diversity': sum(g.fitness for g in population.genomes if hasattr(g, 'fitness')) / max(population.size(), 1)
                }
                
                progress_update = {
                    'status': 'evolving',
                    'current_generation': generation,
                    'max_generations': config_dict['execution']['generations'],
                    'best_fitness': best_genome.fitness if best_genome else 0.0,
                    'best_scores': scores,
                    'diversity_stats': diversity_stats,
                    'elapsed_time': time.time() - start_time,
                    'message': f'Generation {generation}: Best fitness {best_genome.fitness:.3f}' if best_genome else f'Generation {generation} in progress'
                }
                
                # Add cache stats if available
                if cache_stats:
                    progress_update['cache_stats'] = cache_stats
                
                _write_progress_update(progress_file, progress_update)
                
            except Exception as e:
                print(f"⚠️  Progress callback error: {e}")
        
        # Set progress callback on engine
        if hasattr(engine, 'set_progress_callback'):
            engine.set_progress_callback(progress_callback)
        
        print(f"✅ [MODAL] Running evolution with real-time progress tracking...")
        
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
            'run_location': 'modal'
        }
        
        # Final progress update
        _write_progress_update(progress_file, {
            'status': 'completed',
            'success': True,
            'final_result': result,
            'completion_time': time.time(),
            'total_time': evolution_time
        })
        
        print(f"✅ [MODAL] Evolution completed successfully in {evolution_time:.1f}s")
        return result
        
    except Exception as e:
        evolution_time = time.time() - start_time
        error_result = {
            'success': False,
            'error': str(e),
            'experiment_time': evolution_time,
            'run_location': 'modal'
        }
        
        # Error progress update
        _write_progress_update(progress_file, {
            'status': 'failed',
            'success': False,
            'error': str(e),
            'completion_time': time.time(),
            'total_time': evolution_time
        })
        
        print(f"❌ [MODAL] Evolution failed: {e}")
        raise RuntimeError(f"FAIL-FAST: Modal experiment failed: {e}")


def _write_progress_update(progress_file: Path, update_data: dict):
    """Write progress update to file for real-time streaming."""
    try:
        # Read existing progress
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
        
        # Merge update with existing data
        existing_data.update(update_data)
        existing_data['last_update'] = time.time()
        
        # Write updated progress
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
    except Exception as e:
        print(f"⚠️  Could not write progress update: {e}")

# CONTAINER 2: Genome Evaluation
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=32768,
    timeout=1800
)
def evaluate_genome_modal(genome_data: dict, config: dict) -> dict:
    """Evaluate genome using Modal with clean architecture."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import clean Modal service
    from infra.modal.experiment_service import evaluate_genome_modal
    
    # Evaluate using clean architecture
    return evaluate_genome_modal(genome_data, config)

# CONTAINER 3: LoRA Training
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=32768,
    timeout=1800
)
def train_lora_modal(base_model: str, heavy_key, save_path: str, config: dict) -> str:
    """Train LoRA adapter on Modal using clean architecture."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import LoRA training service
    from infra.modal.lora_service import train_lora_adapter_modal
    
    # Train adapter using clean architecture
    return train_lora_adapter_modal(base_model, heavy_key, save_path, config)

# CONTAINER 4: Code Generation
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=1800  # Increased from 600s to 1800s (30 minutes) to match other compute functions
)
def generate_code_modal(model_name: str, adapter_path: str, problem_name: str, buggy_code: str, config: dict, cheap_knobs: dict = None) -> str:
    """Generate code with LoRA adapter on Modal using clean architecture."""
    import sys
    import os
    from pathlib import Path
    
    print(f"🎯 [PRODUCTION] CODE GENERATION START")
    print(f"   • App: {APP_NAME}")
    print(f"   • Model: {model_name}")
    print(f"   • Adapter: {adapter_path}")
    print(f"   • Problem: {problem_name}")
    
    # IMMEDIATE TYPE VALIDATION
    try:
        print(f"   • Cache volume mounted: {os.path.exists('/cache')}")
    except Exception as print_error:
        raise RuntimeError(f"FAIL-FAST: Basic print operations failed: {print_error}. Types: model_name={type(model_name)}, adapter_path={type(adapter_path)}, problem_name={type(problem_name)}, buggy_code={type(buggy_code)}, config={type(config)}")
    
    # Validate input types explicitly
    if not isinstance(model_name, str):
        raise RuntimeError(f"FAIL-FAST: model_name must be str, got {type(model_name)}: {repr(model_name)}")
    if not isinstance(adapter_path, str):
        raise RuntimeError(f"FAIL-FAST: adapter_path must be str, got {type(adapter_path)}: {repr(adapter_path)}")
    if not isinstance(problem_name, str):
        raise RuntimeError(f"FAIL-FAST: problem_name must be str, got {type(problem_name)}: {repr(problem_name)}")
    if not isinstance(buggy_code, str):
        raise RuntimeError(f"FAIL-FAST: buggy_code must be str, got {type(buggy_code)}: {repr(buggy_code)}")
    if not isinstance(config, dict):
        raise RuntimeError(f"FAIL-FAST: config must be dict, got {type(config)}: {repr(config)}")
    
    print(f"✅ All input types validated successfully")
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import clean generation service
    from infra.modal.codellama_service import generate_with_codellama_modal
    from coral.domain.codellama_generation import GenerationRequest
    
    # Get generation parameters from config and cheap knobs
    gen_config = config.get('generation', {})
    
    # 🧮 CATEGORICAL APPROACH: Make data flow explicit and pure
    print(f"🔍 MODAL FUNCTION PARAMETER ANALYSIS:")
    print(f"   • cheap_knobs type: {type(cheap_knobs)}")
    print(f"   • cheap_knobs value: {cheap_knobs}")
    print(f"   • config has generation: {'generation' in config}")
    print(f"   • config has cheap_knobs: {'cheap_knobs' in config}")
    
    # TWO-LOOP ARCHITECTURE: FAIL-FAST if cheap knobs missing during evolution
    if cheap_knobs is None:
        print(f"❌ CATEGORICAL VIOLATION: cheap_knobs is None")
        print(f"   • This violates two-loop architecture purity")
        print(f"   • Parameters should flow: CAFeatures → CheapKnobs → Dict → Modal")
        print(f"   • Boundary condition detected at Modal serialization layer")
        
        # Check if this is benchmarking vs evolution
        is_explicit_benchmark = config.get('_benchmark_mode') == True
        
        if not is_explicit_benchmark:
            # This is evolution - cheap knobs are REQUIRED
            raise ValueError(
                f"FAIL-FAST: Categorical composition broken - cheap_knobs required for evolution. "
                f"CA features → CheapKnobs → Dict chain failed at Modal boundary. "
                f"Two-loop architecture demands CA-derived parameters."
            )
        
        # Explicit benchmark mode - use neutral parameters
        print(f"⚠️ BENCHMARK MODE: Using neutral parameters (no CA derivation)")
        temperature = 0.7
        top_p = 0.9  
        top_k = 50
        max_tokens = gen_config.get('max_tokens', 512)
        repetition_penalty = 1.0
        do_sample = True
        
    else:
        # CATEGORICAL SUCCESS: cheap_knobs properly composed through morphisms
        print(f"✅ CATEGORICAL COMPOSITION SUCCESSFUL:")
        print(f"   • Temperature: {cheap_knobs.get('temperature', 'MISSING'):.3f} (complexity-driven)")
        print(f"   • Top-p: {cheap_knobs.get('top_p', 'MISSING'):.3f} (intensity-driven)")
        print(f"   • Top-k: {cheap_knobs.get('top_k', 'MISSING')} (convergence-driven)")
        print(f"   • Repetition penalty: {cheap_knobs.get('repetition_penalty', 'MISSING'):.3f} (periodicity-driven)")
        print(f"   • Max tokens: {cheap_knobs.get('max_new_tokens', 'MISSING')} (feature-derived)")
        print(f"   • Sampling: {cheap_knobs.get('do_sample', 'MISSING')} (CA-controlled)")
        
        # Validate all required keys are present (morphism completeness)
        required_keys = ['temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_new_tokens', 'do_sample']
        missing_keys = [key for key in required_keys if key not in cheap_knobs]
        
        if missing_keys:
            raise ValueError(f"FAIL-FAST: Categorical morphism incomplete - missing keys: {missing_keys}")
        
        # Use cheap knobs for generation parameters (PURE ASSIGNMENT)
        temperature = cheap_knobs['temperature']
        top_p = cheap_knobs['top_p']
        top_k = cheap_knobs['top_k']
        max_tokens = cheap_knobs['max_new_tokens']
        repetition_penalty = cheap_knobs['repetition_penalty']
        do_sample = cheap_knobs['do_sample']
    
    # Create properly structured generation request with runtime parameters
    try:
        print(f"🔧 Creating GenerationRequest with runtime parameters...")
        print(f"   • Problem: {repr(problem_name)} ({len(problem_name)} chars)")
        print(f"   • Buggy code: {len(buggy_code)} chars")
        print(f"   • Runtime params: T={temperature}, p={top_p}, k={top_k}, tokens={max_tokens}")
        
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
        print(f"✅ GenerationRequest created directly: {request}")
        
    except Exception as request_error:
        print(f"❌ ERROR creating GenerationRequest:")
        print(f"   • Error type: {type(request_error)}")
        print(f"   • Error message: {request_error}")
        raise RuntimeError(f"FAIL-FAST: GenerationRequest creation failed: {request_error}")
    
    # Generate using clean service with error isolation
    try:
        print(f"🔄 About to call generate_with_codellama_modal...")
        print(f"   • Request: {request}")
        result = generate_with_codellama_modal(request)
        print(f"✅ generate_with_codellama_modal completed successfully")
    except Exception as generation_error:
        print(f"❌ generate_with_codellama_modal failed:")
        print(f"   • Error type: {type(generation_error)}")
        print(f"   • Error message: {generation_error}")
        print(f"   • Request was: {request}")
        raise RuntimeError(f"FAIL-FAST: CodeLlama generation failed: {generation_error}")
    
    # POST-PROCESSING: Add missing imports based on function content
    generated_code = result.generated_code
    
    # Auto-detect and add missing imports
    missing_imports = []
    
    # Check for string module usage
    if any(pattern in generated_code for pattern in ['string.digits', 'string.ascii', 'string.letters']):
        if 'import string' not in generated_code:
            missing_imports.append('import string')
    
    # Check for math module usage
    if any(pattern in generated_code for pattern in ['math.', 'sqrt(', 'log(', 'exp(']):
        if 'import math' not in generated_code:
            missing_imports.append('import math')
    
    # Check for random module usage
    if any(pattern in generated_code for pattern in ['random.', 'randint(', 'choice(']):
        if 'import random' not in generated_code:
            missing_imports.append('import random')
    
    # Add missing imports at the top
    if missing_imports:
        import_block = '\n'.join(missing_imports) + '\n\n'
        generated_code = import_block + generated_code
        print(f"🔧 Added missing imports: {missing_imports}")
    
    return generated_code

# CONTAINER 5: QuixBugs Evaluation
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=4,
    memory=8192,
    timeout=600
)
def evaluate_code_modal(
    generated_code: str,
    problem_name: str,
    problem_data: dict
) -> dict:
    """
    Evaluate generated code using Modal infrastructure where QuixBugs dataset is available.
    """
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    print(f"🧪 [MODAL] Evaluating code for {problem_name}")
    print(f"   • Generated code: {len(generated_code)} characters")
    
    try:
        # Import Modal evaluation infrastructure
        from coral.domain.quixbugs_evaluation import evaluate_quixbugs_code
        from infra.modal.experiment_service import load_real_test_cases_modal
        
        # Load test cases using Modal infrastructure
        dataset_path = "/cache/quixbugs_dataset"
        test_cases = load_real_test_cases_modal(problem_name, dataset_path)
        
        # Evaluate using domain logic
        evaluation_result = evaluate_quixbugs_code(
            generated_code=generated_code,
            problem=problem_data,
            test_cases=test_cases
        )
        
        # Convert to dict for serialization
        result_dict = {
            'bugfix': evaluation_result.bugfix,
            'style': evaluation_result.style,
            'security': evaluation_result.security,
            'runtime': evaluation_result.runtime,
            'test_cases_passed': evaluation_result.test_cases_passed,
            'test_cases_run': evaluation_result.test_cases_run,
            'syntax_valid': evaluation_result.compilation_status == 'success',
            'compilation_status': evaluation_result.compilation_status,
            'function_defined': evaluation_result.function_defined
        }
        
        print(f"✅ [MODAL] Evaluation completed: {evaluation_result.test_cases_passed}/{evaluation_result.test_cases_run} tests")
        return result_dict
        
    except Exception as e:
        print(f"❌ [MODAL] Evaluation failed: {e}")
        return {
            'error': str(e),
            'bugfix': 0.0,
            'style': 0.0,
            'security': 0.0,
            'runtime': 0.0,
            'test_cases_passed': 0,
            'test_cases_run': 0,
            'syntax_valid': False,
            'compilation_status': 'failed',
            'function_defined': False
        }

# CONTAINER 6: Complete Benchmark Runner
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB", 
    memory=16384,
    timeout=10800  # 3 hours for comprehensive benchmarks (15 problems × 12 minutes each)
)
def run_complete_benchmark_modal(
    config_dict: dict,
    num_problems: int = 3
) -> dict:
    """
    SMART BENCHMARK: Train once, evaluate many - leverages CORAL-X cache architecture.
    Uses shared adapters across all problems for 10x speedup.
    """
    import sys
    import time
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    print(f"🎯 [MODAL] Complete Benchmark Runner Starting")
    print(f"   • Problems to test: {num_problems}")
    
    # Hardcoded evolved parameters from your successful evolution (fitness: 0.945)
    evolved_params = {
        'r': 16,                    # From Modal logs: evolved_r16_
        'lora_alpha': 32.0,        # From Modal logs: _a32.0_
        'lora_dropout': 0.1,       # Standard evolved dropout
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'task_type': 'CAUSAL_LM',
        'adapter_type': 'dora',    # Your successful evolution used DoRA
        'max_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50
    }
    
    # Baseline parameters for comparison
    baseline_params = {
        'r': 8,                     # Conservative rank
        'lora_alpha': 16.0,        # Conservative alpha
        'lora_dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj'],  # Fewer modules
        'task_type': 'CAUSAL_LM',
        'adapter_type': 'lora',    # Standard LoRA
        'max_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50
    }
    
    print(f"🧬 Evolved params: r={evolved_params['r']}, α={evolved_params['lora_alpha']}, {evolved_params['adapter_type']}")
    print(f"📊 Baseline params: r={baseline_params['r']}, α={baseline_params['lora_alpha']}, {baseline_params['adapter_type']}")
    
    try:
        # Load QuixBugs problems
        from adapters.quixbugs_real import QuixBugsRealAdapter
        
        adapter = QuixBugsRealAdapter()
        problems = list(adapter.problems())
        
        if not problems:
            raise RuntimeError("FAIL-FAST: No problems available from dataset")
        
        # Select problems for testing - use more diverse problems
        if num_problems <= 5:
            # Small test - use first few
            selected_problems = problems[:num_problems]
        else:
            # Comprehensive test - select diverse problems for better coverage
            import random
            random.seed(42)  # Reproducible selection
            selected_problems = random.sample(problems, min(num_problems, len(problems)))
        print(f"📋 Selected problems: {[p.get('name') for p in selected_problems]}")
        
        # 🧠 SMART CACHE: Run benchmarks with shared adapter reuse
        results = []
        base_model = "codellama/CodeLlama-7b-Python-hf"
        
        for i, problem in enumerate(selected_problems, 1):
            problem_name = problem.get('name', f'problem_{i}')
            print(f"\n{'='*60}")
            print(f"🎯 PROBLEM {i}/{num_problems}: {problem_name}")
            print(f"{'='*60}")
            
            # Generate with evolved parameters (will cache adapter after first use)
            print(f"🧬 Generating with EVOLVED parameters...")
            evolved_start = time.time()
            evolved_code = benchmark_inference_modal.remote(
                base_model,
                problem_name,
                problem.get('buggy_code', ''),
                'evolved',
                {'adapter_config': evolved_params}
            )
            evolved_time = time.time() - evolved_start
            
            # Generate with baseline parameters (will cache adapter after first use)
            print(f"📊 Generating with BASELINE parameters...")
            baseline_start = time.time()
            baseline_code = benchmark_inference_modal.remote(
                base_model,
                problem_name,
                problem.get('buggy_code', ''),
                'baseline', 
                {'adapter_config': baseline_params}
            )
            baseline_time = time.time() - baseline_start
            
            # Evaluate both codes
            print(f"🧪 Evaluating EVOLVED code...")
            evolved_eval = evaluate_code_modal.remote(
                evolved_code,
                problem_name,
                problem
            )
            
            print(f"🧪 Evaluating BASELINE code...")
            baseline_eval = evaluate_code_modal.remote(
                baseline_code,
                problem_name,
                problem
            )
            
            # Calculate improvements - with defensive defaults
            evolved_bugfix = evolved_eval.get('bugfix', 0) if isinstance(evolved_eval, dict) else 0
            evolved_style = evolved_eval.get('style', 0) if isinstance(evolved_eval, dict) else 0
            evolved_security = evolved_eval.get('security', 0) if isinstance(evolved_eval, dict) else 0
            evolved_runtime = evolved_eval.get('runtime', 0) if isinstance(evolved_eval, dict) else 0
            evolved_tests = evolved_eval.get('test_cases_passed', 0) if isinstance(evolved_eval, dict) else 0
            
            baseline_bugfix = baseline_eval.get('bugfix', 0) if isinstance(baseline_eval, dict) else 0
            baseline_style = baseline_eval.get('style', 0) if isinstance(baseline_eval, dict) else 0
            baseline_security = baseline_eval.get('security', 0) if isinstance(baseline_eval, dict) else 0
            baseline_runtime = baseline_eval.get('runtime', 0) if isinstance(baseline_eval, dict) else 0
            baseline_tests = baseline_eval.get('test_cases_passed', 0) if isinstance(baseline_eval, dict) else 0
            
            improvements = {
                'bugfix': evolved_bugfix - baseline_bugfix,
                'style': evolved_style - baseline_style,
                'security': evolved_security - baseline_security,
                'runtime': evolved_runtime - baseline_runtime,
                'tests_passed_diff': evolved_tests - baseline_tests
            }
            
            # Store results
            result = {
                'problem': problem_name,
                'evolved_result': {
                    'generated_code': evolved_code,
                    'evaluation_result': evolved_eval,
                    'generation_time': evolved_time
                },
                'baseline_result': {
                    'generated_code': baseline_code,
                    'evaluation_result': baseline_eval,
                    'generation_time': baseline_time
                },
                'improvements': improvements
            }
            
            results.append(result)
            
            # Show immediate results
            print(f"📊 RESULTS for {problem_name}:")
            print(f"   • Evolved tests: {evolved_eval.get('test_cases_passed', 0)}/{evolved_eval.get('test_cases_run', 0)}")
            print(f"   • Baseline tests: {baseline_eval.get('test_cases_passed', 0)}/{baseline_eval.get('test_cases_run', 0)}")
            
            for metric, improvement in improvements.items():
                if metric != 'tests_passed_diff':
                    status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "➖" if improvement == 0 else "❌"
                    print(f"   • {metric.capitalize()}: {improvement:+.3f} {status}")
        
        # Calculate overall results
        total_problems = len(results)
        
        if total_problems == 0:
            raise RuntimeError("FAIL-FAST: No problems completed - cannot calculate benchmark results")
        
        evolved_wins = sum(1 for r in results if sum(r['improvements'].values()) > 0)
        
        # Average improvements - with safe division
        avg_improvements = {}
        for metric in ['bugfix', 'style', 'security', 'runtime', 'tests_passed_diff']:
            avg_improvements[metric] = sum(r['improvements'][metric] for r in results) / total_problems
        
        analysis = {
            'summary': {
                'total_problems': total_problems,
                'evolved_wins': evolved_wins,
                'baseline_wins': total_problems - evolved_wins,
                'win_rate': evolved_wins / total_problems * 100
            },
            'average_improvements': avg_improvements,
            'evolved_parameters': evolved_params,
            'baseline_parameters': baseline_params,
            'detailed_results': results
        }
        
        print(f"\n🎉 COMPLETE BENCHMARK RESULTS:")
        print(f"🏆 Evolved wins: {evolved_wins}/{total_problems} ({evolved_wins/total_problems*100:.1f}%)")
        print(f"📈 Average improvements:")
        for metric, improvement in avg_improvements.items():
            if metric != 'tests_passed_diff':
                status = "🔥" if improvement > 0.1 else "✅" if improvement > 0 else "➖"
                print(f"   • {metric.capitalize()}: {improvement:+.3f} {status}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Complete benchmark failed: {e}")
        raise RuntimeError(f"FAIL-FAST: Complete benchmark failed: {e}")

# CONTAINER 7: Legacy Benchmarking
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB", 
    memory=16384,
    timeout=1800  # Increased from 900s to 1800s for consistency with generation functions
)
def benchmark_inference_modal(
    model_name: str,
    problem_name: str, 
    buggy_code: str,
    model_type: str,  # "evolved" or "baseline"
    benchmark_config: dict
) -> str:
    """
    Dedicated Modal function for benchmark inference.
    Creates evolved vs baseline adapters with specific parameters.
    """
    import sys
    import os
    import time
    from pathlib import Path
    
    # Add coralx to Python path (same as generate_code_modal)
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    print(f"🎯 BENCHMARK INFERENCE: {model_type.upper()} model for {problem_name}")
    
    # Get adapter configuration
    adapter_config = benchmark_config.get('adapter_config', {})
    
    # 🧠 SMART CACHE: Use shared adapters instead of per-problem training
    adapter_id = f"{model_type}_r{adapter_config.get('r', 8)}_a{adapter_config.get('lora_alpha', 16)}"
    adapter_path = f"/cache/benchmark_adapters/{adapter_id}_shared"  # Shared across problems
    
    print(f"   🔧 Adapter config: {adapter_config}")
    print(f"   💾 Adapter path: {adapter_path}")
    
    # Check if adapter exists
    adapter_exists = Path(adapter_path).exists()
    
    if not adapter_exists:
        print(f"🔄 Training {model_type} adapter with specific parameters...")
        
        try:
            from coral.domain.lora_training import train_codellama_lora
            from coral.domain.lora_training import LoRAConfig
            
            # Create LoRA config with benchmark parameters (using correct AdapterConfig parameters)
            lora_config = LoRAConfig(
                r=adapter_config.get('r', 8),  # 🔥 FIXED: Use r= not rank=
                alpha=adapter_config.get('lora_alpha', 16.0),
                dropout=adapter_config.get('lora_dropout', 0.1),
                target_modules=adapter_config.get('target_modules', ['q_proj', 'v_proj']),
                task_type=adapter_config.get('task_type', 'CAUSAL_LM'),
                adapter_type=adapter_config.get('adapter_type', 'lora')
            )
            
            print(f"   📝 {model_type.title()} LoRA config: {lora_config}")
            
            # Train the adapter (this will use the same training data as evolution)
            training_start = time.time()
            
            # Create HeavyGenes object for train_codellama_lora
            from infra.adapter_cache import HeavyGenes
            heavy_genes = HeavyGenes(
                rank=lora_config.r,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                target_modules=lora_config.target_modules,
                adapter_type=lora_config.adapter_type,
                run_id=None  # No specific run_id for benchmarking
            )
            
            result_path = train_codellama_lora(
                base_ckpt=model_name,
                heavy_genes=heavy_genes,
                save_to=adapter_path
            )
            training_time = time.time() - training_start
            
            # Function now returns string path directly - verify it exists
            from pathlib import Path
            if not Path(result_path).exists():
                raise RuntimeError(f"Benchmark adapter training failed: adapter not found at {result_path}")
                
            print(f"✅ {model_type.title()} adapter trained in {training_time:.1f}s: {result_path}")
            
        except Exception as training_error:
            print(f"❌ {model_type.title()} adapter training failed: {training_error}")
            raise RuntimeError(f"FAIL-FAST: Benchmark adapter training failed: {training_error}")
    
    else:
        print(f"✅ Using cached {model_type} adapter: {adapter_path}")
    
    # Generate code with the benchmark adapter
    try:
        from infra.modal.codellama_service import generate_with_codellama_modal
        from coral.domain.codellama_generation import GenerationRequest
        
        # Create generation request with model-specific parameters
        gen_params = adapter_config.copy()
        
        request = GenerationRequest(
            problem_name=problem_name,
            buggy_code=buggy_code,
            model_name=model_name,
            max_tokens=gen_params.get('max_tokens', 512),
            temperature=gen_params.get('temperature', 0.7),
            adapter_path=adapter_path,
            top_p=gen_params.get('top_p', 0.9),
            top_k=gen_params.get('top_k', 50)
        )
        
        print(f"   🎯 Generating with {model_type} adapter...")
        print(f"   📋 Generation params: temp={request.temperature}, top_p={request.top_p}")
        
        generation_start = time.time()
        result = generate_with_codellama_modal(request)
        generation_time = time.time() - generation_start
        
        print(f"✅ {model_type.title()} generation completed in {generation_time:.1f}s")
        print(f"   📝 Generated code: {len(result.generated_code)} characters")
        
        return result.generated_code
        
    except Exception as generation_error:
        print(f"❌ {model_type.title()} generation failed: {generation_error}")
        raise RuntimeError(f"FAIL-FAST: Benchmark generation failed: {generation_error}")


@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=1800
)
def run_benchmarks_modal(config_dict: dict, winners_data: list) -> dict:
    """Run benchmarks on Modal using clean architecture."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Import benchmarking components
    from benchmarks.benchmark_runner import run_benchmarks
    from coral.config.loader import create_config_from_dict
    
    # Reconstruct config and run benchmarks
    config = create_config_from_dict(config_dict)
    return run_benchmarks(config, winners_data)

# CONTAINER 6: Model Caching
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=1800
)
def download_model_to_cache():
    """Download model to cache volume."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from pathlib import Path
    
    # Use standard values for model caching (consistent with config)
    model_name = "codellama/CodeLlama-7b-Python-hf"  # Standard CodeLlama model
    cache_dir = "/cache/models"  # Modal volume path for model cache
    
    # Check if model is properly cached (look for actual model files)
    cache_path = Path(cache_dir)
    model_cache_path = cache_path / f"models--{model_name.replace('/', '--')}"
    
    # Check for actual model weight files, not just the directory
    if model_cache_path.exists():
        blob_files = list((model_cache_path / "blobs").glob("*"))
        total_size = sum(f.stat().st_size for f in blob_files if f.is_file())
        
        # CodeLlama-7B should be ~13GB, if less than 1GB then incomplete
        if total_size > 1_000_000_000:  # 1GB threshold
            return {"status": "already_cached", "cache_dir": cache_dir, "size_gb": total_size / 1e9}
        else:
            print(f"🔄 Model cache incomplete ({total_size / 1e6:.1f}MB), re-downloading...")
    
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Clean up memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {"status": "downloaded", "cache_dir": cache_dir}

# CONTAINER 7: Smart Dataset Management
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=2,
    memory=4096,
    timeout=600
)
def setup_quixbugs_dataset_modal() -> dict:
    """Smart QuixBugs dataset setup - auto-download if missing."""
    import sys
    import os
    from pathlib import Path
    
    print(f"📦 [SMART] QuixBugs Dataset Setup")
    print(f"   • Cache volume mounted: {os.path.exists('/cache')}")
    
    # Check if dataset already exists
    dataset_path = Path("/cache/quixbugs_dataset")
    if dataset_path.exists() and len(list(dataset_path.glob("*.py"))) > 10:
        print(f"✅ Dataset already exists with {len(list(dataset_path.glob('**/*.py')))} Python files")
        return {
            "status": "already_exists",
            "dataset_path": str(dataset_path),
            "file_count": len(list(dataset_path.glob("**/*.py")))
        }
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    # Auto-download dataset
    try:
        print(f"🔄 Downloading QuixBugs dataset to {dataset_path}...")
        
        # Create dataset directory
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Import and run dataset setup
        from infra.modal.dataset_service import cache_quixbugs_dataset_modal
        
        # Download and cache dataset
        final_path = cache_quixbugs_dataset_modal()
        
        # Verify download
        python_files = list(final_path.glob("**/*.py"))
        json_files = list(final_path.glob("**/*.json"))
        
        print(f"✅ Dataset downloaded successfully")
        print(f"   • Python files: {len(python_files)}")
        print(f"   • JSON files: {len(json_files)}")
        
        return {
            "status": "downloaded",
            "dataset_path": str(final_path),
            "python_files": len(python_files),
            "json_files": len(json_files)
        }
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "dataset_path": str(dataset_path)
        }

# CONTAINER 8: Real-Time Benchmark Monitor
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=2,
    memory=4096,
    timeout=43200  # 12 hours to run alongside evolution (increased from 4 hours)
)
def realtime_benchmark_monitor_modal(config_dict: dict) -> dict:
    """
    Modal-native real-time benchmark monitoring.
    Monitors /cache/adapters for new adapters and benchmarks them instantly.
    """
    import sys
    import time
    import json
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if not coralx_path.exists():
        raise RuntimeError("FAIL-FAST: CoralX codebase not found in Modal environment")
    sys.path.insert(0, str(coralx_path))
    
    print(f"🔄 [MODAL] Starting real-time benchmark monitor...")
    
    # Monitor state
    benchmarked_adapters = set()
    benchmark_results = []
    adapter_cache_dir = Path("/cache/adapters")
    
    start_time = time.time()
    max_runtime = 4 * 3600  # 4 hours max
    
    while (time.time() - start_time) < max_runtime:
        try:
            if not adapter_cache_dir.exists():
                print(f"   📁 Waiting for adapter cache directory...")
                time.sleep(60)
                continue
            
            # Find new adapter directories
            new_adapters = []
            for adapter_dir in adapter_cache_dir.glob("adapter_*"):
                if not adapter_dir.is_dir():
                    continue
                    
                adapter_hash = adapter_dir.name
                
                # Skip if already benchmarked
                if adapter_hash in benchmarked_adapters:
                    continue
                    
                # Check if adapter is complete
                required_files = ['adapter_config.json', 'adapter_model.safetensors']
                if not all((adapter_dir / f).exists() for f in required_files):
                    continue  # Still training
                    
                new_adapters.append((adapter_dir, adapter_hash))
            
            # Benchmark new adapters
            for adapter_dir, adapter_hash in new_adapters:
                print(f"\n🎯 [MODAL] NEW ADAPTER DETECTED: {adapter_hash}")
                
                try:
                    # Quick benchmark on Modal
                    result = benchmark_single_adapter_modal.remote(
                        str(adapter_dir), 
                        adapter_hash, 
                        config_dict
                    )
                    
                    benchmark_results.append(result)
                    benchmarked_adapters.add(adapter_hash)
                    
                    print(f"   ✅ Benchmarked: {result.get('avg_score', 0.0):.3f} avg score")
                    
                except Exception as e:
                    print(f"   ⚠️ Benchmark failed for {adapter_hash}: {e}")
            
            # Save results periodically
            if benchmark_results and len(benchmark_results) % 5 == 0:
                save_realtime_results_modal.remote(benchmark_results)
            
            # Sleep before next check
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(60)
    
    # Final save
    if benchmark_results:
        save_realtime_results_modal.remote(benchmark_results)
    
    print(f"\n🏁 [MODAL] Real-time monitoring complete:")
    print(f"   • Total adapters benchmarked: {len(benchmark_results)}")
    print(f"   • Runtime: {(time.time() - start_time)/3600:.1f} hours")
    
    return {
        'total_adapters': len(benchmark_results),
        'results': benchmark_results,
        'runtime_hours': (time.time() - start_time) / 3600
    }


@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=1800
)
def benchmark_single_adapter_modal(adapter_dir_str: str, adapter_hash: str, config_dict: dict) -> dict:
    """
    Benchmark a single adapter on Modal using real QuixBugs problems.
    """
    import sys
    import time
    import json
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    sys.path.insert(0, str(coralx_path))
    
    print(f"🧪 [MODAL] REAL-TIME BENCHMARK: {adapter_hash}")
    
    try:
        from adapters.quixbugs_real import QuixBugsRealAdapter
        
        adapter_dir = Path(adapter_dir_str)
        
        # Load adapter configuration
        adapter_config_path = adapter_dir / 'adapter_config.json'
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Extract parameters
        r = adapter_config.get('r', 8)
        alpha = adapter_config.get('lora_alpha', 16.0)
        adapter_type = 'dora' if adapter_config.get('use_dora', False) else 'lora'
        
        print(f"   🔧 Adapter params: r={r}, α={alpha}, type={adapter_type}")
        
        # Load test problems
        adapter_provider = QuixBugsRealAdapter()
        problems = list(adapter_provider.problems())
        
        if not problems:
            raise RuntimeError("No problems available for benchmarking")
        
        # Select 3 representative problems for quick benchmark
        # Use adapter-specific seed for diverse problem coverage
        import random
        import hashlib
        
        # Create deterministic but unique seed from adapter hash
        seed_source = f"{adapter_hash}_benchmark"
        adapter_seed = int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16)
        random.seed(adapter_seed)
        
        # Filter to test-only problems to avoid training data contamination
        test_problem_names = {
            'bitcount', 'bucketsort', 'detect_cycle', 'find_first_in_sorted',
            'gcd', 'get_factors', 'hanoi', 'is_valid_parenthesization',
            'kheapsort', 'lcs_length', 'lis', 'max_sublist_sum',
            'mergesort', 'next_palindrome', 'pascal', 'powerset',
            'quicksort', 'rpn_eval', 'shortest_paths', 'sieve',
            'sqrt', 'subsequences', 'topological_ordering', 'wrap'
        }
        
        # Only use test problems for benchmarking
        test_problems_only = [p for p in problems if p.get('name') in test_problem_names]
        
        if len(test_problems_only) < 3:
            print(f"   ⚠️ Warning: Only {len(test_problems_only)} test problems available")
            test_problems = random.sample(problems, min(3, len(problems)))  # Fallback
        else:
            test_problems = random.sample(test_problems_only, min(3, len(test_problems_only)))
        
        print(f"   🎯 Testing on {len(test_problems)} problems: {[p.get('name') for p in test_problems]}")
        
        # Validate no training data contamination
        selected_names = {p.get('name') for p in test_problems}
        if not selected_names.issubset(test_problem_names):
            contaminated = selected_names - test_problem_names
            print(f"   ⚠️ Warning: Training data contamination detected: {contaminated}")
        
        # Run quick benchmarks using Modal infrastructure
        benchmark_start = time.time()
        problem_results = []
        total_score = 0.0
        
        for problem in test_problems:
            problem_name = problem.get('name', 'unknown')
            
            try:
                # Generate code with this adapter
                generated_code = generate_code_modal.remote(
                    "codellama/CodeLlama-7b-Python-hf",
                    str(adapter_dir),
                    problem_name,
                    problem.get('buggy_code', ''),
                    config_dict
                )
                
                # Evaluate the generated code
                eval_result = evaluate_code_modal.remote(
                    generated_code,
                    problem_name,
                    problem
                )
                
                # Calculate quick score
                bugfix_score = eval_result.get('bugfix', 0.0)
                tests_passed = eval_result.get('test_cases_passed', 0)
                tests_run = eval_result.get('test_cases_run', 1)
                
                quick_score = bugfix_score
                total_score += quick_score
                
                problem_results.append({
                    'problem': problem_name,
                    'score': quick_score,
                    'tests_passed': tests_passed,
                    'tests_run': tests_run,
                    'generated_code_length': len(generated_code)
                })
                
                print(f"   • {problem_name}: {quick_score:.3f} ({tests_passed}/{tests_run} tests)")
                
            except Exception as e:
                print(f"   ⚠️ Problem {problem_name} failed: {e}")
                problem_results.append({
                    'problem': problem_name,
                    'score': 0.0,
                    'error': str(e)
                })
        
        avg_score = total_score / max(len(test_problems), 1)
        benchmark_time = time.time() - benchmark_start
        
        result = {
            'adapter_hash': adapter_hash,
            'adapter_dir': adapter_dir_str,
            'timestamp': time.time(),
            'avg_score': avg_score,
            'adapter_params': {'r': r, 'alpha': alpha, 'type': adapter_type},
            'problem_results': problem_results,
            'benchmark_time': benchmark_time,
            'problems_tested': len(test_problems)
        }
        
        print(f"   ✅ Real-time benchmark complete: {avg_score:.3f} avg score ({benchmark_time:.1f}s)")
        return result
        
    except Exception as e:
        print(f"   ❌ Adapter benchmark failed: {e}")
        return {
            'adapter_hash': adapter_hash,
            'adapter_dir': adapter_dir_str,
            'timestamp': time.time(),
            'error': str(e),
            'avg_score': 0.0
        }


@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=2048,
    timeout=300
)
def save_realtime_results_modal(results: list):
    """Save real-time benchmark results to Modal volume."""
    import json
    import time
    from pathlib import Path
    
    try:
        results_dir = Path("/cache/realtime_benchmarks")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"realtime_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_adapters': len(results),
                'results': results
            }, f, indent=2)
        
        print(f"💾 [MODAL] Saved real-time results: {len(results)} adapters")
        
    except Exception as e:
        print(f"⚠️ Failed to save real-time results: {e}")


# CONTAINER 9: Smart Dependency Resolver
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=2,
    memory=4096,
    timeout=1800
)
def ensure_dependencies_modal() -> dict:
    """Ensure all dependencies are available before starting evolution."""
    import os
    import time
    from pathlib import Path
    
    print(f"🔧 [SMART] Ensuring all dependencies are ready...")
    
    results = {
        "model_cache": False,
        "dataset": False,
        "adapters_dir": False,
        "total_time": 0
    }
    
    start_time = time.time()
    
    # 1. Ensure model cache directory exists
    model_cache_dir = Path("/cache/models")
    if not model_cache_dir.exists():
        print(f"🔧 Creating model cache directory...")
        model_cache_dir.mkdir(parents=True, exist_ok=True)
    results["model_cache"] = True
    print(f"✅ Model cache directory ready: {model_cache_dir}")
    
    # 2. Ensure dataset exists
    dataset_path = Path("/cache/quixbugs_dataset")
    if not dataset_path.exists() or len(list(dataset_path.glob("**/*.py"))) < 10:
        print(f"🔄 Setting up QuixBugs dataset...")
        dataset_result = setup_quixbugs_dataset_modal.remote()
        if dataset_result["status"] in ["downloaded", "already_exists"]:
            results["dataset"] = True
            print(f"✅ Dataset ready: {dataset_result['status']}")
        else:
            print(f"❌ Dataset setup failed: {dataset_result.get('error', 'unknown')}")
    else:
        results["dataset"] = True
        print(f"✅ Dataset already ready")
    
    # 3. Ensure adapters directory exists
    adapters_dir = Path("/cache/adapters")
    if not adapters_dir.exists():
        print(f"🔧 Creating adapters directory...")
        adapters_dir.mkdir(parents=True, exist_ok=True)
    results["adapters_dir"] = True
    print(f"✅ Adapters directory ready: {adapters_dir}")
    
    results["total_time"] = time.time() - start_time
    results["all_ready"] = all(results[key] for key in ["model_cache", "dataset", "adapters_dir"])
    
    print(f"🎯 Dependency check complete in {results['total_time']:.2f}s")
    print(f"   • All dependencies ready: {results['all_ready']}")
    
    return results

# Additional utility containers...
# (keeping the rest of the functions from the original file)

@app.local_entrypoint()
def test_modal_functions():
    """Test the deployed Modal functions - production testing."""
    print(f"🧪 Testing {APP_NAME} functions...")
    
    # Test model download
    try:
        result = download_model_to_cache.remote()
        print(f"✅ Model caching: {result}")
    except Exception as e:
        print(f"❌ Model caching failed: {e}")
    
    # Test cache coordination (NEW)
    try:
        print(f"\n🔧 Testing cache coordination...")
        cache_result = test_cache_coordination_modal.remote()
        print(f"✅ Cache coordination: {cache_result.get('success', False)}")
        if not cache_result.get('success', False):
            print(f"   ❌ Error: {cache_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Cache coordination test failed: {e}")
    
    print(f"✅ {APP_NAME} function tests complete!")

@app.function(volumes={"/cache": coral_volume})
def cleanup_corrupted_adapters():
    """Clean up corrupted and problematic adapters from Modal volume."""
    import shutil
    from pathlib import Path
    from datetime import datetime, timedelta
    
    print("🧹 ADAPTER CLEANUP STARTING")
    print("=" * 50)
    
    adapters_dir = Path("/cache/adapters")
    if not adapters_dir.exists():
        print("❌ Adapters directory not found")
        return {"error": "Adapters directory not found"}
    
    cleanup_stats = {
        'total_adapters': 0,
        'corrupted_removed': 0,
        'empty_removed': 0,
        'training_tmp_removed': 0,
        'good_adapters': 0,
        'storage_freed_mb': 0,
        'old_adapters_removed': 0
    }
    
    # Get current time for age-based cleanup
    now = datetime.now()
    cutoff_time = now - timedelta(hours=72)  # Remove adapters older than 3 days
    
    # Scan all adapter directories
    for adapter_path in adapters_dir.iterdir():
        if not adapter_path.is_dir():
            continue
            
        if not adapter_path.name.startswith('adapter_'):
            continue
            
        cleanup_stats['total_adapters'] += 1
        
        try:
            # Check directory contents
            contents = list(adapter_path.iterdir())
            
            # Case 1: Empty directory
            if not contents:
                print(f"🗑️  EMPTY: {adapter_path.name}")
                cleanup_stats['storage_freed_mb'] += _get_dir_size_mb(adapter_path)
                shutil.rmtree(adapter_path)
                cleanup_stats['empty_removed'] += 1
                continue
            
            # Case 2: Only training_tmp directory (corruption from failed training)
            if len(contents) == 1 and contents[0].name.startswith('training_tmp'):
                print(f"🗑️  CORRUPTED (training_tmp only): {adapter_path.name}")
                cleanup_stats['storage_freed_mb'] += _get_dir_size_mb(adapter_path)
                shutil.rmtree(adapter_path)
                cleanup_stats['training_tmp_removed'] += 1
                continue
            
            # Case 3: Contains training_tmp alongside real files (cleanup needed)
            training_tmp_dirs = [c for c in contents if c.is_dir() and c.name.startswith('training_tmp')]
            if training_tmp_dirs:
                print(f"🧹 CLEANING training_tmp from: {adapter_path.name}")
                for tmp_dir in training_tmp_dirs:
                    cleanup_stats['storage_freed_mb'] += _get_dir_size_mb(tmp_dir)
                    shutil.rmtree(tmp_dir)
                # Don't remove the whole adapter, just the training_tmp parts
                contents = list(adapter_path.iterdir())  # Refresh contents
            
            # Case 4: Check for required adapter files
            has_config = any(f.name == 'adapter_config.json' for f in contents if f.is_file())
            has_model = any(f.name.endswith('.safetensors') or f.name.endswith('.bin') for f in contents if f.is_file())
            
            if not has_config or not has_model:
                print(f"🗑️  INCOMPLETE: {adapter_path.name} (missing config={not has_config}, model={not has_model})")
                cleanup_stats['storage_freed_mb'] += _get_dir_size_mb(adapter_path)
                shutil.rmtree(adapter_path)
                cleanup_stats['corrupted_removed'] += 1
                continue
            
            # Case 5: Age-based cleanup (if older than cutoff)
            mod_time = datetime.fromtimestamp(adapter_path.stat().st_mtime)
            if mod_time < cutoff_time:
                print(f"🗑️  OLD: {adapter_path.name} (created {mod_time.strftime('%Y-%m-%d %H:%M')})")
                cleanup_stats['storage_freed_mb'] += _get_dir_size_mb(adapter_path)
                shutil.rmtree(adapter_path)
                cleanup_stats['old_adapters_removed'] += 1
                continue
            
            # Case 6: Valid adapter - keep it
            adapter_size = _get_dir_size_mb(adapter_path)
            print(f"✅ GOOD: {adapter_path.name} ({adapter_size:.1f}MB)")
            cleanup_stats['good_adapters'] += 1
            
        except Exception as e:
            print(f"❌ Error processing {adapter_path.name}: {e}")
            continue
    
    print(f"\n✅ CLEANUP COMPLETED:")
    print(f"   • Total adapters scanned: {cleanup_stats['total_adapters']}")
    print(f"   • Empty directories removed: {cleanup_stats['empty_removed']}")
    print(f"   • Training_tmp corrupted removed: {cleanup_stats['training_tmp_removed']}")
    print(f"   • Incomplete adapters removed: {cleanup_stats['corrupted_removed']}")
    print(f"   • Old adapters removed: {cleanup_stats['old_adapters_removed']}")
    print(f"   • Good adapters preserved: {cleanup_stats['good_adapters']}")
    print(f"   • Storage freed: {cleanup_stats['storage_freed_mb']:.1f} MB")
    
    # Also clean up legacy files
    legacy_files = list(adapters_dir.glob("lora_adapter_*"))
    legacy_files.extend(list(adapters_dir.glob("training_output")))
    
    legacy_removed = 0
    for legacy_item in legacy_files:
        if legacy_item.is_dir():
            # Check if it's an old-style lora adapter
            mod_time = datetime.fromtimestamp(legacy_item.stat().st_mtime)
            if mod_time < cutoff_time:
                print(f"🗑️  LEGACY: {legacy_item.name}")
                cleanup_stats['storage_freed_mb'] += _get_dir_size_mb(legacy_item)
                shutil.rmtree(legacy_item)
                legacy_removed += 1
    
    if legacy_removed > 0:
        print(f"   • Legacy adapters removed: {legacy_removed}")
    
    return cleanup_stats

def _get_dir_size_mb(path) -> float:
    """Get directory size in MB."""
    try:
        from pathlib import Path
        path = Path(path)
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)
    except:
        return 0.0

@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    scaledown_window=300,
    timeout=1800
)
def test_adapter_functionality():
    """Test if adapters are loading and working correctly despite PEFT warnings."""
    
    print("🧪 ADAPTER FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test with a simple known adapter
    from pathlib import Path
    
    adapters_dir = Path("/cache/adapters")
    available_adapters = [d for d in adapters_dir.iterdir() if d.is_dir() and d.name.startswith('adapter_')]
    
    if not available_adapters:
        return {"error": "No adapters found"}
    
    # Pick the first available adapter
    test_adapter = available_adapters[0]
    print(f"🎯 Testing adapter: {test_adapter.name}")
    
    try:
        # Load base model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "codellama/CodeLlama-7b-Python-hf"
        model_cache_dir = "/cache/models"
        
        print(f"📥 Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=model_cache_dir,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        print(f"🔍 Base model state:")
        print(f"   • Has peft_config: {hasattr(base_model, 'peft_config')}")
        print(f"   • Model type: {type(base_model).__name__}")
        print(f"   • Has generate: {hasattr(base_model, 'generate')}")
        
        # Load adapter
        from peft import PeftModel
        print(f"🔗 Loading adapter from: {test_adapter}")
        
        model_with_adapter = PeftModel.from_pretrained(base_model, str(test_adapter))
        
        print(f"🔍 Adapter-loaded model state:")
        print(f"   • Has peft_config: {hasattr(model_with_adapter, 'peft_config')}")
        print(f"   • Model type: {type(model_with_adapter).__name__}")
        print(f"   • Has generate: {hasattr(model_with_adapter, 'generate')}")
        
        # Test simple generation
        test_prompt = "def fibonacci(n):\n    # Fix this buggy function\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)  # This is correct but inefficient\n\n# Fixed version:\ndef fibonacci_fixed(n):"
        
        print(f"🎯 Testing generation with adapter...")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model_with_adapter.device)
        
        with torch.no_grad():
            outputs = model_with_adapter.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_text[len(test_prompt):].strip()
        
        print(f"✅ Generation successful!")
        print(f"📝 Generated code (first 200 chars): {generated_code[:200]}...")
        
        # Test if adapter parameters are being used
        adapter_params = sum(p.numel() for p in model_with_adapter.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_with_adapter.parameters())
        adapter_ratio = adapter_params / total_params * 100
        
        print(f"📊 Adapter analysis:")
        print(f"   • Trainable parameters: {adapter_params:,}")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Adapter ratio: {adapter_ratio:.3f}%")
        
        # Success indicators
        success_indicators = []
        if adapter_params > 0:
            success_indicators.append("✅ Adapter parameters detected")
        if len(generated_code) > 20:
            success_indicators.append("✅ Code generation working")
        if adapter_ratio > 0.01:  # Should be ~0.1-1% for LoRA
            success_indicators.append("✅ Reasonable adapter ratio")
        
        print(f"\n🎯 FUNCTIONALITY TEST RESULTS:")
        for indicator in success_indicators:
            print(f"   {indicator}")
        
        return {
            "success": True,
            "adapter_tested": test_adapter.name,
            "trainable_params": adapter_params,
            "adapter_ratio": adapter_ratio,
            "generated_code_length": len(generated_code),
            "success_indicators": len(success_indicators)
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "adapter_tested": test_adapter.name
        }

@app.function(volumes={"/cache": coral_volume})
def clear_all_adapters():
    """NUCLEAR OPTION: Clear all adapters from Modal cache."""
    import shutil
    from pathlib import Path
    
    print("💥 NUCLEAR CACHE CLEAR STARTING")
    print("=" * 50)
    print("⚠️  WARNING: This will remove ALL cached adapters!")
    
    adapters_dir = Path("/cache/adapters")
    if not adapters_dir.exists():
        print("✅ No adapters directory found - already clean")
        return {"status": "already_clean", "adapters_removed": 0}
    
    # Count adapters before removal
    adapters_before = len([d for d in adapters_dir.iterdir() if d.is_dir() and d.name.startswith('adapter_')])
    
    # Calculate storage to be freed
    total_size_mb = 0
    for adapter_path in adapters_dir.iterdir():
        if adapter_path.is_dir():
            total_size_mb += _get_dir_size_mb(adapter_path)
    
    print(f"🔍 Found: {adapters_before} adapters ({total_size_mb:.1f} MB)")
    
    # Remove entire adapters directory
    shutil.rmtree(adapters_dir)
    
    # Recreate empty directory
    adapters_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ NUCLEAR CLEAR COMPLETED:")
    print(f"   • Adapters removed: {adapters_before}")
    print(f"   • Storage freed: {total_size_mb:.1f} MB")
    print(f"   • Cache directory recreated: {adapters_dir}")
    
    return {
        "status": "cleared",
        "adapters_removed": adapters_before,
        "storage_freed_mb": total_size_mb,
        "cache_dir": str(adapters_dir)
    }

@app.function(image=coral_image)
def check_dora_availability():
    """Check if DoRA is available in the current peft version."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if coralx_path.exists():
        sys.path.insert(0, str(coralx_path))
    
    try:
        # Check peft version
        import peft
        peft_version = peft.__version__ if hasattr(peft, '__version__') else "unknown"
        print(f"📦 PEFT Version: {peft_version}")
        
        # Check DoRA availability the same way as lora_training.py
        try:
            import inspect
            from peft import LoraConfig as PeftLoraConfig
            
            # Check if LoraConfig has use_dora parameter
            sig = inspect.signature(PeftLoraConfig.__init__)
            dora_available = 'use_dora' in sig.parameters
            
            print(f"🔍 DoRA Support Check:")
            print(f"   • LoraConfig parameters: {list(sig.parameters.keys())}")
            print(f"   • use_dora parameter present: {dora_available}")
            
            if dora_available:
                print(f"✅ DoRA is AVAILABLE (peft>={peft_version})")
            else:
                print(f"❌ DoRA is NOT AVAILABLE (peft={peft_version}, need peft>=0.10)")
                
            return {
                "peft_version": peft_version,
                "dora_available": dora_available,
                "lora_config_params": list(sig.parameters.keys())
            }
            
        except Exception as e:
            print(f"❌ Error checking DoRA: {e}")
            return {"error": str(e), "peft_version": peft_version}
            
    except Exception as e:
        print(f"❌ Error importing peft: {e}")
        return {"error": f"peft import failed: {e}"}

@app.function(image=coral_image, volumes={"/cache": coral_volume})
def test_emergent_behavior_tracking():
    """Test emergent behavior tracking to ensure logs appear properly."""
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if coralx_path.exists():
        sys.path.insert(0, str(coralx_path))
    
    try:
        from coral.domain.emergent_behavior_integration import SimpleEmergentTracker
        
        # Test with Modal volume path
        output_dir = Path("/cache/emergent_behavior")
        print(f"🧪 TESTING EMERGENT BEHAVIOR TRACKING")
        print(f"   📁 Output directory: {output_dir}")
        
        # Create tracker
        tracker = SimpleEmergentTracker(output_dir)
        print(f"✅ SimpleEmergentTracker created successfully")
        
        # Test tracking with dummy data
        print(f"🔍 Testing emergent behavior detection...")
        
        # Create dummy data for testing
        from coral.domain.genome import CASeed, LoRAConfig
        dummy_seed = CASeed(grid_size=8, rule=30, steps=10, initial_state=[1]*64)
        dummy_lora = LoRAConfig(r=8, alpha=16.0, dropout=0.1, target_modules=['q_proj', 'v_proj'])
        
        behaviors = tracker.track_evaluation(
            genome_id="test_genome_001",
            problem_name="test_problem",
            ca_seed=dummy_seed,
            lora_config=dummy_lora,
            fitness_score=0.85,
            multi_objective_scores={'bugfix': 0.9, 'style': 0.8, 'security': 0.7, 'runtime': 0.9},
            code_output="def test_function(): return True",
            test_results={'passed': 5, 'total': 5},
            evaluation_time=1.5,
            metadata={'generation': 1}
        )
        
        if behaviors:
            print(f"🎯 EMERGENT BEHAVIORS DETECTED: {len(behaviors)}")
            for behavior in behaviors:
                print(f"   🌟 {behavior.behavior_type}: {behavior.description}")
                print(f"      📊 Confidence: {behavior.confidence:.3f}")
        else:
            print(f"📊 No emergent behaviors detected in test (normal for test data)")
        
        # Check if files were created
        if output_dir.exists():
            files = list(output_dir.iterdir())
            print(f"📁 Files created in {output_dir}: {[f.name for f in files]}")
        else:
            print(f"📁 Output directory not created yet: {output_dir}")
        
        print(f"✅ Emergent behavior tracking test completed successfully")
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "behaviors_detected": len(behaviors) if behaviors else 0,
            "tracker_initialized": True
        }
        
    except Exception as e:
        print(f"❌ Emergent behavior tracking test failed: {e}")
        import traceback
        print(f"🔍 Error details: {traceback.format_exc()}")
        return {
            "status": "failed",
            "error": str(e),
            "tracker_initialized": False
        }

# CONTAINER 1: Real-time Progress Reading
@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,
    timeout=30
)
def get_evolution_progress_modal() -> dict:
    """Read current evolution progress from Modal volume for real-time streaming."""
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
        
        if 'current_generation' in progress_data and 'max_generations' in progress_data:
            progress_data['progress_percent'] = (progress_data['current_generation'] / max(progress_data['max_generations'], 1)) * 100
        
        return progress_data
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'message': 'Could not read progress file'
        }


@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    cpu=1,
    memory=1024,
    timeout=30
)
def get_emergent_alerts_modal() -> list:
    """Read emergent behavior alerts from Modal volume."""
    import json
    from pathlib import Path
    
    try:
        # Try multiple possible alert file locations
        alert_files = [
            Path("/cache/emergent_behavior/progress_log.json"),
            Path("/cache/emergent_behavior/alerts.json"),
            Path("/cache/emergent_simple/progress_log.json"),
            Path("/cache/emergent_simple/alerts.json")
        ]
        
        alerts = []
        
        for alert_file in alert_files:
            if alert_file.exists():
                try:
                    with open(alert_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract alerts from different possible formats
                    if 'alerts' in data:
                        alerts.extend(data['alerts'])
                    elif 'recent_behaviors' in data:
                        # Convert behaviors to alert format
                        for behavior in data['recent_behaviors']:
                            alerts.append({
                                'pattern_type': behavior.get('behavior_type', 'unknown'),
                                'confidence': behavior.get('confidence', 0),
                                'genome_id': behavior.get('genome_id', 'unknown'),
                                'problem_name': behavior.get('problem', 'unknown'),
                                'description': behavior.get('description', 'Emergent behavior detected'),
                                'generation': behavior.get('generation', 0),
                                'timestamp': behavior.get('timestamp', 'unknown')
                            })
                    break  # Use first available file
                except Exception as parse_error:
                    continue  # Try next file
        
        # Return recent alerts only (last 5)
        return alerts[-5:] if len(alerts) > 5 else alerts
        
    except Exception as e:
        return []


def _write_emergent_alert(alert_data: dict):
    """Write an emergent behavior alert to the alerts file for real-time streaming."""
    try:
        alerts_file = Path("/cache/emergent_behavior/alerts.json")
        alerts_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing alerts
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                data = json.load(f)
        else:
            data = {'alerts': []}
        
        # Add new alert with timestamp
        alert_data['timestamp'] = time.time()
        data['alerts'].append(alert_data)
        
        # Keep only last 50 alerts to prevent file bloat
        if len(data['alerts']) > 50:
            data['alerts'] = data['alerts'][-50:]
        
        # Write updated alerts
        with open(alerts_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"⚠️  Could not write emergent alert: {e}")

@app.function(
    image=coral_image,
    volumes={"/cache": coral_volume},
    gpu="A100-40GB",
    memory=16384,
    timeout=1800
)
def test_cache_coordination_modal():
    """Test cache coordination between training and generation phases."""
    
    print("🧪 CACHE COORDINATION TEST")
    print("=" * 50)
    
    import sys
    from pathlib import Path
    
    # Add coralx to Python path
    coralx_path = Path("/root/coralx")
    if coralx_path.exists():
        sys.path.insert(0, str(coralx_path))
    
    try:
        # Import required modules
        from coral.domain.genome import Genome, CASeed, LoRAConfig
        from infra.adapter_cache import HeavyGenes
        import numpy as np
        
        # Create the EXACT genome configuration from the failing logs
        print("🧬 Creating test genome with failing configuration...")
        
        ca_seed = CASeed(
            grid=np.random.randint(0, 2, (8, 8)),
            rule=190,
            steps=23
        )
        
        lora_cfg = LoRAConfig(
            r=4,
            alpha=32.0,
            dropout=0.15,
            target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
            adapter_type='dora'
        )
        
        genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_cfg,
            id='gen0_genome0003',  # Use the exact ID from logs
            run_id='dora_test_v1'  # Use the exact run_id from logs
        )
        
        print(f"✅ Test genome created: {genome.id}")
        print(f"   • Run ID: {genome.run_id}")
        print(f"   • Adapter type: {genome.lora_cfg.adapter_type}")
        print(f"   • LoRA params: r={genome.lora_cfg.r}, α={genome.lora_cfg.alpha}")
        
        # Phase 1: Calculate training-time hash
        print(f"\n🔧 PHASE 1: Training-time hash calculation...")
        
        training_heavy_genes = HeavyGenes.from_lora_config(
            genome.lora_cfg, 
            run_id=genome.run_id
        )
        training_hash = training_heavy_genes.to_hash()
        
        print(f"   📊 Training hash: {training_hash}")
        print(f"   📁 Expected adapter path: /cache/adapters/adapter_{training_hash}")
        
        # Check if this matches the failing logs
        expected_failing_hash = "0c04b775a4280b97"
        if training_hash == expected_failing_hash:
            print(f"   ✅ Hash matches failing logs!")
        else:
            print(f"   ❌ Hash mismatch! Expected: {expected_failing_hash}, Got: {training_hash}")
        
        # Phase 2: Simulate Modal serialization/deserialization
        print(f"\n🔄 PHASE 2: Modal serialization/deserialization...")
        
        # Simulate genome serialization (as done by ModalExecutor)
        serialized = {
            'id': genome.id,
            'seed': {
                'grid': genome.seed.grid.tolist(),
                'rule': genome.seed.rule,
                'steps': genome.seed.steps
            },
            'lora_config': {
                'r': genome.lora_cfg.r,
                'alpha': genome.lora_cfg.alpha,
                'dropout': genome.lora_cfg.dropout,
                'target_modules': list(genome.lora_cfg.target_modules),
                'adapter_type': getattr(genome.lora_cfg, 'adapter_type', 'lora')
            },
            'run_id': getattr(genome, 'run_id', None)
        }
        
        print(f"   📤 Serialized successfully")
        print(f"      • run_id: {serialized.get('run_id')}")
        print(f"      • adapter_type: {serialized['lora_config'].get('adapter_type')}")
        print(f"      • target_modules: {serialized['lora_config']['target_modules']}")
        
        # Simulate genome reconstruction (as done in evaluate_genome_modal)
        reconstructed_lora = LoRAConfig(
            r=serialized['lora_config']['r'],
            alpha=serialized['lora_config']['alpha'],
            dropout=serialized['lora_config']['dropout'],
            target_modules=tuple(serialized['lora_config']['target_modules']),
            adapter_type=serialized['lora_config'].get('adapter_type', 'lora')
        )
        
        reconstructed_genome = Genome(
            seed=CASeed(
                grid=np.array(serialized['seed']['grid']),
                rule=serialized['seed']['rule'],
                steps=serialized['seed']['steps']
            ),
            lora_cfg=reconstructed_lora,
            id=serialized['id'],
            run_id=serialized.get('run_id')
        )
        
        print(f"   📥 Reconstructed successfully")
        print(f"      • run_id: {reconstructed_genome.run_id}")
        print(f"      • adapter_type: {reconstructed_genome.lora_cfg.adapter_type}")
        
        # Phase 3: Calculate generation-time hash
        print(f"\n🎯 PHASE 3: Generation-time hash calculation...")
        
        generation_heavy_genes = HeavyGenes.from_lora_config(
            reconstructed_genome.lora_cfg,
            run_id=reconstructed_genome.run_id
        )
        generation_hash = generation_heavy_genes.to_hash()
        
        print(f"   📊 Generation hash: {generation_hash}")
        print(f"   📁 Lookup adapter path: /cache/adapters/adapter_{generation_hash}")
        
        # Phase 4: Hash consistency verification
        print(f"\n🔍 PHASE 4: Hash consistency verification...")
        
        if training_hash == generation_hash:
            print(f"   ✅ HASH CONSISTENCY VERIFIED!")
            print(f"      Both phases produce: {training_hash}")
            hash_consistent = True
        else:
            print(f"   ❌ HASH MISMATCH DETECTED!")
            print(f"      Training: {training_hash}")
            print(f"      Generation: {generation_hash}")
            print(f"      This will cause cache misses!")
            hash_consistent = False
        
        # Phase 5: Adapter existence check
        print(f"\n📂 PHASE 5: Adapter existence check...")
        
        adapters_dir = Path("/cache/adapters")
        if not adapters_dir.exists():
            print(f"   ❌ Adapters directory doesn't exist: {adapters_dir}")
            adapters_exist = False
        else:
            available_adapters = [d.name for d in adapters_dir.iterdir() if d.is_dir() and d.name.startswith('adapter_')]
            print(f"   📊 Found {len(available_adapters)} adapters")
            
            expected_adapter_name = f"adapter_{generation_hash}"
            if expected_adapter_name in available_adapters:
                print(f"   ✅ Expected adapter EXISTS: {expected_adapter_name}")
                adapters_exist = True
            else:
                print(f"   ❌ Expected adapter MISSING: {expected_adapter_name}")
                print(f"   📋 Available adapters (first 10):")
                for adapter in available_adapters[:10]:
                    print(f"      • {adapter}")
                if len(available_adapters) > 10:
                    print(f"      ... and {len(available_adapters) - 10} more")
                adapters_exist = False
        
        # Phase 6: Test result summary
        print(f"\n📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        success = hash_consistent and (adapters_exist or hash_consistent)
        
        if success:
            print(f"✅ CACHE COORDINATION TEST PASSED!")
            print(f"   • Hash consistency: ✅")
            print(f"   • Expected behavior: Training and generation use same hash")
        else:
            print(f"❌ CACHE COORDINATION TEST FAILED!")
            print(f"   • Hash consistency: {'✅' if hash_consistent else '❌'}")
            print(f"   • Adapter existence: {'✅' if adapters_exist else '❌'}")
            if not hash_consistent:
                print(f"   • Root cause: Serialization/deserialization changes hash")
            elif not adapters_exist:
                print(f"   • Root cause: Expected adapter not found in cache")
        
        return {
            "success": success,
            "hash_consistent": hash_consistent,
            "training_hash": training_hash,
            "generation_hash": generation_hash,
            "expected_adapter_exists": adapters_exist,
            "total_adapters": len(available_adapters) if 'available_adapters' in locals() else 0,
            "test_genome_id": genome.id,
            "run_id": genome.run_id,
            "adapter_type": genome.lora_cfg.adapter_type
        }
        
    except Exception as e:
        print(f"❌ Cache coordination test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    test_modal_functions() 