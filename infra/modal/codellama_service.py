"""
Modal service for CodeLlama generation.
Infrastructure layer - handles Modal-specific implementation details.

Side effects:
- Uses GPU memory globally (caches models)
- Performs aggressive CUDA memory management
- Modifies PyTorch environment variables
"""
import time as time_module
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from plugins.quixbugs_codellama.codellama_generation import GenerationRequest, GenerationResult, create_codellama_prompt, extract_function_from_generation
from infra.adapter_cache import is_modal_environment


# Enhanced cache: (model_name, dtype) -> (model, tokenizer)
_model_cache: Dict[Tuple[str, str], Tuple[object, object]] = {}
_last_volume_reload = 0

class GenerationError(Exception):
    """Domain-specific generation errors for granular handling."""
    pass

class AdapterError(Exception):
    """Adapter loading/validation errors."""
    pass

def clear_cuda(min_required_gb: float = 0) -> float:
    """Consolidated GPU memory cleanup with free memory reporting."""
    import torch
    import gc
    
    if not torch.cuda.is_available():
        return float('inf')  # No GPU constraints
    
    # Aggressive cleanup sequence
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    # Reset fragmentation stats if available
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()
    
    # Calculate free memory in GB
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    memory_free = memory_total - memory_allocated
    
    print(f"🧹 GPU cleanup: {memory_free:.2f} GB free (required: {min_required_gb:.1f} GB)")
    
    return memory_free

def get_cached_model(model_name: str, torch_dtype: str = "float16", cache_dir: str = "/cache/models"):
    """Get or load model with enhanced caching by (model_name, dtype)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    cache_key = (model_name, torch_dtype)
    
    if cache_key in _model_cache:
        print(f"⚡ Using cached model: {model_name} ({torch_dtype})")
        return _model_cache[cache_key]
    
    print(f"📥 Loading model: {model_name} ({torch_dtype}) [WARM CACHE]")
    
    # Pre-warm model cache if this is first load
    import os
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    
    # Load tokenizer with aggressive caching
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            use_fast=True  # Use fast tokenizer
        )
        print(f"⚡ Tokenizer loaded from local cache")
    except Exception:
        print(f"📦 Downloading tokenizer to warm cache...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            use_fast=True
        )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"🔧 Set padding token: {tokenizer.eos_token}")
    
    # Load model with optimized settings
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype_obj,
            device_map="auto",  # Better device mapping
            cache_dir=cache_dir,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2"  # Faster attention if available
        )
        print(f"⚡ Model loaded from local cache")
    except Exception as e:
        print(f"📦 Downloading model to warm cache (~13GB)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype_obj,
                device_map="auto",
                cache_dir=cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
        except Exception:
            # Fallback without flash attention
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype_obj,
                device_map="auto",
                cache_dir=cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
    
    # Cache the loaded model and tokenizer
    _model_cache[cache_key] = (model, tokenizer)
    print(f"⚡ Model cached for future use: {cache_key}")
    
    return model, tokenizer

def load_adapter_if_needed(base_model, adapter_path: Optional[str]):
    """Load adapter with simplified path validation and soft fallback."""
    if not adapter_path:
        print(f"📄 No adapter specified, using base model")
        return base_model
    
    adapter_path_obj = Path(adapter_path)
    
    # Simplified validation with soft fallback
    if not adapter_path_obj.exists() or not adapter_path_obj.is_dir():
        print(f"⚠️  Adapter {adapter_path} missing/invalid; using base model")
        return base_model
    
    try:
        from peft import PeftModel
        print(f"🚀 Loading adapter: {adapter_path_obj}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path_obj))
        print(f"✅ Adapter loaded successfully")
        return model
    except Exception as e:
        raise AdapterError(f"Adapter loading failed: {e}")

def smart_volume_reload():
    """Optimized Modal volume reload with 5-minute throttling."""
    global _last_volume_reload
    
    if not is_modal_environment():
        return
    
    current_time = time_module.time()
    if current_time - _last_volume_reload < 300:  # 5 minute throttle
        return
    
    try:
        import modal
        volume = modal.Volume.from_name("coral-x-clean-cache")
        volume.reload()
        _last_volume_reload = current_time
        print(f"✅ Modal volume reloaded (throttled)")
    except Exception as e:
        print(f"⚠️  Volume reload warning: {e}")

def generate_with_codellama_modal(request: GenerationRequest) -> GenerationResult:
    """
    Modal function for CodeLlama generation with optimized memory management.
    
    Raises:
        GenerationError: For generation-specific failures
        AdapterError: For adapter loading issues
    """
    import torch
    
    start_time = time_module.time()
    
    try:
        # 1. AGGRESSIVE MEMORY CLEANUP
        free_memory = clear_cuda(min_required_gb=4.0)
        
        if free_memory < 4.0:
            # Emergency cleanup: clear cache and retry
            global _model_cache
            _model_cache.clear()
            print(f"🚨 Emergency cache clear due to low memory")
            
            free_memory = clear_cuda(min_required_gb=4.0)
            if free_memory < 4.0:
                # Set expandable segments as last resort
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                free_memory = clear_cuda()
                
                if free_memory < 4.0:
                    raise GenerationError(
                        f"Insufficient GPU memory: {free_memory:.1f} GB free, need 4.0 GB"
                    )
        
        # 2. LOAD MODEL WITH ENHANCED CACHING
        model, tokenizer = get_cached_model(
            model_name=request.model_name,
            torch_dtype="float16",  # Could be parameterized later
            cache_dir="/cache/models"
        )
        
        # 3. SMART VOLUME RELOAD (THROTTLED)
        if request.adapter_path:
            smart_volume_reload()
        
        # 4. LOAD ADAPTER WITH SOFT FALLBACK
        model = load_adapter_if_needed(model, request.adapter_path)
        
        # 5. MEMORY CHECK BEFORE GENERATION
        pre_gen_memory = clear_cuda(min_required_gb=2.0)
        if pre_gen_memory < 2.0:
            print(f"⚠️  Low memory before generation: {pre_gen_memory:.1f} GB")
        
        # 6. CREATE PROMPT WITH VALIDATION
        if not isinstance(request.problem_name, str) or not isinstance(request.buggy_code, str):
            raise GenerationError(f"Invalid request parameters: problem_name and buggy_code must be strings")
        
        problem = {
            'name': request.problem_name,
            'buggy_code': request.buggy_code
        }
        
        try:
            prompt = create_codellama_prompt(problem)
        except Exception as e:
            raise GenerationError(f"Prompt creation failed: {e}")
        
        # 7. DYNAMIC PROMPT LENGTH CLAMPING
        max_prompt_length = getattr(model.config, 'max_position_embeddings', 2048) - request.max_tokens - 32
        max_prompt_length = max(512, min(max_prompt_length, 1024))  # Reasonable bounds
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_prompt_length
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 8. GENERATION WITH INFERENCE_MODE AND OOM RECOVERY
        try:
            with torch.inference_mode():  # More efficient than no_grad
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(request.max_tokens, 256),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=getattr(request, 'repetition_penalty', 1.1),
                    do_sample=getattr(request, 'do_sample', True),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False
                )
        except torch.cuda.OutOfMemoryError:
            print(f"🔧 OOM during generation, retrying with minimal settings...")
            clear_cuda()
            
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(request.max_tokens, 128),
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        do_sample=getattr(request, 'do_sample', True),
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False
                    )
                print(f"✅ Generation succeeded with emergency settings")
            except torch.cuda.OutOfMemoryError:
                raise GenerationError("GPU memory too fragmented for generation")
        
        # 9. DECODE AND EXTRACT
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        if generated_text.startswith(input_text):
            new_generation = generated_text[len(input_text):].strip()
        else:
            new_generation = generated_text
        
        if len(new_generation) < 10:
            raise GenerationError(f"Generated text too short: {len(new_generation)} chars")
        
        # 10. EXTRACT FUNCTION
        try:
            code_solution = extract_function_from_generation(new_generation, request.problem_name)
        except Exception as e:
            raise GenerationError(f"Function extraction failed: {e}")
        
        generation_time = time_module.time() - start_time
        
        # 11. FINAL CLEANUP
        del outputs, inputs
        if 'new_generation' in locals():
            del new_generation
        if 'generated_text' in locals():
            del generated_text
        clear_cuda()
        
        print(f"✅ Generation completed in {generation_time:.2f}s")
        
        return GenerationResult(
            generated_code=code_solution,
            function_name=request.problem_name,
            generation_time=generation_time
        )
        
    except (GenerationError, AdapterError):
        # Re-raise domain errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors
        generation_time = time_module.time() - start_time
        clear_cuda()  # Always cleanup on error
        raise GenerationError(f"Unexpected generation failure: {e}")

def create_generation_request(problem: Dict[str, Any], model_name: str, max_tokens: int = 512, 
                            temperature: float = 0.7, adapter_path: Optional[str] = None) -> GenerationRequest:
    """Helper to create GenerationRequest objects with consistent defaults."""
    return GenerationRequest(
        problem_name=problem.get('name', 'unknown'),
        buggy_code=problem.get('buggy_code', ''),
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        adapter_path=adapter_path,
        top_p=0.9,
        top_k=50
    )

def generate_baseline_solution_modal(problem: Dict[str, Any]) -> str:
    """Generate baseline solution using vanilla CodeLlama (no LoRA)."""
    request = create_generation_request(
        problem=problem,
        model_name='codellama/CodeLlama-7b-Python-hf',
        temperature=0.1,  # Lower temperature for baseline
        adapter_path=None
    )
    
    result = generate_with_codellama_modal(request)
    return result.generated_code

def generate_evolved_solution_modal(problem: Dict[str, Any], experiment_config: Dict[str, Any]) -> str:
    """Generate evolved solution using best LoRA from experiment."""
    request = create_generation_request(
        problem=problem,
        model_name='codellama/CodeLlama-7b-Python-hf',
        temperature=0.7,  # Higher temperature for evolved
        adapter_path=None  # TODO: Use best adapter from experiment
    )
    
    result = generate_with_codellama_modal(request)
    return result.generated_code 