"""
Modal service for CodeLlama generation.
Infrastructure layer - handles Modal-specific implementation details.
"""
import time as time_module
from typing import Dict, Any
from coral.domain.codellama_generation import GenerationRequest, GenerationResult, create_codellama_prompt, extract_function_from_generation
from infra.adapter_cache import is_modal_environment


# Global model cache to prevent reloading
_cached_model = None
_cached_tokenizer = None
_cached_model_name = None

def _cleanup_gpu_memory():
    """Clean up GPU memory and garbage collect."""
    import torch
    import gc
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🧹 GPU memory cleanup completed")

def generate_with_codellama_modal(request: GenerationRequest) -> GenerationResult:
    """
    Modal function for CodeLlama generation.
    Uses real transformers and PEFT for LoRA adaptation with smart memory management.
    """
    import torch
    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    global _cached_model, _cached_tokenizer, _cached_model_name
    
    # Use model cache to prevent reloading
    from pathlib import Path
    model_cache_dir = "/cache/models"
    
    # SMART DEPENDENCY MANAGEMENT: Create cache directory if missing
    cache_path = Path(model_cache_dir)
    if not cache_path.exists():
        print(f"🔧 Creating model cache directory: {model_cache_dir}")
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Model cache directory created")
    else:
        print(f"🔍 Using existing model cache: {model_cache_dir}")
    
    start_time = time_module.time()
    
    try:
        # SMART MEMORY MANAGEMENT: Clean GPU memory before any operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cleared before generation")
        
        # OPTIMIZED CACHING: Only cache base model, reload adapters as needed
        # This prevents memory explosion from multiple adapter+model combinations
        need_base_reload = (_cached_model is None or 
                           _cached_tokenizer is None or 
                           not _cached_model_name or
                           not _cached_model_name.startswith(request.model_name))
        
        # FAIL-FAST: Check available GPU memory before proceeding
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3      # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            memory_free = memory_total - memory_allocated
            
            print(f"🔍 GPU Memory Status:")
            print(f"   • Total: {memory_total:.2f} GB")
            print(f"   • Allocated: {memory_allocated:.2f} GB")
            print(f"   • Cached: {memory_cached:.2f} GB") 
            print(f"   • Free: {memory_free:.2f} GB")
            
            # FAIL-FAST: Require at least 8GB free for CodeLlama-7B + adapter
            min_required_gb = 8.0
            if memory_free < min_required_gb:
                # Force aggressive cleanup
                print(f"⚠️  Low memory ({memory_free:.2f} GB), forcing cleanup...")
                if _cached_model is not None:
                    del _cached_model
                    _cached_model = None
                if _cached_tokenizer is not None:
                    del _cached_tokenizer  
                    _cached_tokenizer = None
                _cached_model_name = None
                _cleanup_gpu_memory()
                
                # Re-check after cleanup
                memory_allocated_after = torch.cuda.memory_allocated() / 1024**3
                memory_free_after = memory_total - memory_allocated_after
                print(f"🧹 After cleanup - Free: {memory_free_after:.2f} GB")
                
                if memory_free_after < min_required_gb:
                    raise RuntimeError(
                        f"FAIL-FAST: Insufficient GPU memory for CodeLlama-7B + adapter.\n"
                        f"Required: {min_required_gb:.1f} GB, Available: {memory_free_after:.2f} GB\n"
                        f"Total GPU memory: {memory_total:.2f} GB\n"
                        f"Try using a smaller model or increasing GPU allocation."
                    )
                need_base_reload = True  # Force reload after cleanup
        
        if need_base_reload:
            print(f"📥 Loading base model: {request.model_name}")
            if request.adapter_path:
                print(f"🔗 Adapter will be loaded dynamically: {request.adapter_path}")
            
            # SMART MODEL LOADING: Auto-download if not cached
            try:
                # Try loading with local_files_only first (fast)
                tokenizer = AutoTokenizer.from_pretrained(
                    request.model_name,
                    cache_dir=model_cache_dir,
                    local_files_only=True  # Use cached model
                )
                print(f"✅ Tokenizer loaded from cache")
            except Exception as cache_error:
                print(f"🔄 Tokenizer not in cache, downloading: {cache_error}")
                # Auto-download if not cached (instead of failing)
                tokenizer = AutoTokenizer.from_pretrained(
                    request.model_name,
                    cache_dir=model_cache_dir,
                    local_files_only=False  # Download if needed
                )
                print(f"✅ Tokenizer downloaded and cached")
            
            # Set padding token for CodeLlama
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"🔧 Set padding token to EOS token: {tokenizer.eos_token}")
            
            # SMART MODEL LOADING: Auto-download if not cached
            try:
                # Try loading with local_files_only first (fast)
                base_model = AutoModelForCausalLM.from_pretrained(
                    request.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=model_cache_dir,
                    local_files_only=True,
                    low_cpu_mem_usage=True
                )
                print(f"✅ Base model loaded from cache")
            except Exception as cache_error:
                print(f"🔄 Model not in cache, downloading (~13GB): {cache_error}")
                # Auto-download if not cached (instead of failing)
                base_model = AutoModelForCausalLM.from_pretrained(
                    request.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=model_cache_dir,
                    local_files_only=False,  # Download if needed
                    low_cpu_mem_usage=True
                )
                print(f"✅ Model downloaded and cached ({request.model_name})")
            
            # Cache only the base model and tokenizer (not adapter combinations)
            # Always cache the full model (not extracted base) to preserve generation methods
            _cached_model = base_model  # Cache the full model with all methods
            _cached_tokenizer = tokenizer
            _cached_model_name = request.model_name  # Cache model name only
            
            print(f"✅ Base model loaded and cached: {request.model_name}")
        else:
            print(f"✅ Using cached base model: {_cached_model_name}")
            tokenizer = _cached_tokenizer
            base_model = _cached_model
        
        # DYNAMIC ADAPTER LOADING: Load adapter on top of base model (not cached)
        if request.adapter_path:
            print(f"🔗 Loading adapter dynamically: {request.adapter_path}")
            
            # ✅ PROACTIVE VOLUME RELOAD - Ensure we see latest artifacts from training containers
            if is_modal_environment():
                try:
                    import modal
                    volume = modal.Volume.from_name("coral-x-clean-cache")
                    volume.reload()
                    print(f"✅ Modal volume reloaded - synced with training containers")
                except Exception as reload_error:
                    print(f"⚠️  Volume reload warning: {reload_error}")
            
            # SIMPLIFIED: Just reload base model fresh (eliminates complex cleanup)
            # This trades slightly more memory for MUCH faster loading (8s -> 1s)
            print(f"🔄 Reloading fresh base model to avoid PEFT conflicts...")
            
            fresh_model = AutoModelForCausalLM.from_pretrained(
                request.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=model_cache_dir,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            base_model = fresh_model  # Use fresh model for adapter loading
            
            # Load new adapter on top of prepared base model
            adapter_path = Path(request.adapter_path)
            
            print(f"🔍 Adapter loading diagnostics:")
            print(f"   • Requested path: {request.adapter_path}")
            print(f"   • Path exists: {adapter_path.exists()}")
            print(f"   • Path is directory: {adapter_path.is_dir() if adapter_path.exists() else 'N/A'}")
            
            # Check parent directory and list contents for debugging
            parent_dir = adapter_path.parent
            if parent_dir.exists():
                print(f"   • Parent directory: {parent_dir}")
                try:
                    adapter_dirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith('adapter_')]
                    print(f"   • Available adapters in {parent_dir}:")
                    for adapter_dir in adapter_dirs[:5]:  # Show first 5
                        print(f"     - {adapter_dir.name}")
                    if len(adapter_dirs) > 5:
                        print(f"     - ... and {len(adapter_dirs) - 5} more")
                except Exception as e:
                    print(f"   • Error listing parent directory: {e}")
            else:
                print(f"   • Parent directory does not exist: {parent_dir}")
            
            # FAIL-FAST: Adapter path must exist and be a directory
            if not adapter_path.exists():
                # CRITICAL: Handle Modal volume sync race condition
                if is_modal_environment():
                    print(f"⚠️  Adapter not found immediately - attempting Modal volume sync...")
                    
                    # 🔥 SOLUTION: Force Modal volume reload to see changes from other containers
                    try:
                        print(f"🔄 Forcing Modal volume reload to sync latest state...")
                        # Force volume reload using Modal API
                        # This ensures we see files committed by other containers (training functions)
                        import modal
                        volume = modal.Volume.from_name("coral-x-clean-cache")  # Match actual volume name
                        volume.reload()
                        print(f"✅ Volume reload completed - checking for adapter...")
                        
                        # Give filesystem a moment to reflect changes
                        import time
                        time.sleep(2)
                        
                        # Check again after reload
                        if adapter_path.exists():
                            print(f"✅ Adapter found after volume reload!")
                        else:
                            print(f"❌ Adapter still not found after volume reload")
                            
                    except Exception as reload_error:
                        print(f"⚠️  Volume reload failed: {reload_error}")
                    
                    # Try up to 3 times with increasing delays
                    for attempt in range(3):
                        print(f"   🔄 Attempt {attempt + 1}/3: Checking adapter availability...")
                        
                        # Force volume sync
                        try:
                            import os
                            os.sync()
                            time_module.sleep(1 + attempt)  # 1s, 2s, 3s delays
                            print(f"   💾 Volume sync completed (attempt {attempt + 1})")
                        except Exception as sync_error:
                            print(f"   ⚠️  Volume sync warning: {sync_error}")
                        
                        # Check if adapter is now available
                        if adapter_path.exists():
                            print(f"   ✅ Adapter found after sync (attempt {attempt + 1})")
                            break
                        else:
                            print(f"   ❌ Adapter still not found (attempt {attempt + 1})")
                            if attempt < 2:
                                print(f"   ⏳ Waiting before next attempt...")
                    
                    # Final check after all attempts
                    if not adapter_path.exists():
                        print(f"❌ CRITICAL: Adapter still not found after {3} sync attempts")
                        # List available adapters for debugging
                        try:
                            available_adapters = [d.name for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith('adapter_')]
                            print(f"   📁 Available adapters: {available_adapters}")
                        except Exception as list_error:
                            print(f"   ❌ Error listing adapters: {list_error}")
                
                # Still not found - fail with comprehensive error
                raise RuntimeError(
                    f"FAIL-FAST: LoRA adapter path does not exist: {request.adapter_path}\n"
                    f"Expected absolute path to trained adapter directory.\n"
                    f"Parent directory exists: {parent_dir.exists()}\n"
                    f"Available adapters: {[d.name for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith('adapter_')] if parent_dir.exists() else 'None'}\n"
                    f"This suggests a Modal volume sync race condition.\n"
                    f"No fallback paths will be tried - fix the path coordination."
                )
            
            if not adapter_path.is_dir():
                raise RuntimeError(
                    f"FAIL-FAST: LoRA adapter path is not a directory: {request.adapter_path}\n"
                    f"Path type: {'file' if adapter_path.is_file() else 'other'}\n"
                    f"LoRA adapters must be saved as directories, not files.\n"
                    f"Check your adapter training implementation."
                )
            
            # SIMPLIFIED: Minimal verification (PEFT will fail fast if files missing)
            print(f"🔗 Loading adapter from: {adapter_path}")
            
            # SIMPLIFIED: Let PEFT handle file verification (fail-fast if missing)
            try:
                from peft import PeftModel
                print(f"🚀 Loading adapter: {adapter_path}")
                model = PeftModel.from_pretrained(base_model, str(adapter_path))
                print(f"✅ Adapter loaded successfully")
            except Exception as e:
                raise RuntimeError(f"FAIL-FAST: Adapter loading failed: {e}")
        else:
            # No adapter path provided, use base model
            model = base_model
            print(f"📄 Using base model (no adapter specified)")
        
        # Create prompt using domain logic
        # FAIL-FAST: Validate inputs before creating problem dict
        if not isinstance(request.problem_name, str):
            raise RuntimeError(f"FAIL-FAST: problem_name must be string, got {type(request.problem_name)}: {request.problem_name}")
        
        if not isinstance(request.buggy_code, str):
            raise RuntimeError(f"FAIL-FAST: buggy_code must be string, got {type(request.buggy_code)}: {request.buggy_code}")
        
        problem = {
            'name': request.problem_name,
            'buggy_code': request.buggy_code
        }
        
        print(f"🔧 DEBUG: Creating prompt with problem:")
        print(f"   • problem: {problem}")
        print(f"   • problem_name: '{request.problem_name}' (type: {type(request.problem_name)})")
        print(f"   • buggy_code length: {len(request.buggy_code)} chars (type: {type(request.buggy_code)})")
        
        try:
            prompt = create_codellama_prompt(problem)
            print(f"✅ Prompt created successfully")
        except Exception as prompt_error:
            print(f"❌ PROMPT CREATION FAILED:")
            print(f"   • Error type: {type(prompt_error)}")
            print(f"   • Error message: {prompt_error}")
            print(f"   • Problem dict keys: {list(problem.keys())}")
            print(f"   • Problem dict values types: {[type(v) for v in problem.values()]}")
            print(f"   • Raw problem_name: {repr(request.problem_name)}")
            print(f"   • Raw buggy_code: {repr(request.buggy_code[:100])}")
            raise RuntimeError(f"FAIL-FAST: Prompt creation failed: {prompt_error}")
        
        print(f"📝 Generated prompt for: {request.problem_name}")
        print(f"🎛️  Generation params: max_tokens={request.max_tokens}, temp={request.temperature}, top_p={request.top_p}, top_k={request.top_k}")
        print(f"🎛️  FULL CHEAP KNOBS:")
        print(f"      • Temperature: {request.temperature:.3f} (complexity-driven)")
        print(f"      • Top-p: {request.top_p:.3f} (intensity-driven)")
        print(f"      • Top-k: {request.top_k} (convergence-driven)")
        print(f"      • Repetition penalty: {request.repetition_penalty:.3f} (periodicity-driven)")
        print(f"      • Max tokens: {request.max_tokens} (feature-derived)")
        print(f"      • Sampling: {request.do_sample} (CA-controlled)")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with specified parameters INCLUDING CHEAP KNOBS
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,  # NEW: Cheap knob
                do_sample=request.do_sample,                    # NEW: Cheap knob
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # CRITICAL DEBUG: Check if generation actually happened
        input_length = len(inputs['input_ids'][0])
        output_length = len(outputs[0])
        new_tokens = output_length - input_length
        
        print(f"🔍 GENERATION DEBUG:")
        print(f"   • Input tokens: {input_length}")
        print(f"   • Output tokens: {output_length}")
        print(f"   • New tokens generated: {new_tokens}")
        
        # Extract only the newly generated part
        if new_tokens <= 0:
            print(f"❌ CRITICAL: CodeLlama generated {new_tokens} new tokens!")
            print(f"   • This explains why extraction is failing")
            print(f"   • Input prompt length: {len(prompt)} chars")
            print(f"   • Generated text length: {len(generated_text)} chars")
            print(f"   • Generated text preview: {repr(generated_text[:200])}")
            raise RuntimeError(f"CodeLlama generated {new_tokens} new tokens. Check generation parameters.")
        
        # Decode only the new tokens
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        if generated_text.startswith(input_text):
            # Extract only the new part
            new_generation = generated_text[len(input_text):].strip()
            print(f"✅ Extracted new generation: {len(new_generation)} chars")
            print(f"📝 New generation preview: {repr(new_generation[:200])}")
        else:
            # Fallback: use full text but warn
            new_generation = generated_text
            print(f"⚠️  Could not isolate new generation, using full text")
        
        if not new_generation or len(new_generation) < 10:
            print(f"❌ CRITICAL: New generation too short: {len(new_generation)} chars")
            print(f"   • New generation: {repr(new_generation)}")
            raise RuntimeError(f"CodeLlama generated insufficient new content: {len(new_generation)} chars")
        
        # Extract function from generated response using domain logic
        print(f"📝 Raw generated text (first 200 chars): {generated_text[:200]}...")
        print(f"🔧 Attempting function extraction for: '{request.problem_name}'")
        
        try:
            code_solution = extract_function_from_generation(new_generation, request.problem_name)
            print(f"✅ EXTRACTION SUCCESS: Found function ({len(code_solution)} chars)")
            print(f"📋 Extracted code preview: {code_solution[:150]}...")
        except Exception as extraction_error:
            print(f"❌ EXTRACTION FAILED: {extraction_error}")
            print(f"🔍 New generation for debugging:")
            print("─" * 50)
            for i, line in enumerate(new_generation.split('\n')):
                print(f"{i:3d}: {line}")
            print("─" * 50)
            
            # Try to provide a helpful fallback or re-raise with more context
            raise RuntimeError(f"Function extraction failed for '{request.problem_name}': {extraction_error}")
        
        generation_time = time_module.time() - start_time
        
        print(f"✅ Generation completed in {generation_time:.2f}s")
        print(f"📝 Generated tokens: {len(outputs[0]) - len(inputs['input_ids'][0])}")
        print(f"🎯 Final result preview: {code_solution[:100]}...")
        
        # MEMORY CLEANUP: Free adapter memory if loaded (but keep base model cached)
        if request.adapter_path and hasattr(model, 'unload'):
            try:
                # Some PEFT models have unload method
                model.unload()
                print(f"🧹 Adapter unloaded from memory")
            except:
                pass
        
        # Clear intermediate tensors and free GPU memory
        if torch.cuda.is_available():
            del outputs, inputs
            if 'new_generation' in locals():
                del new_generation
            if 'generated_text' in locals():
                del generated_text
            _cleanup_gpu_memory()
        
        return GenerationResult(
            generated_code=code_solution,
            function_name=request.problem_name,
            generation_time=generation_time
        )
        
    except Exception as e:
        generation_time = time_module.time() - start_time
        print(f"❌ CodeLlama generation failed: {e}")
        
        # MEMORY CLEANUP: Ensure GPU memory is cleared even on error
        _cleanup_gpu_memory()
        
        raise RuntimeError(f"Real CodeLlama generation failed: {e}")


def generate_baseline_solution_modal(problem: Dict[str, Any]) -> str:
    """Generate baseline solution using vanilla CodeLlama (no LoRA)."""
    request = GenerationRequest(
        problem_name=problem.get('name', 'unknown'),
        buggy_code=problem.get('buggy_code', ''),
        model_name='codellama/CodeLlama-7b-Python-hf',
        max_tokens=512,
        temperature=0.1,  # Lower temperature for baseline
        adapter_path=None,  # No LoRA for baseline
        top_p=0.9,
        top_k=50
    )
    
    result = generate_with_codellama_modal(request)
    return result.generated_code


def generate_evolved_solution_modal(problem: Dict[str, Any], experiment_config: Dict[str, Any]) -> str:
    """Generate evolved solution using best LoRA from experiment."""
    # For now, use same generation as baseline but with different temperature
    # In real implementation, this would use the best LoRA adapter from evolution
    request = GenerationRequest(
        problem_name=problem.get('name', 'unknown'),
        buggy_code=problem.get('buggy_code', ''),
        model_name='codellama/CodeLlama-7b-Python-hf',
        max_tokens=512,
        temperature=0.7,  # Higher temperature for evolved
        adapter_path=None,  # TODO: Use best adapter from experiment
        top_p=0.9,
        top_k=50
    )
    
    result = generate_with_codellama_modal(request)
    return result.generated_code 