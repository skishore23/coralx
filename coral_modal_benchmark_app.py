#!/usr/bin/env python3
"""
CORAL-X Benchmark Modal App
==========================

Separate Modal app for realtime benchmark testing with neutral parameters.
This app is isolated from evolution to prevent parameter conflicts.

Architecture:
- App Name: coral-x-benchmarks
- Parameters: Fixed neutral (temp=0.7, top_p=0.9, top_k=50)
- Purpose: Fair adapter comparison, realtime monitoring
- GPU Pool: Separate from evolution
"""

import modal
import time
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Benchmark Modal App (separate from evolution)
benchmark_app = modal.App("coral-x-benchmarks")

# Shared cache volume (same as evolution)
cache_volume = modal.Volume.from_name("coral-cache", create_if_missing=True)

# Benchmark-specific GPU configuration
BENCHMARK_GPU = modal.gpu.A100(size="40GB", count=1)
BENCHMARK_IMAGE = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.0.0",
    "transformers>=4.30.0", 
    "peft>=0.4.0",
    "safetensors>=0.3.0",
    "accelerate>=0.20.0",
    "bitsandbytes>=0.39.0"
])

@dataclass
class BenchmarkRequest:
    """Benchmark generation request with neutral parameters."""
    adapter_path: str
    problem_name: str
    buggy_code: str
    # NEUTRAL PARAMETERS ONLY
    temperature: float = 0.7    # Fixed neutral
    top_p: float = 0.9         # Fixed neutral  
    top_k: int = 50            # Fixed neutral
    max_tokens: int = 512      # Fixed neutral

@benchmark_app.function(
    gpu=BENCHMARK_GPU,
    image=BENCHMARK_IMAGE,
    volumes={"/cache": cache_volume},
    timeout=300,
    memory=16384
)
def benchmark_generate_modal(request: BenchmarkRequest) -> Dict[str, Any]:
    """
    Benchmark code generation with NEUTRAL parameters only.
    
    This function is completely isolated from evolution parameters,
    ensuring fair adapter comparison.
    """
    print(f"🔧 BENCHMARK GENERATION START")
    print(f"   • App: coral-x-benchmarks (isolated)")
    print(f"   • Mode: NEUTRAL PARAMETERS ONLY")
    print(f"   • Problem: {request.problem_name}")
    print(f"   • Adapter: {request.adapter_path}")
    
    print(f"🎯 NEUTRAL PARAMETERS (no CA influence):")
    print(f"   • Temperature: {request.temperature} (fixed neutral)")
    print(f"   • Top-p: {request.top_p} (fixed neutral)")
    print(f"   • Top-k: {request.top_k} (fixed neutral)")
    print(f"   • Max tokens: {request.max_tokens} (fixed neutral)")
    
    try:
        # Import generation logic
        from coral.domain.codellama_generation import create_bugfix_prompt
        
        # Load model and adapter
        print(f"🔧 Loading CodeLlama with neutral parameters...")
        
        # Load base model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "codellama/CodeLlama-7b-Python-hf"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/cache/models"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="/cache/models"
        )
        
        # Load adapter with PEFT
        from peft import PeftModel
        print(f"🔗 Loading adapter: {request.adapter_path}")
        model = PeftModel.from_pretrained(model, request.adapter_path)
        
        # Create prompt
        problem_dict = {
            "name": request.problem_name,
            "buggy_code": request.buggy_code
        }
        prompt = create_bugfix_prompt(problem_dict)
        
        # Generate with NEUTRAL parameters only
        print(f"🎯 Generating with neutral parameters...")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,  # Fixed 0.7
                top_p=request.top_p,             # Fixed 0.9
                top_k=request.top_k,             # Fixed 50
                do_sample=True,                  # Always sample for benchmarks
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode result
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract new generation
        if generated_text.startswith(prompt):
            result = generated_text[len(prompt):].strip()
        else:
            result = generated_text.strip()
        
        print(f"✅ Benchmark generation completed: {len(result)} chars")
        
        return {
            "generated_code": result,
            "parameters_used": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "mode": "neutral_benchmark"
            },
            "adapter_path": request.adapter_path,
            "problem_name": request.problem_name
        }
        
    except Exception as e:
        print(f"❌ Benchmark generation failed: {e}")
        raise RuntimeError(f"Benchmark generation failed: {e}")

@benchmark_app.function(
    image=BENCHMARK_IMAGE,
    volumes={"/cache": cache_volume},
    timeout=600
)
def run_realtime_benchmark_modal(
    adapter_paths: List[str],
    test_problems: List[Dict[str, Any]],
    benchmark_interval: int = 300  # 5 minutes
) -> Dict[str, Any]:
    """
    Run continuous realtime benchmarks on evolved adapters.
    
    This function monitors adapter performance with neutral parameters,
    providing unbiased comparison data.
    """
    print(f"🚀 REALTIME BENCHMARK SYSTEM ACTIVE")
    print(f"   • Mode: Continuous monitoring")
    print(f"   • Adapters: {len(adapter_paths)}")
    print(f"   • Test problems: {len(test_problems)}")
    print(f"   • Interval: {benchmark_interval}s")
    
    results = []
    
    for adapter_path in adapter_paths:
        print(f"\n📊 Benchmarking adapter: {adapter_path}")
        
        adapter_results = []
        
        for problem in test_problems:
            request = BenchmarkRequest(
                adapter_path=adapter_path,
                problem_name=problem["name"],
                buggy_code=problem["buggy_code"]
                # Neutral parameters set by default
            )
            
            try:
                result = benchmark_generate_modal.remote(request)
                adapter_results.append(result)
                print(f"   ✅ {problem['name']}: {len(result['generated_code'])} chars")
                
            except Exception as e:
                print(f"   ❌ {problem['name']}: {e}")
                adapter_results.append({"error": str(e)})
        
        results.append({
            "adapter_path": adapter_path,
            "results": adapter_results,
            "timestamp": time.time()
        })
    
    return {
        "benchmark_results": results,
        "parameters_used": "neutral_only",
        "benchmark_type": "realtime_monitoring"
    }

if __name__ == "__main__":
    print("🚀 CORAL-X Benchmark Modal App")
    print("   • Isolated from evolution")
    print("   • Neutral parameters only")
    print("   • Fair adapter comparison") 