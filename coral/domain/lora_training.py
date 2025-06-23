"""
LoRA Training Domain Logic - Pure Functions for PEFT Training
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING
import time
import torch
from pathlib import Path

if TYPE_CHECKING:
    from infra.adapter_cache import HeavyGenes

# 🔥 REMOVED: Global DORA_AVAILABLE flag - now config-driven!
# No more global state that creates local/Modal environment mismatches


@dataclass(frozen=True)
class AdapterConfig:
    """Immutable adapter configuration supporting both LoRA and DoRA."""
    r: int               # 🔥 FIXED: Use PEFT convention (r not rank)
    alpha: float
    dropout: float
    target_modules: list
    task_type: str = "CAUSAL_LM"
    adapter_type: str = "lora"  # "lora" or "dora"


# Backward compatibility alias
LoRAConfig = AdapterConfig


@dataclass(frozen=True)
class LoRATrainingResult:
    """Immutable result of LoRA training."""
    adapter_path: str
    training_time: float
    training_loss: float
    model_size_mb: float
    success: bool
    error_message: Optional[str] = None


def create_lora_config_from_features(features: Dict[str, float]) -> AdapterConfig:
    """Pure function to derive LoRA config from CA features."""
    # Map CA features to LoRA parameters using mathematical functions
    complexity = features.get('complexity', 0.5)
    intensity = features.get('intensity', 0.5)
    periodicity = features.get('periodicity', 0.5)
    convergence = features.get('convergence', 0.5)
    
    # Rank: Higher complexity → higher rank (more parameters)
    rank = max(4, min(32, int(4 + complexity * 28)))
    
    # Alpha: Balance between complexity and intensity  
    alpha = 16.0 + (complexity + intensity) * 24.0
    
    # Dropout: Higher convergence → lower dropout (more stable)
    dropout = max(0.05, min(0.3, 0.3 - convergence * 0.25))
    
    return AdapterConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        adapter_type="lora"  # Default to LoRA for this function
    )


def train_codellama_lora(base_ckpt: str, heavy_genes: 'HeavyGenes', save_to: str) -> str:  # 🔥 FIX: Expect specific type
    """
    Train CodeLlama LoRA adapter with given heavy genes.
    Function name from coral-x-codellama.md specification.
    
    Args:
        base_ckpt: Base model checkpoint (e.g., "codellama/CodeLlama-7b-Python-hf")
        heavy_genes: HeavyGenes object (use HeavyGenes.from_lora_config() to create)
        save_to: Path to save the trained adapter
    
    Returns:
        str: Path to the saved adapter
    """
    # Validate input type for clarity
    from infra.adapter_cache import HeavyGenes
    if not isinstance(heavy_genes, HeavyGenes):
        raise TypeError(
            f"FAIL-FAST: Expected HeavyGenes object, got {type(heavy_genes)}. "
            f"Use HeavyGenes.from_lora_config() to create from LoRAConfig."
        )
    
    # Extract values directly from HeavyGenes dataclass - clean and type-safe
    rank = heavy_genes.rank
    alpha = heavy_genes.alpha
    dropout = heavy_genes.dropout
    target_modules = heavy_genes.target_modules
    adapter_type = heavy_genes.adapter_type
    run_id = heavy_genes.run_id
    
    print(f"🚀 TRAIN_CODELLAMA_{adapter_type.upper()}: Starting {adapter_type.upper()} training")
    print(f"   Base checkpoint: {base_ckpt}")
    print(f"   Heavy genes: {heavy_genes}")
    print(f"   Save path: {save_to}")
    
    print(f"🔧 Adapter configuration: type={adapter_type}, r={rank}, α={alpha}, dropout={dropout}")
    
    # Create adapter configuration (supports both LoRA and DoRA)
    lora_config = AdapterConfig(
        r=int(rank),           # 🔥 FIXED: Use PEFT convention (r)
        alpha=float(alpha),
        dropout=float(dropout),
        target_modules=list(target_modules),
        adapter_type=adapter_type  # 🔥 FIX: Use adapter_type from heavy_key
    )
    
    # Prepare training data (load from QuixBugs if available)
    training_data = []
    try:
        # Try to load QuixBugs training data using auto-detection
        from adapters.quixbugs_real import QuixBugsRealAdapter
        
        print(f"📁 Loading QuixBugs training data...")
        adapter = QuixBugsRealAdapter()  # Use auto-detection
        problems = list(adapter.problems())
        
        # 🔥 CRITICAL: Use centralized training problems list for DoRA/LoRA training
        # This MUST match what's excluded from evaluation to prevent data leakage
        from .dataset_constants import QUIXBUGS_TRAINING_PROBLEMS
        TRAINING_PROBLEMS = QUIXBUGS_TRAINING_PROBLEMS
        
        training_count = 0
        for problem in problems:
            problem_name = problem.get('name', 'unknown')
            
            # Only use designated training problems
            if problem_name not in TRAINING_PROBLEMS:
                continue
                
            buggy_code = problem.get('buggy_code', '')
            correct_code = problem.get('correct_code', '')
            
            if buggy_code and correct_code and buggy_code != correct_code:
                # ENHANCEMENT: Data augmentation - create multiple training examples per problem
                base_examples = [
                    {
                        "instruction": f"Fix the bug in this Python function from {problem_name}:",
                        "input": buggy_code.strip(),
                        "output": correct_code.strip()
                    },
                    {
                        "instruction": f"Debug and correct the following Python code for {problem_name}:",
                        "input": buggy_code.strip(),
                        "output": correct_code.strip()
                    },
                    {
                        "instruction": f"The following function has a bug. Provide the corrected version:",
                        "input": buggy_code.strip(),
                        "output": correct_code.strip()
                    },
                    {
                        "instruction": f"Analyze and fix the error in this {problem_name} implementation:",
                        "input": buggy_code.strip(),
                        "output": correct_code.strip()
                    }
                ]
                
                training_data.extend(base_examples)
                training_count += len(base_examples)
        
        print(f"✅ Loaded {len(training_data)} QuixBugs training examples")
        print(f"📊 DORA/LORA TRAIN/TEST SPLIT VALIDATION:")
        print(f"   • Training problems: {len(TRAINING_PROBLEMS)} (CENTRALIZED - prevents data leakage)")
        print(f"   • Training examples: {training_count}")
        print(f"   • Total QuixBugs problems: {len(problems)}")
        print(f"   • Test problems available: {len(problems) - len(TRAINING_PROBLEMS)}")
        print(f"   • 🛡️  ANTI-CONTAMINATION: DoRA/LoRA adapters will NEVER see test problems")
        
        if training_count != len(TRAINING_PROBLEMS):
            print(f"⚠️  WARNING: Expected {len(TRAINING_PROBLEMS)} training problems, got {training_count}")
            
    except Exception as e:
        raise RuntimeError(
            f"FAIL-FAST: QuixBugs training data not available: {e}\n"
            f"Cannot train LoRA adapter without real training data.\n"
            f"Run: python setup_quixbugs_dataset.py setup"
        )
    
    # SIMPLIFIED: Train adapter directly in Modal volume (no move needed)
    target_path = Path(save_to)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Train adapter directly - function will raise exception on failure
    adapter_path = train_lora_adapter_direct(
        base_model_name=base_ckpt,
        lora_config=lora_config,
        training_data=training_data,
        save_directly_to=save_to  # Train directly to Modal volume
    )
    
    # Verify adapter exists at expected location
    if not target_path.exists():
        raise RuntimeError(f"CRITICAL: Adapter not found after training: {save_to}")
    
    print(f"✅ TRAIN_CODELLAMA_{adapter_type.upper()} completed: {save_to}")
    # Return the path string as expected by adapter cache
    return save_to


def train_lora_adapter_direct(
    base_model_name: str,
    lora_config: AdapterConfig,
    training_data: list,
    save_directly_to: str
) -> str:
    """
    Train LoRA adapter and save directly to target path (simplified for Modal volume).
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import os
        
        start_time = time.time()
        target_path = Path(save_directly_to)
        
        print(f"🚀 Training {lora_config.adapter_type.upper()} adapter directly to: {target_path}")
        print(f"   r={lora_config.r}, α={lora_config.alpha}, dropout={lora_config.dropout}")
        
        # Use Modal volume for model cache
        model_cache_dir = "/cache/models" if Path("/cache").exists() else "./cache/models"
        
        print(f"📥 Loading base model: {base_model_name}")
        print(f"🗂️  Using model cache: {model_cache_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=model_cache_dir,
            local_files_only=True  # Use cached model
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 Set padding token to EOS token: {tokenizer.eos_token}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=model_cache_dir,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        # 🔥 CONFIG-DRIVEN: Configure adapter based on config, fail-fast if unavailable
        if lora_config.adapter_type == "dora":
            # Check DoRA support at runtime - FAIL-FAST if not available
            if not _check_dora_support_runtime():
                raise RuntimeError(
                    f"FAIL-FAST: DoRA requested (adapter_type='dora') but not available.\n"
                    f"Current peft version does not support use_dora parameter.\n"
                    f"Install peft>=0.10 to use DoRA adapters.\n"
                    f"NO FALLBACKS - fix your environment or change config to adapter_type='lora'"
                )
            
            print(f"🚀 Using DoRA adapter: r={lora_config.r}, α={lora_config.alpha}")
            peft_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                target_modules=lora_config.target_modules,
                use_dora=True  # 🔥 Config says DoRA, environment supports it → use DoRA
            )
        else:
            # Use LoRA (default)
            print(f"🚀 Using LoRA adapter: r={lora_config.r}, α={lora_config.alpha}")
            peft_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                target_modules=lora_config.target_modules,
                use_dora=False  # Config says LoRA → use LoRA
            )
        
        model = get_peft_model(model, peft_config)
        
        # Prepare training data
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,
                max_length=512,
                return_tensors=None
            )
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
            return tokenized
        
        formatted_data = []
        for item in training_data:
            text = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Output: {item['output']}{tokenizer.eos_token}"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        columns_to_remove = [col for col in tokenized_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
        if columns_to_remove:
            tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            return_tensors="pt"
        )
        
        # Disable wandb completely to avoid API key issues
        import os
        import shutil
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        
        # CRITICAL FIX: Aggressive cleanup of target directory and any leftover temp dirs
        print(f"🧹 AGGRESSIVE CLEANUP: Cleaning target and temp directories")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove target directory completely if exists
        if target_path.exists():
            shutil.rmtree(target_path)
            print(f"   🗑️  Removed existing target: {target_path}")
        
        # Clean up ANY leftover training_tmp directories in parent directory
        parent_dir = target_path.parent
        for item in parent_dir.glob("training_tmp*"):
            if item.is_dir():
                shutil.rmtree(item)
                print(f"   🗑️  Removed leftover temp dir: {item}")
        
        # Create unique temp directory with process ID to avoid collisions
        import uuid
        temp_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        temp_training_dir = parent_dir / f"training_tmp_{temp_suffix}"
        temp_training_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created clean temp training dir: {temp_training_dir}")
        
        # ENHANCED: Training arguments for proper LoRA fine-tuning
        total_examples = len(training_data)
        steps_per_epoch = max(1, total_examples // (2 * 4))  # batch_size * grad_accumulation
        min_training_steps = max(50, steps_per_epoch * 3)    # Ensure minimum training
        
        training_args = TrainingArguments(
            output_dir=str(temp_training_dir),  # Use separate temp dir
            num_train_epochs=5,  # INCREASED: More epochs for better learning
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=min(20, min_training_steps // 4),  # Dynamic warmup
            logging_steps=max(5, min_training_steps // 10),  # More frequent logging
            save_strategy="no",  # Don't save checkpoints during training
            remove_unused_columns=True,
            fp16=torch.cuda.is_available(),
            report_to=None,
            label_names=["labels"],  # Fix the label_names warning
            max_steps=min_training_steps if total_examples < 50 else -1,  # Force minimum steps
        )
        
        print(f"📊 Training configuration:")
        print(f"   • Total examples: {total_examples}")
        print(f"   • Steps per epoch: {steps_per_epoch}")
        print(f"   • Minimum training steps: {min_training_steps}")
        print(f"   • Epochs: {training_args.num_train_epochs}")
        print(f"   • Warmup steps: {training_args.warmup_steps}")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        print(f"🏃 Starting {lora_config.adapter_type.upper()} training...")
        training_result = trainer.train()
        training_time = time.time() - start_time
        
        # CRITICAL FIX: Ensure target is completely clean before saving
        print(f"💾 Final cleanup and save to: {target_path}")
        
        # Double-check target is clean
        if target_path.exists():
            shutil.rmtree(target_path)
            print(f"   🗑️  Final removal of target directory")
        
        # Create fresh target directory
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"   📁 Created fresh target directory: {target_path}")
        
        # Save the trained adapter to clean directory
        model.save_pretrained(target_path)
        print(f"✅ Adapter saved to clean directory: {target_path}")
        
        # Clean up temporary training directory immediately
        if temp_training_dir.exists():
            shutil.rmtree(temp_training_dir)
            print(f"🧹 Cleaned up temporary training directory: {temp_training_dir}")
        
        # FINAL CLEANUP: Remove any remaining temp dirs that might have been created
        for item in parent_dir.glob("training_tmp*"):
            if item.is_dir() and item != target_path:
                shutil.rmtree(item)
                print(f"🧹 Final cleanup of stray temp dir: {item}")
        
        # ENHANCED VERIFICATION: Check for specific required files with retry logic
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        
        print(f"🔍 Verifying adapter files at: {target_path}")
        if not target_path.exists():
            raise RuntimeError(f"CRITICAL: Target directory not created: {target_path}")
        
        if not target_path.is_dir():
            raise RuntimeError(f"CRITICAL: Target path is not a directory: {target_path}")
        
        # CRITICAL: Check if any training_tmp directories leaked into target
        leaked_temp_dirs = list(target_path.glob("training_tmp*"))
        if leaked_temp_dirs:
            print(f"⚠️  WARNING: Found leaked temp directories in target: {leaked_temp_dirs}")
            for temp_dir in leaked_temp_dirs:
                if temp_dir.is_dir():
                    shutil.rmtree(temp_dir)
                    print(f"   🗑️  Removed leaked temp dir: {temp_dir}")
        
        # List all files for debugging
        all_files = list(target_path.glob("*"))
        all_dirs = [f for f in all_files if f.is_dir()]
        all_files_only = [f for f in all_files if f.is_file()]
        
        print(f"📄 Files in adapter directory: {[f.name for f in all_files_only]}")
        if all_dirs:
            print(f"📁 Directories in adapter directory: {[d.name for d in all_dirs]}")
            # If there are directories, something went wrong
            for dir_item in all_dirs:
                print(f"   📁 Directory contents of {dir_item.name}: {[x.name for x in dir_item.iterdir()]}")
        
        # Check for required files
        missing_files = []
        for required_file in required_files:
            file_path = target_path / required_file
            if not file_path.exists():
                missing_files.append(required_file)
            else:
                file_size = file_path.stat().st_size
                print(f"   ✅ {required_file}: {file_size} bytes")
        
        if missing_files:
            # CRITICAL: Enhanced error message with detailed diagnostics
            error_msg = (
                f"CRITICAL: Missing required adapter files: {missing_files}\n"
                f"Target directory: {target_path}\n"
                f"Files present: {[f.name for f in all_files_only]}\n"
                f"Directories present: {[d.name for d in all_dirs]}\n"
                f"PEFT save_pretrained() may have failed or saved to wrong location"
            )
            
            # Try to find where the files might have gone
            possible_locations = []
            for item in all_dirs:
                for required_file in required_files:
                    if (item / required_file).exists():
                        possible_locations.append(str(item / required_file))
            
            if possible_locations:
                error_msg += f"\nPossible file locations found: {possible_locations}"
                error_msg += "\nAdapter files may have been saved to subdirectory instead of target"
            
            raise RuntimeError(error_msg)
        
        print(f"✅ Adapter verification complete: {len(all_files_only)} files, all required files present")
        
        # Calculate model size
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
        ) / (1024 * 1024)
        
        print(f"✅ {lora_config.adapter_type.upper()} training completed in {training_time:.2f}s")
        print(f"📦 Adapter saved directly to: {target_path}")
        print(f"💾 Trainable parameters: {model_size_mb:.2f}MB")
        
        # CRITICAL: Validate training quality
        final_loss = training_result.training_loss
        expected_min_time = total_examples * 0.1  # Rough heuristic: 0.1s per example minimum
        
        print(f"🔍 Training Quality Assessment:")
        print(f"   • Final training loss: {final_loss:.4f}")
        print(f"   • Training time: {training_time:.2f}s")
        print(f"   • Expected minimum time: {expected_min_time:.2f}s")
        print(f"   • Training examples: {total_examples}")
        
        if training_time < expected_min_time:
            print(f"   ⚠️  WARNING: Training may be too fast for meaningful learning")
        if final_loss > 2.0:
            print(f"   ⚠️  WARNING: High final loss - may need more training")
        if total_examples < 50:
            print(f"   ⚠️  WARNING: Low training examples - consider expanding dataset")
        if total_examples >= 50 and training_time > expected_min_time and final_loss < 1.5:
            print(f"   ✅ Training quality indicators look good")
        
        # Return the adapter path string
        return str(target_path)
        
    except Exception as e:
        # Raise exception directly instead of returning error result
        raise RuntimeError(f"Training failed: {e}")


def train_lora_adapter(
    base_model_name: str,
    lora_config: LoRAConfig,
    training_data: list,
    cache_dir: str
) -> LoRATrainingResult:
    """
    Train LoRA adapter with given configuration using PEFT.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import os
        
        start_time = time.time()
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Training {lora_config.adapter_type.upper()} adapter: r={lora_config.r}, α={lora_config.alpha}, dropout={lora_config.dropout}")
        
        # Set HuggingFace cache environment properly
        model_cache_dir = "/cache/models" if Path("/cache").exists() else str(cache_path / "models")
        os.environ['HF_HOME'] = model_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
        
        print(f"📥 Loading base model: {base_model_name}")
        print(f"🗂️  Using model cache: {model_cache_dir}")
        
        # Check if cache exists and force offline if available
        expected_cache_path = Path(model_cache_dir) / f"models--{base_model_name.replace('/', '--')}"
        use_offline = expected_cache_path.exists()
        
        if use_offline:
            print(f"✅ Using cached model (offline mode)")
            os.environ['HF_HUB_OFFLINE'] = '1'
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=model_cache_dir,
            local_files_only=use_offline
        )
        
        # Fix: Set padding token for CodeLlama (required for batch training)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 Set padding token to EOS token: {tokenizer.eos_token}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=model_cache_dir,
            local_files_only=use_offline,
            low_cpu_mem_usage=True,  # Optimize memory usage
            use_safetensors=True     # Use safetensors format if available
        )
        
        # Configure LoRA/DoRA
        use_dora = lora_config.adapter_type == "dora" and _check_dora_support_runtime()
        adapter_name = "DoRA" if use_dora else "LoRA"
        print(f"🚀 Using {adapter_name} adapter: r={lora_config.r}, α={lora_config.alpha}")
        
        if lora_config.adapter_type == "dora" and not _check_dora_support_runtime():
            raise RuntimeError(
                f"FAIL-FAST: DoRA requested (adapter_type='dora') but not available.\n"
                f"Current peft version does not support use_dora parameter.\n"
                f"Install peft>=0.10 to use DoRA adapters.\n"
                f"NO FALLBACKS - fix your environment or change config to adapter_type='lora'"
            )
        
        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            use_dora=use_dora  # 🔥 Enable DoRA if requested and available
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Prepare training data for causal language modeling
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,  # Don't pad yet, handle in data collator
                max_length=512,
                return_tensors=None
            )
            # For causal LM, labels are the same as input_ids
            # CRITICAL FIX: Ensure labels are properly structured for batched processing
            if isinstance(tokenized["input_ids"][0], list):
                # Batched mode: input_ids is list of lists
                tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
            else:
                # Single example mode: input_ids is a single list
                tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Convert training data to dataset format
        formatted_data = []
        for item in training_data:
            # Format as instruction-following
            text = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Output: {item['output']}{tokenizer.eos_token}"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Remove unnecessary columns (keep only input_ids, attention_mask, labels)
        columns_to_remove = [col for col in tokenized_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
        if columns_to_remove:
            tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
            print(f"🧹 Removed unused columns: {columns_to_remove}")
        
        # CRITICAL: Validate tokenized data structure
        print("🔍 Validating tokenized dataset structure...")
        sample = tokenized_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Input IDs type: {type(sample['input_ids'])}, length: {len(sample['input_ids'])}")
        print(f"   Labels type: {type(sample['labels'])}, length: {len(sample['labels'])}")
        print(f"   Labels match input_ids: {sample['input_ids'] == sample['labels']}")
        
        # Validate no excessive nesting
        if isinstance(sample['input_ids'], list) and len(sample['input_ids']) > 0:
            if isinstance(sample['input_ids'][0], list):
                raise ValueError("CRITICAL: Excessive nesting detected in input_ids - found list of lists instead of flat list")
        if isinstance(sample['labels'], list) and len(sample['labels']) > 0:
            if isinstance(sample['labels'][0], list):
                raise ValueError("CRITICAL: Excessive nesting detected in labels - found list of lists instead of flat list")
        
        print("✅ Dataset structure validation passed")
        
        # CRITICAL FIX: Use DataCollatorForSeq2Seq for better label handling
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,  # Proper label padding
            return_tensors="pt"
        )
        
        # CRITICAL: Test data collator with sample batch
        print("🧪 Testing data collator with sample batch...")
        try:
            # Test with 2 samples to ensure batching works
            test_samples = [tokenized_dataset[0], tokenized_dataset[1]] if len(tokenized_dataset) > 1 else [tokenized_dataset[0]]
            test_batch = data_collator(test_samples)
            print(f"   Batch keys: {list(test_batch.keys())}")
            print(f"   Input IDs shape: {test_batch['input_ids'].shape}")
            print(f"   Labels shape: {test_batch['labels'].shape}")
            print(f"   Attention mask shape: {test_batch['attention_mask'].shape}")
            print("✅ Data collator test passed")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Data collator test failed: {e}")
        
        # CRITICAL: Mac MPS compatibility - detect device and adjust precision
        device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        use_fp16 = torch.cuda.is_available()  # Only use fp16 on CUDA GPUs
        
        print(f"🖥️  Training device: {device_type}, fp16: {use_fp16}")
        if device_type == "mps":
            print("📱 Mac MPS detected - using fp32 precision for compatibility")
        elif device_type == "cpu":
            print("💻 CPU training - using fp32 precision")
        
        # Training arguments - production optimized with CRITICAL FIXES
        training_args = TrainingArguments(
            output_dir=str(cache_path / "training_output"),
            num_train_epochs=3,                     # Reduced epochs for faster training
            per_device_train_batch_size=2 if device_type != "cuda" else 4,  # Smaller batch for non-CUDA
            gradient_accumulation_steps=4,          # Reduced accumulation for faster training
            learning_rate=2e-4,                     # Optimized learning rate for CodeLlama
            warmup_steps=50,                        # Reduced warmup for faster training
            logging_steps=25,                       # More frequent logging
            save_strategy="epoch",                  # Save at end of each epoch
            remove_unused_columns=True,             # Remove unused columns for clean training
            fp16=use_fp16,                          # FIXED: Only enable fp16 on CUDA
            dataloader_pin_memory=True,             # Optimize data loading
            load_best_model_at_end=False,           # Disable to avoid eval/save mismatch
            report_to=None,                         # Disable wandb/tensorboard
            label_names=["labels"],                 # CRITICAL FIX: Explicit label names for PeftModel
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        print(f"🏃 Starting LoRA training...")
        training_result = trainer.train()
        training_time = time.time() - start_time
        
        # Save adapter
        adapter_path = cache_path / f"lora_adapter_{int(time.time())}"
        model.save_pretrained(adapter_path)
        
        # Calculate model size
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
        ) / (1024 * 1024)
        
        print(f"✅ LoRA training completed in {training_time:.2f}s")
        print(f"📦 Adapter saved to: {adapter_path}")
        print(f"💾 Trainable parameters: {model_size_mb:.2f}MB")
        
        return LoRATrainingResult(
            adapter_path=str(adapter_path),
            training_time=training_time,
            training_loss=training_result.training_loss,
            model_size_mb=model_size_mb,
            success=True
        )
        
    except ImportError as e:
        return LoRATrainingResult(
            adapter_path="",
            training_time=0.0,
            training_loss=float('inf'),
            model_size_mb=0.0,
            success=False,
            error_message=f"Missing dependencies: {e}. Install with: pip install transformers peft datasets"
        )
    except Exception as e:
        return LoRATrainingResult(
            adapter_path="",
            training_time=0.0,
            training_loss=float('inf'),
            model_size_mb=0.0,
            success=False,
            error_message=f"Training failed: {e}"
        )


def load_lora_adapter_with_base_model(
    base_model_name: str,
    adapter_path: str,
    device: str = "auto"
):
    """
    Load base model with LoRA adapter applied.
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Use cache directory for model loading with proper HF cache setup
        model_cache_dir = "/cache/models" if Path("/cache").exists() else None
        print(f"📥 Loading base model: {base_model_name}")
        if model_cache_dir:
            print(f"🗂️  Using model cache: {model_cache_dir}")
            
            # Set HuggingFace cache environment
            import os
            os.environ['HF_HOME'] = model_cache_dir
            os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
            
            # Check if cache exists and force offline if available
            expected_cache_path = Path(model_cache_dir) / f"models--{base_model_name.replace('/', '--')}"
            use_offline = expected_cache_path.exists()
            
            if use_offline:
                print(f"✅ Using cached model (offline mode)")
                os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            use_offline = False
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device,
            cache_dir=model_cache_dir,
            local_files_only=use_offline if model_cache_dir else False,
            low_cpu_mem_usage=True,  # Optimize memory usage
            use_safetensors=True     # Use safetensors format if available
        )
        
        print(f"🔗 Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=model_cache_dir,
            local_files_only=use_offline if model_cache_dir else False
        )
        
        # Fix: Set padding token for CodeLlama (required for batch operations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 Set padding token to EOS token: {tokenizer.eos_token}")
        
        return model, tokenizer
        
    except ImportError as e:
        raise RuntimeError(f"Missing dependencies: {e}. Install with: pip install transformers peft")
    except Exception as e:
        raise RuntimeError(f"Failed to load adapter: {e}")


def prepare_training_data_from_problem(problem: Dict[str, Any]) -> list:
    """Pure function to prepare training data from QuixBugs problem."""
    # Extract training examples from problem
    problem_name = problem.get('name', 'unknown')
    buggy_code = problem.get('buggy_code', '')
    correct_code = problem.get('correct_code', '')
    
    if not correct_code:
        # FAIL-FAST: No hardcoded training data
        raise ValueError(
            f"FAIL-FAST: No correct implementation provided for '{problem_name}'. "
            f"Cannot create training data without reference solution."
        )
    
    # Format as instruction-following pairs
    training_examples = [
        {
            "instruction": f"Fix the bug in this Python function for {problem_name}:",
            "input": buggy_code,
            "output": correct_code
        }
    ]
    
    return training_examples 

# NEW: Runtime DoRA availability check - fail-fast, no fallbacks
def _check_dora_support_runtime() -> bool:
    """Check DoRA support at runtime - fail-fast if requested but unavailable."""
    try:
        import inspect
        from peft import LoraConfig as PeftLoraConfig
        
        # Check if LoraConfig has use_dora parameter
        sig = inspect.signature(PeftLoraConfig.__init__)
        return 'use_dora' in sig.parameters
    except Exception:
        return False 