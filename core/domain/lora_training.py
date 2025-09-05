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



@dataclass(frozen=True)
class AdapterConfig:
    """Immutable adapter configuration supporting both LoRA and DoRA."""
    r: int               # Use PEFT convention (r not rank)
    alpha: float
    dropout: float
    target_modules: list
    task_type: str = "CAUSAL_LM"
    adapter_type: str = "lora"  # "lora" or "dora"


# Backward compatibility alias
LoRAConfig = AdapterConfig


def train_lora_adapter_local(base_model: str, heavy_genes, save_path: str, config: Dict[str, Any]) -> str:
    """
    Train a LoRA adapter locally without Modal.
    For development/testing, this creates a minimal adapter structure.
    """
    from pathlib import Path
    import json
    import time

    print("      Training LoRA adapter locally...")
    print(f"         Base model: {base_model}")
    print(f"         Save path: {save_path}")

    # Create adapter directory
    adapter_path = Path(save_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Create adapter configuration
    adapter_config = {
        "r": heavy_genes.rank,
        "lora_alpha": heavy_genes.alpha,
        "lora_dropout": heavy_genes.dropout,
        "target_modules": list(heavy_genes.target_modules),
        "task_type": "CAUSAL_LM",
        "adapter_type": heavy_genes.adapter_type,
        "base_model": base_model
    }

    # Save adapter config
    config_path = adapter_path / "adapter_config.json"
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)

    # For local execution, we create a minimal adapter structure
    # In a real implementation, this would:
    # 1. Load the base model (tinyllama-1.1b-chat-v1.0)
    # 2. Set up LoRA configuration with PEFT
    # 3. Load the fake news dataset
    # 4. Run training loop with the specified parameters
    # 5. Save the trained adapter

    # Simulate training time based on parameters
    training_time = 1.0 + heavy_genes.rank * 0.1  # More parameters = longer training
    time.sleep(training_time)

    # Create a minimal adapter file (normally this would be the actual weights)
    adapter_weights_path = adapter_path / "adapter_model.bin"
    adapter_weights_path.write_text("# Simulated LoRA adapter weights\n# In real implementation, this would contain PyTorch tensors")

    print(f"      Local training complete: {adapter_path}")
    return str(adapter_path)


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


def train_codellama_lora(base_ckpt: str, heavy_genes: 'HeavyGenes', save_to: str, config: Dict[str, Any] = None) -> str:
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
    # Print all paths being used
    print("DEBUG PATHS - train_codellama_lora:")
    print(f"   • base_ckpt: {base_ckpt}")
    print(f"   • save_to: {save_to}")
    print(f"   • config provided: {config is not None}")
    if config:
        print(f"   • config keys: {list(config.keys())}")
        if 'paths' in config:
            print(f"   • config paths: {config['paths']}")
        if 'experiment' in config:
            print(f"   • experiment config: {config.get('experiment', {}).keys()}")

    # Validate input type for clarity
    from infra.adapter_cache import HeavyGenes
    if not isinstance(heavy_genes, HeavyGenes):
        raise TypeError(
            f"Expected HeavyGenes object, got {type(heavy_genes)}. "
            f"Use HeavyGenes.from_lora_config() to create from LoRAConfig."
        )

    # Extract values directly from HeavyGenes dataclass - clean and type-safe
    rank = heavy_genes.rank
    alpha = heavy_genes.alpha
    dropout = heavy_genes.dropout
    target_modules = heavy_genes.target_modules
    adapter_type = heavy_genes.adapter_type
    run_id = heavy_genes.run_id

    print(f"TRAIN_CODELLAMA_{adapter_type.upper()}: Starting {adapter_type.upper()} training")
    print(f"   Base checkpoint: {base_ckpt}")
    print(f"   Heavy genes: {heavy_genes}")
    print(f"   Save path: {save_to}")

    print(f"Adapter configuration: type={adapter_type}, r={rank}, α={alpha}, dropout={dropout}")



    # Create adapter configuration (supports both LoRA and DoRA)
    lora_config = AdapterConfig(
        r=int(rank),           # Use PEFT convention (r)
        alpha=float(alpha),
        dropout=float(dropout),
        target_modules=list(target_modules),
        adapter_type=adapter_type  # Use adapter_type from heavy_genes
    )

    # Prepare training data (load from QuixBugs if available)
    training_data = []
    try:
        # Try to load QuixBugs training data using auto-detection
        from adapters.quixbugs_real import QuixBugsRealAdapter

        print("Loading QuixBugs training data...")
        print(f"Creating QuixBugsRealAdapter with config: {config is not None}")
        adapter = QuixBugsRealAdapter(config=config)  # Pass config for dataset path
        print("QuixBugsRealAdapter created successfully")
        problems = list(adapter.problems())

        # Use centralized training problems list for DoRA/LoRA training
        # This MUST match what's excluded from evaluation to prevent data leakage
        from .dataset_constants import QUIXBUGS_TRAINING_PROBLEMS
        TRAINING_PROBLEMS = QUIXBUGS_TRAINING_PROBLEMS

        training_problems_found = 0
        training_examples_created = 0

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
                        "instruction": "The following function has a bug. Provide the corrected version:",
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
                training_problems_found += 1
                training_examples_created += len(base_examples)

        print(f"Loaded {len(training_data)} QuixBugs training examples")
        print("DORA/LORA TRAIN/TEST SPLIT VALIDATION:")
        print(f"   • Training problems: {len(TRAINING_PROBLEMS)} (CENTRALIZED - prevents data leakage)")
        print(f"   • Training problems found: {training_problems_found}")
        print(f"   • Training examples created: {training_examples_created} ({training_examples_created//training_problems_found if training_problems_found > 0 else 0}x per problem)")
        print(f"   • Total QuixBugs problems: {len(problems)}")
        print(f"   • Test problems available: {len(problems) - len(TRAINING_PROBLEMS)}")
        print("   • ANTI-CONTAMINATION: DoRA/LoRA adapters will NEVER see test problems")

        # Validate problem count matches expected
        if training_problems_found != len(TRAINING_PROBLEMS):
            print(f"WARNING: Expected {len(TRAINING_PROBLEMS)} training problems, got {training_problems_found}")
        else:
            print(f"Training problem count matches expected: {len(TRAINING_PROBLEMS)} problems")

    except Exception as e:
        raise RuntimeError(
            f"QuixBugs training data not available: {e}\n"
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
        save_directly_to=save_to,  # Train directly to Modal volume
        config=config  # Pass config for path resolution
    )

    # Verify adapter exists at expected location
    if not target_path.exists():
        raise RuntimeError(f"Adapter not found after training: {save_to}")

    print(f"TRAIN_CODELLAMA_{adapter_type.upper()} completed: {save_to}")



    # Return the path string as expected by adapter cache
    return save_to


def train_lora_adapter_direct(
    base_model_name: str,
    lora_config: AdapterConfig,
    training_data: list,
    save_directly_to: str,
    config: Dict[str, Any] = None
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

        print(f"Training {lora_config.adapter_type.upper()} adapter directly to: {target_path}")
        print(f"   r={lora_config.r}, α={lora_config.alpha}, dropout={lora_config.dropout}")

        # Use centralized path resolution
        if config is not None:
            from core.config.path_utils import create_path_config_from_dict, get_model_cache_path

            # Get executor type from config
            executor_type = config.get('infra', {}).get('executor', 'modal')
            path_config = create_path_config_from_dict(config, executor_type)
            model_cache_dir = get_model_cache_path(path_config)

            print(f"Loading base model: {base_model_name}")
            print(f"Using model cache: {model_cache_dir} (from {executor_type} config)")
        else:
            raise RuntimeError(
                "No config provided for model caching. "
                "Cache directory cannot be determined. "
                "Configuration is required for proper model management."
            )

        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)

        # Robust model loading: try cache first, then download if needed
        try:
            print("Attempting to load from cache...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                cache_dir=model_cache_dir,
                local_files_only=True  # Try cache first
            )
            print("✅ Tokenizer loaded from cache")
        except Exception as cache_error:
            print(f"⚠️  Cache miss for tokenizer: {cache_error}")
            print("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                cache_dir=model_cache_dir,
                local_files_only=False  # Download if not cached
            )
            print("✅ Tokenizer downloaded and cached")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 Set padding token to EOS token: {tokenizer.eos_token}")

        try:
            print("🔄 Attempting to load model from cache...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=model_cache_dir,
                local_files_only=True,  # Try cache first
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded from cache")
        except Exception as cache_error:
            print(f"⚠️  Cache miss for model: {cache_error}")
            print("📥 Downloading model... (this may take several minutes)")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=model_cache_dir,
                local_files_only=False,  # Download if not cached
                low_cpu_mem_usage=True
            )
            print("✅ Model downloaded and cached")

        # 🔥 CONFIG-DRIVEN: Configure adapter based on config
        if lora_config.adapter_type == "dora":
            # Check DoRA support at runtime - fail immediately if not available
            if not _check_dora_support_runtime():
                raise RuntimeError(
                    "DoRA requested (adapter_type='dora') but not available.\n"
                    "Current peft version does not support use_dora parameter.\n"
                    "Install peft>=0.10 to use DoRA adapters.\n"
                    "NO FALLBACKS - fix your environment or change config to adapter_type='lora'"
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
        import shutil
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

        # Aggressive cleanup of target directory and any leftover temp dirs
        print("AGGRESSIVE CLEANUP: Cleaning target and temp directories")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove target directory completely if exists
        if target_path.exists():
            shutil.rmtree(target_path)
            print(f"   Removed existing target: {target_path}")

        # Clean up ANY leftover training_tmp directories in parent directory
        parent_dir = target_path.parent
        for item in parent_dir.glob("training_tmp*"):
            if item.is_dir():
                shutil.rmtree(item)
                print(f"   Removed leftover temp dir: {item}")

        # Create unique temporary directory with process ID to avoid collisions
        import uuid
        temp_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        temp_training_dir = parent_dir / f"training_tmp_{temp_suffix}"
        temp_training_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created clean temp training dir: {temp_training_dir}")

        # Training arguments for proper LoRA fine-tuning
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

        print("📊 Training configuration:")
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

        # Ensure target is completely clean before saving
        print(f"💾 Final cleanup and save to: {target_path}")

        # Double-check target is clean
        if target_path.exists():
            shutil.rmtree(target_path)
            print("   🗑️  Final removal of target directory")

        # Create fresh target directory
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"   📁 Created fresh target directory: {target_path}")

        # Set correct base model path for PEFT
        # PEFT needs to find the base model config during save - ensure it's available

        # Find the actual cached model directory (HuggingFace cache format)
        hf_cache_model_dir = Path(model_cache_dir) / f"models--{base_model_name.replace('/', '--')}"

        print(f"PEFT save preparation: {base_model_name} -> {target_path}")

        # Check if the HuggingFace cache directory exists
        if not hf_cache_model_dir.exists():
            raise RuntimeError(
                f"HuggingFace cache directory not found: {hf_cache_model_dir}\n"
                f"Model cache is corrupted or incomplete.\n"
                f"Run: modal run coral_queue_modal_app.py::setup_model_cache\n"
                f"NO FALLBACKS - fix the cache setup."
            )

        # Look for the actual model directory with snapshots
        snapshots_dir = hf_cache_model_dir / "snapshots"
        if not snapshots_dir.exists():
            raise RuntimeError(
                f"No snapshots directory in cache: {snapshots_dir}\n"
                f"HuggingFace cache structure is invalid.\n"
                f"Delete cache and re-run: modal run coral_queue_modal_app.py::setup_model_cache\n"
                f"NO FALLBACKS - fix the cache structure."
            )

        # Get the snapshot directory
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            raise RuntimeError(
                f"No snapshots found in: {snapshots_dir}\n"
                f"Model cache is empty or corrupted.\n"
                f"Delete cache and re-run: modal run coral_queue_modal_app.py::setup_model_cache\n"
                f"NO FALLBACKS - fix the cache contents."
            )

        # Use the first (and usually only) snapshot directory
        actual_model_path = snapshot_dirs[0]

        # Verify config.json exists
        config_json = actual_model_path / "config.json"
        if not config_json.exists():
            raise RuntimeError(
                f"config.json not found: {config_json}\n"
                f"Model cache is incomplete - missing configuration file.\n"
                f"Delete cache and re-run: modal run coral_queue_modal_app.py::setup_model_cache\n"
                f"NO FALLBACKS - ensure complete model cache."
            )

        peft_base_path = str(actual_model_path)
        print(f"Using cached model path: {peft_base_path}")

        # Update the model's base_model_name_or_path to the correct path
        try:
            if hasattr(model, 'peft_config') and model.peft_config:
                for config_name, peft_config in model.peft_config.items():
                    peft_config.base_model_name_or_path = peft_base_path
            else:
                raise RuntimeError("No PEFT config found - model may not be properly configured")
        except Exception as config_error:
            raise RuntimeError(
                f"Cannot update PEFT config base model path: {config_error}\n"
                f"PEFT model configuration is invalid or corrupted.\n"
                f"This indicates a deeper issue with the training process."
            )

        # Ensure HuggingFace environment is properly set for PEFT to find cached model
        import os
        os.environ['HF_HOME'] = model_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
        os.environ['HF_DATASETS_CACHE'] = model_cache_dir

        # Save the trained adapter - NO FALLBACKS
        print(f"Saving adapter to: {target_path}")

        try:
            model.save_pretrained(
                target_path,
                save_config=True,
                save_safetensors=True
            )
            print(f"Adapter saved successfully: {target_path}")
        except Exception as save_error:
            # Enhanced diagnostics with cache inspection
            cache_diagnostics = []

            # Check what's actually in the cache
            if Path(model_cache_dir).exists():
                try:
                    cache_items = list(Path(model_cache_dir).iterdir())
                    cache_diagnostics.append(f"Cache directory exists with {len(cache_items)} items")
                    for item in cache_items[:5]:  # Show first 5 items
                        cache_diagnostics.append(f"  • {item.name}")
                    if len(cache_items) > 5:
                        cache_diagnostics.append(f"  • ... and {len(cache_items) - 5} more")
                except Exception as e:
                    cache_diagnostics.append(f"Cache directory exists but cannot list: {e}")
            else:
                cache_diagnostics.append(f"Cache directory does not exist: {model_cache_dir}")

            # Check PEFT model state
            peft_diagnostics = []
            try:
                if hasattr(model, 'peft_config'):
                    for name, config in model.peft_config.items():
                        peft_diagnostics.append(f"PEFT config '{name}': base_model={config.base_model_name_or_path}")
                else:
                    peft_diagnostics.append("No PEFT config found")
            except Exception as e:
                peft_diagnostics.append(f"Cannot inspect PEFT config: {e}")

            # Clear error message with comprehensive diagnostics
            raise RuntimeError(
                f"Adapter save failed: {save_error}\n"
                f"\nTarget: {target_path}\n"
                f"Base model: {base_model_name}\n"
                f"\nCache Diagnostics:\n" + "\n".join(cache_diagnostics) + "\n"
                "\nPEFT Diagnostics:\n" + "\n".join(peft_diagnostics) + "\n"
                "\nSOLUTIONS:\n"
                "1. Pre-cache the model: modal run coral_queue_modal_app.py::setup_model_cache\n"
                "2. Check HuggingFace token access: export HF_TOKEN=your_token\n"
                "3. Verify peft version: pip install peft>=0.10\n"
                "\nNO FALLBACKS - fix the root cause."
            )

        # Clean up temporary training directory immediately
        if temp_training_dir.exists():
            shutil.rmtree(temp_training_dir)
            print(f"Cleaned up temporary training directory: {temp_training_dir}")

        # FINAL CLEANUP: Remove any remaining temp dirs that might have been created
        for item in parent_dir.glob("training_tmp*"):
            if item.is_dir() and item != target_path:
                shutil.rmtree(item)
                print(f"Final cleanup of stray temp dir: {item}")

        # Check for specific required files with retry logic
        required_files = ['adapter_config.json', 'adapter_model.safetensors']

        print("Verifying adapter files...")
        if not target_path.exists():
            raise RuntimeError(f"Target directory not created: {target_path}")

        if not target_path.is_dir():
            raise RuntimeError(f"Target path is not a directory: {target_path}")

        # Check if any training_tmp directories leaked into target
        leaked_temp_dirs = list(target_path.glob("training_tmp*"))
        if leaked_temp_dirs:
            for temp_dir in leaked_temp_dirs:
                if temp_dir.is_dir():
                    shutil.rmtree(temp_dir)

        # List all files for validation
        all_files = list(target_path.glob("*"))
        all_dirs = [f for f in all_files if f.is_dir()]
        all_files_only = [f for f in all_files if f.is_file()]

        # If there are directories, something went wrong
        if all_dirs:
            for dir_item in all_dirs:
                print(f"   Unexpected directory: {dir_item.name}")

        # Check for required files
        missing_files = []
        for required_file in required_files:
            file_path = target_path / required_file
            if not file_path.exists():
                missing_files.append(required_file)

        if missing_files:
            # Enhanced error message with detailed diagnostics
            error_msg = (
                f"Missing required adapter files: {missing_files}\n"
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

        print(f"Adapter verification complete: {len(all_files_only)} files")

        # Calculate model size
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
        ) / (1024 * 1024)

        print(f"{lora_config.adapter_type.upper()} training completed in {training_time:.2f}s")
        print(f"Adapter saved to: {target_path}")
        print(f"Trainable parameters: {model_size_mb:.2f}MB")

        # Validate training quality
        final_loss = training_result.training_loss
        expected_min_time = total_examples * 0.1  # Rough heuristic: 0.1s per example minimum

        # Quality assessment with concise output
        quality_warnings = []
        if training_time < expected_min_time:
            quality_warnings.append("Training may be too fast")
        if final_loss > 2.0:
            quality_warnings.append("High final loss")
        if total_examples < 50:
            quality_warnings.append("Low training examples")

        if quality_warnings:
            print(f"Training quality warnings: {', '.join(quality_warnings)}")
        else:
            print(f"Training quality indicators look good (loss: {final_loss:.3f})")

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
            print("✅ Using cached model (offline mode)")
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
                "DoRA requested (adapter_type='dora') but not available.\n"
                "Current peft version does not support use_dora parameter.\n"
                "Install peft>=0.10 to use DoRA adapters.\n"
                "NO FALLBACKS - fix your environment or change config to adapter_type='lora'"
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
            # Ensure labels are properly structured for batched processing
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

        # Validate tokenized data structure
        print("🔍 Validating tokenized dataset structure...")
        sample = tokenized_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Input IDs type: {type(sample['input_ids'])}, length: {len(sample['input_ids'])}")
        print(f"   Labels type: {type(sample['labels'])}, length: {len(sample['labels'])}")
        print(f"   Labels match input_ids: {sample['input_ids'] == sample['labels']}")

        # Validate no excessive nesting
        if isinstance(sample['input_ids'], list) and len(sample['input_ids']) > 0:
            if isinstance(sample['input_ids'][0], list):
                raise ValueError("Excessive nesting detected in input_ids - found list of lists instead of flat list")
        if isinstance(sample['labels'], list) and len(sample['labels']) > 0:
            if isinstance(sample['labels'][0], list):
                raise ValueError("Excessive nesting detected in labels - found list of lists instead of flat list")

        print("✅ Dataset structure validation passed")

        # Use DataCollatorForSeq2Seq for better label handling
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,  # Proper label padding
            return_tensors="pt"
        )

        # Test data collator with sample batch
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
            raise RuntimeError(f"Data collator test failed: {e}")

        # Mac MPS compatibility - detect device and adjust precision
        device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        use_fp16 = torch.cuda.is_available()  # Only use fp16 on CUDA GPUs

        print(f"🖥️  Training device: {device_type}, fp16: {use_fp16}")
        if device_type == "mps":
            print("📱 Mac MPS detected - using fp32 precision for compatibility")
        elif device_type == "cpu":
            print("💻 CPU training - using fp32 precision")

        # Training arguments - production optimized
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
            label_names=["labels"],                 # Explicit label names for PeftModel
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Train the model
        print("🏃 Starting LoRA training...")
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
        print(f"Trainable parameters: {model_size_mb:.2f}MB")

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
                print("✅ Using cached model (offline mode)")
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
        # No hardcoded training data
        raise ValueError(
            f"  No correct implementation provided for '{problem_name}'. "
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

# NEW: Runtime DoRA availability check
def _check_dora_support_runtime() -> bool:
    """Check DoRA support at runtime."""
    try:
        import inspect
        from peft import LoraConfig as PeftLoraConfig

        # Check if LoraConfig has use_dora parameter
        sig = inspect.signature(PeftLoraConfig.__init__)
        return 'use_dora' in sig.parameters
    except Exception:
        return False
