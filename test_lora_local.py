#!/usr/bin/env python3
"""
Local LoRA Training Test - Debug training issues quickly
"""

def test_lora_training_locally():
    """Test LoRA training setup locally with minimal resources."""
    print("🧪 Local LoRA Training Test")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from transformers import DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import torch
        
        print("✅ All imports successful")
        
        # Use a much smaller model for local testing
        model_name = "microsoft/DialoGPT-small"  # Only ~117MB vs CodeLlama 13GB
        print(f"📥 Loading small test model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 Set padding token to: {tokenizer.eos_token}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"  # Force CPU for local test
        )
        print(f"✅ Model loaded: {model.config.model_type}")
        
        # Configure LoRA (small settings for test)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=4,                    # Small rank for testing
            lora_alpha=16.0,        # Moderate alpha
            lora_dropout=0.1,       # Low dropout
            target_modules=["c_attn"]  # DialoGPT attention module
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Create simple training data
        training_data = [
            {
                "instruction": "Fix this Python function",
                "input": "def add(a, b):\n    return a - b  # Bug: should be +",
                "output": "def add(a, b):\n    return a + b"
            },
            {
                "instruction": "Fix this Python function", 
                "input": "def multiply(x, y):\n    return x + y  # Bug: should be *",
                "output": "def multiply(x, y):\n    return x * y"
            }
        ]
        
        print(f"📊 Training data: {len(training_data)} examples")
        
        # Format and tokenize data
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,
                max_length=256,  # Shorter for local test
                return_tensors=None
            )
            # For causal LM, labels are the same as input_ids  
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Convert to dataset format
        formatted_data = []
        for item in training_data:
            text = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Output: {item['output']}{tokenizer.eos_token}"
            formatted_data.append({"text": text})
            print(f"📝 Example: {text[:100]}...")
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Remove unnecessary columns (keep only input_ids, attention_mask, labels)
        columns_to_remove = [col for col in tokenized_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        
        print(f"✅ Dataset tokenized: {len(tokenized_dataset)} examples")
        print(f"✅ Final columns: {tokenized_dataset.column_names}")
        
        # Check tokenized data structure
        sample = tokenized_dataset[0]
        print(f"🔍 Sample keys: {list(sample.keys())}")
        print(f"🔍 Input IDs length: {len(sample['input_ids'])}")
        print(f"🔍 Labels length: {len(sample['labels'])}")
        print(f"🔍 Labels match input_ids: {sample['input_ids'] == sample['labels']}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # Test data collator
        print("🧪 Testing data collator...")
        batch = data_collator([tokenized_dataset[0], tokenized_dataset[1]])
        print(f"✅ Batch keys: {list(batch.keys())}")
        print(f"✅ Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"✅ Batch labels shape: {batch['labels'].shape}")
        
        # Minimal training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,           # Just 1 epoch for test
            per_device_train_batch_size=1, # Small batch
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_steps=0,
            logging_steps=1,
            save_strategy="no",           # Don't save for test
            remove_unused_columns=True,   # Remove unused columns
            report_to=None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        print("🏃 Starting test training...")
        
        # Try a single forward pass first
        print("🧪 Testing single forward pass...")
        model.train()
        test_batch = data_collator([tokenized_dataset[0]])
        
        with torch.no_grad():
            outputs = model(**test_batch)
            print(f"✅ Forward pass successful!")
            print(f"🔍 Output keys: {list(outputs.keys())}")
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                print(f"✅ Loss computed: {outputs.loss.item():.4f}")
            else:
                print(f"❌ No loss in outputs!")
                return False
        
        # Try actual training
        training_result = trainer.train()
        
        print(f"✅ Training completed!")
        print(f"📊 Training loss: {training_result.training_loss:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install transformers peft datasets torch")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fix():
    """Suggest fixes based on test results."""
    print("\n🔧 SUGGESTED FIXES:")
    print("=" * 50)
    print("1. **Labels Issue**: Ensure labels = input_ids for causal LM")
    print("2. **Data Collator**: Use DataCollatorForLanguageModeling with mlm=False")
    print("3. **Model Type**: Verify model supports causal language modeling")
    print("4. **Tokenizer**: Ensure pad_token is set properly")
    print("5. **Loss Masking**: Check if special tokens need masking in labels")

if __name__ == "__main__":
    print("🚀 Starting local LoRA training test...")
    
    success = test_lora_training_locally()
    
    if success:
        print(f"\n✅ Local test PASSED!")
        print(f"The LoRA training setup works correctly.")
        print(f"The issue might be Modal-specific or CodeLlama-specific.")
    else:
        print(f"\n❌ Local test FAILED!")
        suggest_fix()
        
    print(f"\n📝 Next steps:")
    if success:
        print(f"   • Debug why CodeLlama behaves differently") 
        print(f"   • Check Modal environment differences")
        print(f"   • Verify CodeLlama tokenizer specifics")
    else:
        print(f"   • Fix the basic LoRA training setup")
        print(f"   • Re-test locally until it passes")
        print(f"   • Then deploy to Modal") 