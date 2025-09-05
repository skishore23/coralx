"""LoRA training service for CORAL-X."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator
from datasets import Dataset

from core.common.exceptions import ModelError, ValidationError
from core.common.logging import get_logger
from core.domain.genome import Genome, LoRAConfig as CoralLoRAConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingMetrics:
    """Training metrics for LoRA adaptation."""
    training_time: float
    final_loss: float
    steps_completed: int
    memory_used: float
    convergence_epoch: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "training_time": self.training_time,
            "final_loss": self.final_loss,
            "steps_completed": self.steps_completed,
            "memory_used": self.memory_used,
            "convergence_epoch": self.convergence_epoch
        }


@dataclass(frozen=True)
class LoRATrainingConfig:
    """Configuration for LoRA training."""
    base_model_name: str
    output_dir: Path
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    save_strategy: str = "steps"
    evaluation_strategy: str = "no"
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    report_to: Optional[str] = None
    seed: int = 42


class LoRATrainingService:
    """Service for training LoRA adapters on base models.
    
    This service handles:
    - Loading base models and tokenizers
    - Creating LoRA configurations from genome parameters
    - Training LoRA adapters with proper error handling
    - Saving and loading trained adapters
    - Memory management and optimization
    """

    def __init__(self, config: LoRATrainingConfig):
        """Initialize LoRA training service.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.accelerator = Accelerator()
        self._model_cache: Dict[str, Any] = {}
        self._tokenizer_cache: Dict[str, Any] = {}

        logger.info(f"LoRA training service initialized for {config.base_model_name}")

    def train_adapter(self,
                     genome: Genome,
                     training_data: Dataset,
                     validation_data: Optional[Dataset] = None) -> Tuple[Path, TrainingMetrics]:
        """Train a LoRA adapter from genome parameters.
        
        Args:
            genome: Genome containing LoRA configuration
            training_data: Training dataset
            validation_data: Optional validation dataset
            
        Returns:
            Tuple of (adapter_path, training_metrics)
            
        Raises:
            ModelError: If training fails
            ValidationError: If inputs are invalid
        """
        if not genome.lora_cfg:
            raise ValidationError("Genome must contain LoRA configuration")

        if not training_data:
            raise ValidationError("Training data cannot be empty")

        logger.info(f"Starting LoRA training for genome {genome.id}")
        start_time = time.time()

        try:
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer()

            # Create LoRA configuration
            lora_config = self._create_lora_config(genome.lora_cfg)

            # Apply LoRA to model
            model = get_peft_model(model, lora_config)

            # Prepare data
            train_dataloader = self._prepare_dataloader(training_data, tokenizer, is_training=True)
            eval_dataloader = None
            if validation_data:
                eval_dataloader = self._prepare_dataloader(validation_data, tokenizer, is_training=False)

            # Setup training arguments
            training_args = self._create_training_arguments()

            # Create trainer
            trainer = self._create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                training_args=training_args
            )

            # Train the model
            training_result = trainer.train()

            # Save the adapter
            adapter_path = self._save_adapter(model, genome.id)

            # Calculate metrics
            training_time = time.time() - start_time
            memory_used = self._get_memory_usage()

            metrics = TrainingMetrics(
                training_time=training_time,
                final_loss=training_result.training_loss,
                steps_completed=training_result.global_step,
                memory_used=memory_used
            )

            logger.info(f"LoRA training completed for genome {genome.id} in {training_time:.2f}s")
            return adapter_path, metrics

        except Exception as e:
            logger.error(f"LoRA training failed for genome {genome.id}: {e}")
            raise ModelError(f"LoRA training failed: {e}", cause=e)

    def load_adapter(self, adapter_path: Path, base_model_name: Optional[str] = None) -> PeftModel:
        """Load a trained LoRA adapter.
        
        Args:
            adapter_path: Path to the adapter directory
            base_model_name: Base model name (if different from config)
            
        Returns:
            Loaded PEFT model with adapter
            
        Raises:
            ModelError: If loading fails
        """
        try:
            model_name = base_model_name or self.config.base_model_name
            model, _ = self._load_model_and_tokenizer(model_name)

            # Load the adapter
            model = PeftModel.from_pretrained(model, str(adapter_path))

            logger.info(f"LoRA adapter loaded from {adapter_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load LoRA adapter from {adapter_path}: {e}")
            raise ModelError(f"Failed to load LoRA adapter: {e}", cause=e)

    def _load_model_and_tokenizer(self, model_name: Optional[str] = None) -> Tuple[Any, Any]:
        """Load model and tokenizer with caching.
        
        Args:
            model_name: Model name (uses config default if None)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or self.config.base_model_name

        # Check cache first
        if model_name in self._model_cache:
            return self._model_cache[model_name], self._tokenizer_cache[model_name]

        try:
            logger.debug(f"Loading model and tokenizer: {model_name}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # Cache for reuse
            self._model_cache[model_name] = model
            self._tokenizer_cache[model_name] = tokenizer

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelError(f"Failed to load model {model_name}: {e}", cause=e)

    def _create_lora_config(self, lora_cfg: CoralLoRAConfig) -> LoraConfig:
        """Create LoRA configuration from genome parameters.
        
        Args:
            lora_cfg: LoRA configuration from genome
            
        Returns:
            PEFT LoRA configuration
        """
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias="none",
            inference_mode=False
        )

    def _prepare_dataloader(self,
                           dataset: Dataset,
                           tokenizer: Any,
                           is_training: bool = True) -> DataLoader:
        """Prepare data loader for training.
        
        Args:
            dataset: Dataset to prepare
            tokenizer: Tokenizer for text processing
            is_training: Whether this is for training (vs evaluation)
            
        Returns:
            Prepared DataLoader
        """
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Create data loader
        return DataLoader(
            tokenized_dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            collate_fn=data_collator
        )

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments.
        
        Returns:
            TrainingArguments instance
        """
        return TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            adam_epsilon=self.config.adam_epsilon,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=self.config.report_to,
            seed=self.config.seed,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=torch.cuda.is_available(),
            remove_unused_columns=False
        )

    def _create_trainer(self,
                       model: Any,
                       tokenizer: Any,
                       train_dataloader: DataLoader,
                       eval_dataloader: Optional[DataLoader],
                       training_args: TrainingArguments) -> Trainer:
        """Create trainer instance.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            training_args: Training arguments
            
        Returns:
            Trainer instance
        """
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset if eval_dataloader else None,
            tokenizer=tokenizer,
            data_collator=train_dataloader.collate_fn
        )

    def _save_adapter(self, model: Any, genome_id: str) -> Path:
        """Save trained adapter.
        
        Args:
            model: Trained model with adapter
            genome_id: Genome identifier for naming
            
        Returns:
            Path to saved adapter
        """
        adapter_path = self.config.output_dir / f"adapter_{genome_id}"
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Save adapter
        model.save_pretrained(str(adapter_path))

        logger.info(f"Adapter saved to {adapter_path}")
        return adapter_path

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0

    def cleanup_cache(self) -> None:
        """Clean up model and tokenizer cache."""
        self._model_cache.clear()
        self._tokenizer_cache.clear()
        logger.info("Model cache cleaned up")
