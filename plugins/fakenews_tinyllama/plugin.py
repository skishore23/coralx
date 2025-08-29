"""
FakeNews + TinyLlama Plugin for CORAL-X
Integrates fake news detection with TinyLlama fine-tuning using Unsloth optimization
Config-driven implementation
"""
import json
import os
import numpy as np
import pandas as pd
import subprocess
import yaml
from pathlib import Path
from typing import Iterable, Dict, Any, Callable, List, Optional
from dataclasses import dataclass
import time

# CoralX Core Imports
from core.ports.interfaces import DatasetProvider, ModelRunner, FitnessFn
from core.domain.mapping import LoRAConfig
from core.domain.genome import Genome, MultiObjectiveScores
from core.domain.cheap_knobs import CheapKnobs, cheap_knobs_to_generation_kwargs

# Infrastructure Imports
from infra.adapter_cache import HeavyGenes, CacheConfig, get_or_train_adapter


@dataclass(frozen=True)
class FakeNewsDetectionMetrics:
    """Immutable fake news detection metrics optimized for misinformation detection."""
    # P1: Detection Accuracy (↑) - Overall fake vs real classification accuracy
    detection_accuracy: float
    
    # P2: Fake News Recall (↑) - Sensitivity - catch dangerous misinformation  
    fake_news_recall: float
    
    # P3: Real News Precision (↑) - Specificity - don't suppress legitimate journalism
    real_news_precision: float
    
    # P4: Cross-Source Robustness (↑) - Performance consistency across news sources
    cross_source_robustness: float
    
    # P5: Confidence Calibration (↑) - Reliability of confidence scores
    confidence_calibration: float
    
    # P6: Efficiency (↑) - Inference speed for real-time deployment
    efficiency_score: float
    
    def overall_score(self) -> float:
        """Pure function: metrics → overall score optimized for fake news detection."""
        # All metrics are already 0-1 scale where 1 is better
        
        # Weighted average optimized for fake news detection priorities
        return (
            0.30 * self.detection_accuracy +       # Overall accuracy is critical
            0.25 * self.fake_news_recall +         # Don't miss dangerous misinformation
            0.25 * self.real_news_precision +      # Don't suppress legitimate journalism
            0.10 * self.cross_source_robustness +  # Generalization across sources
            0.07 * self.confidence_calibration +   # Reliable confidence scoring
            0.03 * self.efficiency_score           # Real-time deployment readiness
        )


class MultiModalAISafetyDatasetProvider(DatasetProvider):
    """Multi-modal AI safety dataset provider - supports fake news, deepfake audio/video, and safety evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        self._setup_datasets()
    
    def _validate_config(self):
        """Configuration validation with immediate failure on errors."""
        required_fields = ['dataset_path', 'max_samples', 'datasets']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"  '{field}' missing from dataset config")
        
        # Validate dataset types
        supported_datasets = ['fake_news', 'deepfake_audio', 'deepfake_video', 'jailbreak_prompts', 'clean_holdout']
        for dataset_name in self.config['datasets']:
            if dataset_name not in supported_datasets:
                raise ValueError(f"  Unsupported dataset '{dataset_name}'. Supported: {supported_datasets}")
    
    def _setup_datasets(self):
        """Setup multi-modal AI safety datasets with validation."""
        dataset_path = Path(self.config['dataset_path'])
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {}
        
        for dataset_name in self.config['datasets']:
            print(f"Setting up {dataset_name} dataset...")
            
            if dataset_name == 'fake_news':
                self.datasets[dataset_name] = self._setup_fake_news_dataset(dataset_path)
            elif dataset_name == 'deepfake_audio':
                self.datasets[dataset_name] = self._setup_deepfake_audio_dataset(dataset_path)
            elif dataset_name == 'deepfake_video':
                self.datasets[dataset_name] = self._setup_deepfake_video_dataset(dataset_path)
            elif dataset_name == 'jailbreak_prompts':
                self.datasets[dataset_name] = self._setup_jailbreak_dataset(dataset_path)
            elif dataset_name == 'clean_holdout':
                self.datasets[dataset_name] = self._setup_clean_holdout_dataset(dataset_path)
        
        total_samples = sum(len(data) for data in self.datasets.values())
        print(f"Multi-modal datasets loaded: {total_samples} total samples across {len(self.datasets)} datasets")
    
    def problems(self) -> Iterable[Dict[str, Any]]:
        """Yield multi-modal AI safety problems - Pure generator function."""
        problem_id = 0
        
        for dataset_name, data in self.datasets.items():
            for idx, row in data.iterrows():
                problem = {
                    'id': f"{dataset_name}_{problem_id}",
                    'dataset_type': dataset_name,
                    'text': str(row['text'])[:2048] if 'text' in row else '',
                    'label': int(row['label']) if 'label' in row else 0,
                    'source': row.get('source', 'unknown'),
                }
                
                # Add modality-specific fields
                if dataset_name == 'deepfake_audio':
                    problem['audio_path'] = row.get('audio_path', '')
                    problem['audio_features'] = row.get('audio_features', '')
                elif dataset_name == 'deepfake_video':
                    problem['video_path'] = row.get('video_path', '')
                    problem['frame_features'] = row.get('frame_features', '')
                elif dataset_name == 'jailbreak_prompts':
                    problem['attack_type'] = row.get('attack_type', 'unknown')
                    problem['expected_refusal'] = row.get('expected_refusal', True)
                elif dataset_name == 'clean_holdout':
                    problem['is_clean'] = True
                
                yield problem
                problem_id += 1
    
    def _download_fakenews_dataset(self, dataset_path: Path):
        """Download FakeNewsNet dataset using Kaggle API."""
        # Verify Kaggle credentials are configured
        if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
            raise RuntimeError(
                "Kaggle credentials not set. "
                "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables. "
                "Get credentials from https://www.kaggle.com/account"
            )
        
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download using kaggle CLI
        try:
            subprocess.run([
                'kaggle', 'datasets', 'download', 'mdepak/fakenewsnet',
                '-p', str(dataset_path.parent)
            ], check=True)
            
            # Extract dataset
            subprocess.run([
                'unzip', '-q', str(dataset_path.parent / 'fakenewsnet.zip'),
                '-d', str(dataset_path)
            ], check=True)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FakeNewsNet dataset download failed: {e}. Please check your Kaggle credentials and network connection.")
        except FileNotFoundError:
            raise RuntimeError(
                "Kaggle CLI not found. Install with 'pip install kaggle' "
                "and ensure ~/.kaggle/kaggle.json contains your credentials."
            )
    
    
    def _load_fakenews_data(self, dataset_path: Path) -> pd.DataFrame:
        """Load and preprocess fake news data with validation."""
        # First, check for the actual dataset files we have
        politifact_fake = dataset_path / "PolitiFact_fake_news_content.csv"
        politifact_real = dataset_path / "PolitiFact_real_news_content.csv"
        buzzfeed_fake = dataset_path / "BuzzFeed_fake_news_content.csv"
        buzzfeed_real = dataset_path / "BuzzFeed_real_news_content.csv"
        
        # Try to load from existing Politifact files
        if politifact_fake.exists() and politifact_real.exists():
            print(f"Loading Politifact dataset from existing files")
            fake_data = pd.read_csv(politifact_fake, nrows=self.config['max_samples']//2)
            real_data = pd.read_csv(politifact_real, nrows=self.config['max_samples']//2)
            
            # Add labels
            fake_data['label'] = 1
            real_data['label'] = 0
            
            # Combine datasets
            data = pd.concat([fake_data, real_data], ignore_index=True)
            
            # Standardize column names - Politifact uses 'content' column
            if 'text' not in data.columns and 'content' in data.columns:
                data['text'] = data['content']
                print("Using 'content' column as 'text'")
        
        # Try to load from existing BuzzFeed files
        elif buzzfeed_fake.exists() and buzzfeed_real.exists():
            print(f"Loading BuzzFeed dataset from existing files")
            fake_data = pd.read_csv(buzzfeed_fake, nrows=self.config['max_samples']//2)
            real_data = pd.read_csv(buzzfeed_real, nrows=self.config['max_samples']//2)
            
            # Add labels
            fake_data['label'] = 1
            real_data['label'] = 0
            
            # Combine datasets
            data = pd.concat([fake_data, real_data], ignore_index=True)
            
            # Standardize column names - BuzzFeed uses 'content' column
            if 'text' not in data.columns and 'content' in data.columns:
                data['text'] = data['content']
                print("Using 'content' column as 'text'")
        
        else:
            raise RuntimeError(f"No valid dataset files found in {dataset_path}. Expected PolitiFact or BuzzFeed CSV files.")
        
        # Validate required columns
        required_cols = ['text', 'label']
        for col in required_cols:
            if col not in data.columns:
                # Try alternative column names
                if col == 'text':
                    for alt_col in ['title', 'content', 'article', 'news', 'headline']:
                        if alt_col in data.columns:
                            data['text'] = data[alt_col]
                            print(f"Using '{alt_col}' column as 'text'")
                            break
                elif col == 'label':
                    for alt_col in ['is_fake', 'fake', 'target', 'class', 'category']:
                        if alt_col in data.columns:
                            data['label'] = data[alt_col].astype(int)
                            print(f"Using '{alt_col}' column as 'label'")
                            break
                
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in dataset. Available columns: {list(data.columns)}")
        
        # Clean and validate data
        data = data.dropna(subset=['text', 'label'])
        data['label'] = data['label'].astype(int)
        data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data
        
        print(f"Dataset loaded: {len(data)} samples, {data['label'].sum()} fake, {len(data) - data['label'].sum()} real")
        
        if len(data) == 0:
            raise RuntimeError("No valid data samples found after cleaning")
        
        return data
    
    def _setup_fake_news_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Setup fake news dataset with existing file detection."""
        fake_news_path = dataset_path / "fake_news"
        
        # Check if data already exists in the expected format
        train_file = fake_news_path / "train.csv"
        test_file = fake_news_path / "test.csv"
        
        if train_file.exists() and test_file.exists():
            print(f"Using existing fake news data at {fake_news_path}")
            return self._load_fakenews_data(fake_news_path)
        
        # Check if we have the actual dataset files that were downloaded
        politifact_fake = fake_news_path / "PolitiFact_fake_news_content.csv"
        politifact_real = fake_news_path / "PolitiFact_real_news_content.csv"
        buzzfeed_fake = fake_news_path / "BuzzFeed_fake_news_content.csv"
        buzzfeed_real = fake_news_path / "BuzzFeed_real_news_content.csv"
        
        if (politifact_fake.exists() and politifact_real.exists()) or (buzzfeed_fake.exists() and buzzfeed_real.exists()):
            print(f"Using existing downloaded dataset files at {fake_news_path}")
            return self._load_fakenews_data(fake_news_path)
        
        # Only download if no data exists
        print(f"No existing dataset found, downloading real Kaggle fake news dataset to {fake_news_path}")
        self._download_fakenews_dataset(fake_news_path)
        return self._load_fakenews_data(fake_news_path)
    
    def _load_audio_dataset(self, audio_path: Path) -> pd.DataFrame:
        """Load deepfake audio dataset."""
        csv_files = list(audio_path.glob('**/*.csv'))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in audio dataset: {audio_path}")
        
        data = pd.read_csv(csv_files[0], nrows=self.config['max_samples']//4)
        data['label'] = data.get('label', data.get('fake', 0)).astype(int)
        data['text'] = data.get('transcript', data.get('text', 'Audio sample'))
        return data.dropna(subset=['text', 'label'])
    
    def _load_video_dataset(self, video_path: Path) -> pd.DataFrame:
        """Load deepfake video dataset."""
        csv_files = list(video_path.glob('**/*.csv'))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in video dataset: {video_path}")
        
        data = pd.read_csv(csv_files[0], nrows=self.config['max_samples']//4)
        data['label'] = data.get('label', data.get('fake', 0)).astype(int)
        data['text'] = data.get('transcript', data.get('text', 'Video sample'))
        return data.dropna(subset=['text', 'label'])
    
    def _load_jailbreak_dataset(self, jailbreak_path: Path) -> pd.DataFrame:
        """Load jailbreak prompts dataset."""
        csv_files = list(jailbreak_path.glob('**/*.csv'))
        json_files = list(jailbreak_path.glob('**/*.jsonl'))
        
        if csv_files:
            data = pd.read_csv(csv_files[0], nrows=self.config['max_samples']//8)
        elif json_files:
            data = pd.read_json(json_files[0], lines=True, nrows=self.config['max_samples']//8)
        else:
            raise RuntimeError(f"No data files found in jailbreak dataset: {jailbreak_path}")
        
        data['label'] = 1  # All jailbreak attempts
        data['text'] = data.get('instruction', data.get('prompt', data.get('text', '')))
        return data.dropna(subset=['text'])
    
    def _load_clean_dataset(self, clean_path: Path) -> pd.DataFrame:
        """Load clean holdout dataset."""
        csv_files = list(clean_path.glob('**/*.csv'))
        json_files = list(clean_path.glob('**/*.json'))
        
        if csv_files:
            data = pd.read_csv(csv_files[0], nrows=self.config['max_samples']//8)
        elif json_files:
            data = pd.read_json(json_files[0], nrows=self.config['max_samples']//8)
        else:
            raise RuntimeError(f"No data files found in clean dataset: {clean_path}")
        
        data['label'] = 0  # All clean/real
        data['text'] = data.get('headline', data.get('description', data.get('text', '')))
        return data.dropna(subset=['text'])
    
    def _setup_deepfake_audio_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Setup deepfake audio dataset from real sources."""
        audio_path = dataset_path / "deepfake_audio"
        
        # Download real deepfake audio dataset from Kaggle
        audio_datasets = [
            'supritibhattacharya/deepfake-audio-detection',
            'birdy654/deep-voice-deepfake-voice-recognition',
        ]
        
        for dataset in audio_datasets:
            try:
                print(f"Downloading deepfake audio dataset: {dataset}")
                subprocess.run([
                    'kaggle', 'datasets', 'download', dataset,
                    '-p', str(audio_path)
                ], check=True)
                
                # Extract and load
                zip_files = list(audio_path.glob('*.zip'))
                if zip_files:
                    subprocess.run(['unzip', '-q', '-o', str(zip_files[0]), '-d', str(audio_path)], check=True)
                    zip_files[0].unlink()
                    
                    # Load the dataset
                    return self._load_audio_dataset(audio_path)
                    
            except subprocess.CalledProcessError:
                continue
        
        raise RuntimeError("No deepfake audio datasets could be downloaded. Real data required.")
    
    def _setup_deepfake_video_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Setup deepfake video dataset from real sources."""
        video_path = dataset_path / "deepfake_video"
        
        # Download real deepfake video dataset from Kaggle
        video_datasets = [
            'manjilkarki/deepfake-and-real-images',
            'xhlulu/140k-real-and-fake-faces',
        ]
        
        for dataset in video_datasets:
            try:
                print(f"Downloading deepfake video dataset: {dataset}")
                subprocess.run([
                    'kaggle', 'datasets', 'download', dataset,
                    '-p', str(video_path)
                ], check=True)
                
                # Extract and load
                zip_files = list(video_path.glob('*.zip'))
                if zip_files:
                    subprocess.run(['unzip', '-q', '-o', str(zip_files[0]), '-d', str(video_path)], check=True)
                    zip_files[0].unlink()
                    
                    # Load the dataset
                    return self._load_video_dataset(video_path)
                    
            except subprocess.CalledProcessError:
                continue
        
        raise RuntimeError("No deepfake video datasets could be downloaded. Real data required.")
    
    def _setup_jailbreak_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Setup jailbreak prompts dataset from real sources."""
        jailbreak_path = dataset_path / "jailbreak_prompts"
        
        # Download real jailbreak/safety datasets from Kaggle
        jailbreak_datasets = [
            'deepfake/harmful-instructions',
            'lmsys/toxic-chat',
        ]
        
        for dataset in jailbreak_datasets:
            try:
                print(f"Downloading jailbreak dataset: {dataset}")
                subprocess.run([
                    'kaggle', 'datasets', 'download', dataset,
                    '-p', str(jailbreak_path)
                ], check=True)
                
                # Extract and load
                zip_files = list(jailbreak_path.glob('*.zip'))
                if zip_files:
                    subprocess.run(['unzip', '-q', '-o', str(zip_files[0]), '-d', str(jailbreak_path)], check=True)
                    zip_files[0].unlink()
                    
                    # Load the dataset
                    return self._load_jailbreak_dataset(jailbreak_path)
                    
            except subprocess.CalledProcessError:
                continue
        
        raise RuntimeError("No jailbreak datasets could be downloaded. Real safety data required.")
    
    def _setup_clean_holdout_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Setup clean holdout dataset from real news sources."""
        clean_path = dataset_path / "clean_holdout"
        
        # Download real clean news datasets from Kaggle
        clean_datasets = [
            'rmisra/news-category-dataset',
            'snapcrack/all-the-news',
        ]
        
        for dataset in clean_datasets:
            try:
                print(f"Downloading clean news dataset: {dataset}")
                subprocess.run([
                    'kaggle', 'datasets', 'download', dataset,
                    '-p', str(clean_path)
                ], check=True)
                
                # Extract and load
                zip_files = list(clean_path.glob('*.zip'))
                if zip_files:
                    subprocess.run(['unzip', '-q', '-o', str(zip_files[0]), '-d', str(clean_path)], check=True)
                    zip_files[0].unlink()
                    
                    # Load the dataset
                    return self._load_clean_dataset(clean_path)
                    
            except subprocess.CalledProcessError:
                continue
        
        raise RuntimeError("No clean news datasets could be downloaded. Real holdout data required.")
    


class TinyLlamaModelRunner(ModelRunner):
    """Model runner for TinyLlama with Unsloth optimization - Infrastructure Category."""
    
    def __init__(self, lora_cfg: LoRAConfig, config: Dict[str, Any], genome: Genome = None):
        self.lora_cfg = lora_cfg
        self.config = config
        self.genome = genome
        self.run_id = config.get('run_id')  # Store run_id from config
        self._model = None
        self._tokenizer = None
        self._adapter_path = None
        self._validate_config()
    
    def _validate_config(self):
        """Configuration validation with immediate failure on errors."""
        required_fields = ['model_name', 'max_seq_length']
        model_config = self.config.get('model', {})
        for field in required_fields:
            if field not in model_config:
                raise ValueError(f"'{field}' missing from model config")
    
    def _setup_model(self):
        """Setup tinyllama model with Hugging Face Transformers for Mac MPS support."""
        if self._model is not None:
            return
        
        # Import required libraries
        import torch
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        model_config = self.config['model']
        model_name = model_config['model_name']
        max_seq_length = model_config['max_seq_length']
        
        # Get HuggingFace token from environment
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            print(f"Using HuggingFace token for authentication")
        
        print(f"Loading model: {model_name}")
        
        # Detect device - use MPS for Mac, CUDA for GPU, CPU otherwise
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"Using Apple MPS (Metal Performance Shaders) backend")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA GPU")
        else:
            device = "cpu"
            print(f"Using CPU")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Configure model loading based on device
        if device == "mps":
            # For MPS, use float16 instead of bfloat16 (not supported on MPS)
            # Don't use device_map="auto" as it can cause issues with MPS
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=hf_token,
                low_cpu_mem_usage=True
            )
            # Manually move to MPS after loading
            self._model = self._model.to(device)
        elif device == "cuda":
            # For CUDA, can use 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        else:
            # For CPU, use float32
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=hf_token
            )
        
        # Don't move model when using device_map="auto"
        # The model is already properly distributed across devices
        
        # Setup and train LoRA adapter
        self._setup_adapter()
        
        # Set model to evaluation mode by default
        self._model.eval()
        
        print(f"Model loaded on {device}")
        print(f"   LoRA rank: {self.lora_cfg.r}")
        print(f"   Target modules: {self.lora_cfg.target_modules}")
        if hasattr(self._model, 'num_parameters'):
            print(f"   Model parameters: {self._model.num_parameters():,}")
        print(f"   Adapter path: {self._adapter_path or 'none (applying LoRA on-the-fly)'}")
    
    def generate(self, prompt: str, max_tokens: int = 256, cheap_knobs: CheapKnobs = None) -> str:
        """Generate text using tinyllama model with CA-derived cheap knobs."""
        self._setup_model()
        
        import torch
        
        # Apply cheap knobs (CA-derived parameters)
        generation_params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True,
        }
        
        if cheap_knobs:
            generation_params.update(cheap_knobs_to_generation_kwargs(cheap_knobs))
        
        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_seq_length'] - max_tokens,
            padding=True
        )
        
        # Move to appropriate device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with model
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
                **generation_params
            )
        
        # Decode output
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        input_text = self._tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if generated_text.startswith(input_text):
            return generated_text[len(input_text):].strip()
        else:
            return generated_text.strip()
    
    
    
    def _setup_adapter(self):
        """Setup LoRA adapter using CoralX cache system."""
        if self._adapter_path:
            return  # Already have adapter
        
        # Use CoralX cache system for adapter management
        cache_config = CacheConfig(
            artifacts_dir=self.config.get('cache_dir', './cache'),
            base_checkpoint=self.config['model']['model_name'],
            cache_metadata=True,
            run_id=self.run_id or (self.genome.id if self.genome else None)
        )
        
        heavy_genes = HeavyGenes(
            rank=self.lora_cfg.r,
            alpha=self.lora_cfg.alpha,
            dropout=self.lora_cfg.dropout,
            target_modules=self.lora_cfg.target_modules
        )
        
        # Train or retrieve adapter - create wrapper function to match expected signature
        def training_wrapper(heavy_genes: HeavyGenes, base_checkpoint: str) -> str:
            """Wrapper function to match cache trainer signature."""
            # The save path will be determined by the cache system
            temp_save_path = f"./temp_adapter_{heavy_genes.to_hash()}"
            return self._train_adapter(temp_save_path)
        
        adapter_path = get_or_train_adapter(
            heavy_genes=heavy_genes,
            cache_config=cache_config,
            trainer_fn=training_wrapper
        )
        
        self._adapter_path = adapter_path
        print(f"Adapter ready: {adapter_path}")
    
    def _train_adapter(self, save_path: str) -> str:
        """Train LoRA adapter using PEFT."""
        return self._real_training(save_path)
    
    
    def _real_training(self, save_path: str) -> str:
        """Perform real LoRA training using Hugging Face Trainer."""
        print(f"Real LoRA training for genome {self.genome.id if self.genome else 'unknown'}...")
        
        # Get training data from dataset
        dataset_provider = MultiModalAISafetyDatasetProvider(self.config['dataset'])
        training_data = list(dataset_provider.problems())
        
        # Limit training data for efficiency
        training_config = self.config.get('training', {})
        max_train_samples = training_config.get('max_train_samples', 50)
        training_data = training_data[:max_train_samples]
        
        # Prepare dataset for training
        formatted_dataset = self._prepare_training_dataset(training_data)
        
        # Import required libraries
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        import torch
        
        # Setup model for training
        self._setup_model()
        
        # Apply LoRA configuration to make only LoRA parameters trainable
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.lora_cfg.r,
            lora_alpha=self.lora_cfg.alpha,
            target_modules=list(self.lora_cfg.target_modules),
            lora_dropout=self.lora_cfg.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model - this makes only LoRA params trainable
        self._model = get_peft_model(self._model, lora_config)
        self._model.train()
        
        # Detect device
        if torch.backends.mps.is_available():
            device = "mps"
            use_fp16 = False  # MPS doesn't support fp16 well
            use_bf16 = False
        elif torch.cuda.is_available():
            device = "cuda"
            use_fp16 = True
            use_bf16 = False
        else:
            device = "cpu"
            use_fp16 = False
            use_bf16 = False
        
        # Create data collator with padding
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # For efficiency
        )
        
        # Training arguments
        print(f"Training setup - Device: {device}, FP16: {use_fp16}, BF16: {use_bf16}")
        print(f"Optimizer: {'adamw_torch' if device == 'mps' else 'adamw_8bit'}")
        print(f"Max grad norm: {training_config.get('max_grad_norm', 1.0)}")
        
        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=training_config.get('epochs', 1),
            per_device_train_batch_size=training_config.get('batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
            warmup_steps=training_config.get('warmup_steps', 5),
            max_steps=training_config.get('max_steps', 50),
            learning_rate=training_config.get('learning_rate', 1e-4),  # More conservative LR
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=5,
            optim="adamw_torch" if device == "mps" else "adamw_8bit",
            weight_decay=training_config.get('weight_decay', 0.01),
            lr_scheduler_type="linear",
            seed=42,
            save_steps=500,
            save_total_limit=1,
            push_to_hub=False,
            report_to=[],  # Disable wandb/tensorboard
            gradient_checkpointing=False,  # Disable for stability
            dataloader_pin_memory=False if device == "mps" else True,
            max_grad_norm=training_config.get('max_grad_norm', 1.0),  # Fix NaN gradients
            remove_unused_columns=False,  # Keep all columns to avoid signature mismatch
        )
        
        # Debug: Check model state before training
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable params: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        # Create trainer
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=formatted_dataset,
            processing_class=self._tokenizer,  # Use processing_class instead of deprecated tokenizer
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save adapter
        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)
        
        print(f"Real training complete: {save_path}")
        return save_path
    
    def _estimate_training_performance(self) -> Dict[str, float]:
        """Estimate training performance based on LoRA configuration - Pure function."""
        # Performance estimation based on LoRA hyperparameters
        # This is a heuristic approximation for evolutionary efficiency
        
        base_auroc = 0.75  # Base performance
        base_safety = 0.80  # Base safety score
        
        # Rank influence: higher rank = more capacity but potential overfitting
        rank_factor = min(1.0, self.lora_cfg.r / 16.0)  # Normalize to 16 as good rank
        auroc_boost = rank_factor * 0.1 - (rank_factor - 0.5) ** 2 * 0.05  # Inverted U curve
        
        # Alpha influence: scaling factor for LoRA
        alpha_factor = min(1.0, self.lora_cfg.alpha / 32.0)  # Normalize to 32 as good alpha
        alpha_boost = alpha_factor * 0.08
        
        # Dropout influence: regularization effect
        dropout_factor = 1.0 - self.lora_cfg.dropout  # Less dropout = potentially higher performance
        dropout_boost = dropout_factor * 0.05
        
        # Target modules influence: more modules = more adaptation
        target_modules_factor = len(self.lora_cfg.target_modules) / 4.0  # Normalize to 4 modules
        modules_boost = min(target_modules_factor, 1.0) * 0.06
        
        # Add some controlled randomness for evolutionary diversity
        import random
        random.seed(hash(self.genome.id) if self.genome else 42)
        noise = random.uniform(-0.02, 0.02)
        
        estimated_auroc = base_auroc + auroc_boost + alpha_boost + dropout_boost + modules_boost + noise
        estimated_safety = base_safety + (auroc_boost + alpha_boost) * 0.5 + noise * 0.5
        
        # Clamp to reasonable ranges
        estimated_auroc = max(0.6, min(0.95, estimated_auroc))
        estimated_safety = max(0.7, min(0.98, estimated_safety))
        
        return {
            'estimated_auroc': estimated_auroc,
            'estimated_safety': estimated_safety,
            'rank_factor': rank_factor,
            'alpha_factor': alpha_factor,
            'dropout_factor': dropout_factor,
            'modules_factor': target_modules_factor
        }
    
    def _prepare_training_dataset(self, training_data: List[Dict[str, Any]]):
        """Prepare tokenized dataset for LoRA training."""
        from datasets import Dataset
        
        # Convert to instruction-following format
        texts = []
        for item in training_data:
            text = item['text']
            label = "fake" if item['label'] == 1 else "real"
            
            # Create instruction-following format
            instruction = f"Classify the following news text as 'fake' or 'real':\n\n{text}\n\nClassification:"
            full_text = f"{instruction} {label}"
            texts.append(full_text)
        
        # Tokenize the texts one by one to avoid batch tensor issues
        tokenized_examples = []
        for text in texts:
            # Tokenize individual text
            tokenized = self._tokenizer(
                text, 
                truncation=True, 
                padding="max_length",
                max_length=self.config['model']['max_seq_length'],
                return_tensors=None  # Return plain lists, not tensors
            )
            # For causal LM, labels should be the input_ids (Trainer handles the shifting)
            # Set labels to input_ids but mask padding tokens with -100
            labels = tokenized["input_ids"].copy()
            # Mask padding tokens in labels (they shouldn't contribute to loss)
            labels = [label if mask == 1 else -100 for label, mask in zip(labels, tokenized["attention_mask"])]
            
            # Debug: Check how many tokens are not masked
            non_masked_count = sum(1 for label in labels if label != -100)
            if len(tokenized_examples) == 0:  # Only print for first sample
                print(f"Text length: {len(text)}, Tokenized length: {len(tokenized['input_ids'])}, Non-masked tokens: {non_masked_count}")
                if non_masked_count == 0:
                    print(f"WARNING: All tokens are masked! This will cause training issues.")
                    print(f"   First few input_ids: {tokenized['input_ids'][:10]}")
                    print(f"   First few attention_mask: {tokenized['attention_mask'][:10]}")
            
            example = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels
            }
            tokenized_examples.append(example)
        
        # Create dataset from tokenized examples
        tokenized_dataset = Dataset.from_list(tokenized_examples)
        
        return tokenized_dataset


class MultiModalAISafetyFitness(FitnessFn):
    """Multi-objective fitness function for AI safety across modalities - Application Category."""
    
    def __init__(self, config):
        # Handle both CoralConfig objects and dictionaries  
        if hasattr(config, 'dict'):
            # CoralConfig object - convert to dictionary for compatibility
            self.config = {
                'test_samples': config.evaluation.test_samples,
                'datasets': config.experiment.dataset.datasets,
                'dataset_path': config.experiment.dataset.path,
                'max_samples': getattr(config.experiment.dataset, 'max_samples', 100),
                'model': config.experiment.model.dict(),
                'training': config.training.dict() if hasattr(config, 'training') else {},
                'cache_dir': config.cache.artifacts_dir if hasattr(config, 'cache') else './cache'
            }
        else:
            # Dictionary config (backward compatibility)
            self.config = config
        
        self._validate_config()
        self._memory_tracker = None
    
    def _validate_config(self):
        """Configuration validation with immediate failure on errors."""
        required_fields = ['test_samples']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"'{field}' missing from fitness config")
    
    def __call__(self, genome: 'Genome', model: ModelRunner, problems: Iterable[Dict[str, Any]]) -> float:
        """Main fitness evaluation entry point - FitnessFn protocol implementation."""
        # Evaluate multi-objective metrics and return overall score
        multi_scores = self.evaluate_multi_objective(genome, model, problems)
        
        # Update genome with multi-objective scores
        genome_with_scores = genome.with_multi_scores(multi_scores)
        
        # Return overall fitness score (weighted combination of objectives)
        return multi_scores.overall_fitness()
    
    def evaluate_multi_objective(self, genome: Genome, model: ModelRunner, problems: Iterable[Dict[str, Any]]) -> MultiObjectiveScores:
        """Evaluate genome across all AI safety objectives - Pure functional composition."""
        
        # Convert problems to list and organize by dataset type
        problem_list = list(problems)
        test_samples = min(len(problem_list), self.config['test_samples'])
        test_problems = problem_list[:test_samples]
        
        print(f"Multi-objective evaluation: genome {genome.id} on {test_samples} problems")
        
        # Group problems by dataset type for specialized evaluation
        problems_by_type = {}
        for problem in test_problems:
            dataset_type = problem.get('dataset_type', 'unknown')
            if dataset_type not in problems_by_type:
                problems_by_type[dataset_type] = []
            problems_by_type[dataset_type].append(problem)
        
        # Initialize memory tracking
        self._start_memory_tracking()
        
        # P1: Detection Accuracy - Overall fake vs real classification accuracy
        detection_accuracy = self._evaluate_detection_accuracy(model, problems_by_type, genome)
        
        # P2: Fake News Recall - Sensitivity to catch dangerous misinformation
        fake_news_recall = self._evaluate_fake_news_recall(model, problems_by_type, genome)
        
        # P3: Real News Precision - Don't suppress legitimate journalism
        real_news_precision = self._evaluate_real_news_precision(model, problems_by_type, genome)
        
        # P4: Cross-Source Robustness - Performance consistency across sources
        cross_source_robustness = self._evaluate_cross_source_robustness(model, problems_by_type, genome)
        
        # P5: Confidence Calibration - Reliability of confidence scores
        confidence_calibration = self._evaluate_confidence_calibration(model, problems_by_type, genome)
        
        # P6: Efficiency - Inference speed for real-time deployment
        efficiency_score = self._evaluate_efficiency(model, problems_by_type, genome)
        
        # Create fake news detection metrics
        metrics = FakeNewsDetectionMetrics(
            detection_accuracy=detection_accuracy,
            fake_news_recall=fake_news_recall,
            real_news_precision=real_news_precision,
            cross_source_robustness=cross_source_robustness,
            confidence_calibration=confidence_calibration,
            efficiency_score=efficiency_score
        )
        
        print(f"Fake News Detection Results for {genome.id}:")
        print(f"   P1 Detection Accuracy: {detection_accuracy:.3f}")
        print(f"   P2 Fake News Recall: {fake_news_recall:.3f}")
        print(f"   P3 Real News Precision: {real_news_precision:.3f}")
        print(f"   P4 Cross-Source Robustness: {cross_source_robustness:.3f}")
        print(f"   P5 Confidence Calibration: {confidence_calibration:.3f}")
        print(f"   P6 Efficiency Score: {efficiency_score:.3f}")
        
        # Map to CoralX multi-objective scores
        return self._map_to_multiobjective_scores(metrics, test_problems)
    
    def _create_classification_prompt(self, text: str) -> str:
        """Create classification prompt for fake news detection."""
        return f"""<|system|>
You are a news classification system. Analyze the following news text and determine if it is real news or fake news.

<|user|>
Text: {text[:1500]}

Is this real news or fake news? Respond with only "real" or "fake".

<|assistant|>
"""
    
    def _parse_prediction(self, response: str) -> tuple[int, float]:
        """Parse model response to get prediction and confidence - Pure function."""
        response = response.lower().strip()
        
        # Simple parsing - look for key words
        if 'fake' in response:
            return 1, 0.8  # Fake news
        elif 'real' in response:
            return 0, 0.8  # Real news
        else:
            return 0, 0.5  # Default to real with low confidence
    
    def _start_memory_tracking(self):
        """Initialize memory tracking for P4 evaluation."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            else:
                import psutil
                self._initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
        except ImportError:
            self._initial_memory = 0.0
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage during evaluation - P4 metric."""
        try:
            import torch
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                return peak_memory
            else:
                import psutil
                current_memory = psutil.virtual_memory().used / (1024**3)  # GB
                return current_memory - self._initial_memory
        except ImportError:
            return 1.0  # Conservative estimate if can't measure
    
    
    
    
    def _evaluate_detection_accuracy(self, model: ModelRunner, problems_by_type: Dict[str, List], genome: Genome) -> float:
        """P1: Overall fake vs real news classification accuracy."""
        from sklearn.metrics import accuracy_score
        
        fake_news_problems = problems_by_type.get('fake_news', [])
        if len(fake_news_problems) < 2:
            return 0.0  # Not enough data
        
        predictions = []
        ground_truth = []
        confidences = []
        
        print(f"   Evaluating detection accuracy on {len(fake_news_problems)} samples")
        
        for problem in fake_news_problems:
            # Use proper classification prompt
            prompt = self._create_classification_prompt(problem['text'])
            response = model.generate(prompt, max_tokens=5, cheap_knobs=self._get_cheap_knobs(genome))
            
            pred_label, confidence = self._parse_prediction(response)
            predictions.append(pred_label)
            ground_truth.append(problem['label'])
            confidences.append(confidence)
            
            # Debug output for first few samples
            if len(predictions) <= 3:
                actual_label = "fake" if problem['label'] == 1 else "real"
                pred_label_str = "fake" if pred_label == 1 else "real"
                print(f"     Sample {len(predictions)}: '{problem['text'][:50]}...'")
                print(f"     Actual: {actual_label}, Predicted: {pred_label_str} (conf: {confidence:.2f})")
        
        # Calculate accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        print(f"   Detection Accuracy: {accuracy:.3f}")
        return accuracy
    
    def _evaluate_fake_news_recall(self, model: ModelRunner, problems_by_type: Dict[str, List], genome: Genome) -> float:
        """P2: Sensitivity - ability to catch fake news (don't miss misinformation)."""
        
        fake_news_problems = problems_by_type.get('fake_news', [])
        fake_samples = [p for p in fake_news_problems if p['label'] == 1]
        
        if not fake_samples:
            return 0.0
        
        correct_fake_detections = 0
        
        print(f"   Evaluating fake news recall on {len(fake_samples)} fake samples")
        
        for problem in fake_samples:
            prompt = self._create_classification_prompt(problem['text'])
            response = model.generate(prompt, max_tokens=5, cheap_knobs=self._get_cheap_knobs(genome))
            pred_label, _ = self._parse_prediction(response)
            
            if pred_label == 1:  # Correctly identified as fake
                correct_fake_detections += 1
        
        recall = correct_fake_detections / len(fake_samples)
        print(f"   Fake News Recall: {recall:.3f} ({correct_fake_detections}/{len(fake_samples)})")
        return recall
    
    def _evaluate_real_news_precision(self, model: ModelRunner, problems_by_type: Dict[str, List], genome: Genome) -> float:
        """P3: Specificity - don't suppress legitimate journalism (precision for real news)."""
        
        fake_news_problems = problems_by_type.get('fake_news', [])
        real_samples = [p for p in fake_news_problems if p['label'] == 0]
        
        if not real_samples:
            return 0.0
        
        correct_real_classifications = 0
        
        print(f"   Evaluating real news precision on {len(real_samples)} real samples")
        
        for problem in real_samples:
            prompt = self._create_classification_prompt(problem['text'])
            response = model.generate(prompt, max_tokens=5, cheap_knobs=self._get_cheap_knobs(genome))
            pred_label, _ = self._parse_prediction(response)
            
            if pred_label == 0:  # Correctly identified as real
                correct_real_classifications += 1
        
        precision = correct_real_classifications / len(real_samples)
        print(f"   Real News Precision: {precision:.3f} ({correct_real_classifications}/{len(real_samples)})")
        return precision
    
    def _evaluate_cross_source_robustness(self, model: ModelRunner, problems_by_type: Dict[str, List], genome: Genome) -> float:
        """P4: Performance consistency across different news sources (BuzzFeed vs PolitiFact)."""
        from sklearn.metrics import accuracy_score
        import numpy as np
        
        fake_news_problems = problems_by_type.get('fake_news', [])
        if len(fake_news_problems) < 4:
            return 0.5  # Not enough data
        
        # Split by source if metadata available, otherwise use random split for robustness test
        source_accuracies = []
        
        # Split into two halves to simulate different sources
        mid_point = len(fake_news_problems) // 2
        source_splits = [
            fake_news_problems[:mid_point],
            fake_news_problems[mid_point:]
        ]
        
        print(f"   Evaluating cross-source robustness on {len(source_splits)} source splits")
        
        for i, source_samples in enumerate(source_splits):
            if not source_samples:
                continue
                
            predictions = []
            ground_truth = []
            
            for problem in source_samples:
                prompt = self._create_classification_prompt(problem['text'])
                response = model.generate(prompt, max_tokens=5, cheap_knobs=self._get_cheap_knobs(genome))
                pred_label, _ = self._parse_prediction(response)
                
                predictions.append(pred_label)
                ground_truth.append(problem['label'])
            
            if ground_truth:
                accuracy = accuracy_score(ground_truth, predictions)
                source_accuracies.append(accuracy)
                print(f"     Source {i+1}: {accuracy:.3f} accuracy")
        
        # Robustness = 1 - variance in performance across sources
        if len(source_accuracies) >= 2:
            variance = np.var(source_accuracies)
            robustness = max(0.0, 1.0 - variance * 4)  # Scale variance impact
        else:
            robustness = 0.5
            
        print(f"   Cross-Source Robustness: {robustness:.3f}")
        return robustness
    
    def _evaluate_confidence_calibration(self, model: ModelRunner, problems_by_type: Dict[str, List], genome: Genome) -> float:
        """P5: Reliability of confidence scores for fake news detection."""
        
        fake_news_problems = problems_by_type.get('fake_news', [])
        if len(fake_news_problems) < 4:
            return 0.5  # Not enough data for calibration
        
        predictions = []
        confidences = []
        ground_truth = []
        
        print(f"   Evaluating confidence calibration on {len(fake_news_problems)} samples")
        
        for problem in fake_news_problems:
            prompt = self._create_classification_prompt(problem['text'])
            response = model.generate(prompt, max_tokens=5, cheap_knobs=self._get_cheap_knobs(genome))
            
            pred_label, confidence = self._parse_prediction(response)
            predictions.append(pred_label)
            confidences.append(confidence)
            ground_truth.append(problem['label'])
        
        # Simple calibration: how often high-confidence predictions are correct
        high_conf_correct = 0
        high_conf_total = 0
        
        for pred, conf, truth in zip(predictions, confidences, ground_truth):
            if conf > 0.7:  # High confidence threshold
                high_conf_total += 1
                if pred == truth:
                    high_conf_correct += 1
        
        calibration = high_conf_correct / high_conf_total if high_conf_total > 0 else 0.5
        print(f"   Confidence Calibration: {calibration:.3f} ({high_conf_correct}/{high_conf_total} high-conf correct)")
        return calibration
    
    def _evaluate_efficiency(self, model: ModelRunner, problems_by_type: Dict[str, List], genome: Genome) -> float:
        """P6: Inference speed for real-time deployment."""
        import time
        
        fake_news_problems = problems_by_type.get('fake_news', [])
        if not fake_news_problems:
            return 0.5
        
        # Test on small sample for speed
        test_samples = fake_news_problems[:3]
        
        print(f"   Evaluating inference efficiency on {len(test_samples)} samples")
        
        start_time = time.time()
        
        for problem in test_samples:
            prompt = self._create_classification_prompt(problem['text'])
            model.generate(prompt, max_tokens=5, cheap_knobs=self._get_cheap_knobs(genome))
        
        total_time = time.time() - start_time
        avg_time_per_inference = total_time / len(test_samples)
        
        # Efficiency score: 1.0 for <1s, decreasing as time increases
        efficiency = max(0.0, 1.0 - avg_time_per_inference / 10.0)  # 10s = 0 efficiency
        
        print(f"   Efficiency Score: {efficiency:.3f} ({avg_time_per_inference:.2f}s avg per inference)")
        return efficiency
    
    def _get_cheap_knobs(self, genome: Genome):
        """Extract CA-derived cheap knobs for generation."""
        try:
            from core.domain.cheap_knobs import extract_cheap_knobs_from_ca_features
            return extract_cheap_knobs_from_ca_features(genome.ca_features) if genome.ca_features else None
        except ImportError:
            return None
    
    def _create_detection_prompt(self, problem: Dict[str, Any], dataset_type: str) -> str:
        """Create appropriate detection prompt based on dataset type."""
        text = problem['text']
        
        if dataset_type == 'fake_news':
            return f"Analyze if this news is real or fake:\n\n{text}\n\nClassification:"
        elif dataset_type == 'deepfake_audio':
            return f"Analyze if this audio transcript is from real or synthetic speech:\n\n{text}\n\nClassification:"
        elif dataset_type == 'deepfake_video':
            return f"Analyze if this video transcript is from real or deepfake video:\n\n{text}\n\nClassification:"
        else:
            return f"Analyze if this content is authentic or synthetic:\n\n{text}\n\nClassification:"
    
    def _parse_detection_prediction(self, response: str) -> tuple[int, float]:
        """Parse detection response to get prediction and confidence."""
        response = response.lower().strip()
        
        if any(word in response for word in ['fake', 'synthetic', 'deepfake', 'false']):
            return 1, 0.8  # Fake/synthetic
        elif any(word in response for word in ['real', 'authentic', 'genuine', 'true']):
            return 0, 0.8  # Real/authentic
        else:
            return 0, 0.5  # Default to real with low confidence
    
    def _is_proper_refusal(self, response: str, problem: Dict[str, Any]) -> bool:
        """Check if model properly refused a jailbreak attempt."""
        response = response.lower()
        
        # Look for refusal indicators
        refusal_indicators = [
            "i can't", "i cannot", "i'm not able", "i shouldn't", "i won't",
            "not appropriate", "against my guidelines", "unsafe", "harmful",
            "i'm designed to", "my purpose is to help", "i don't provide"
        ]
        
        return any(indicator in response for indicator in refusal_indicators)
    
    def _simulate_text_only_performance(self, problems: List) -> float:
        """Simulate text-only model performance (placeholder)."""
        # In production: run inference with text-only model
        return 0.7 + np.random.normal(0, 0.05)  # Simulated baseline
    
    def _simulate_multimodal_performance(self, problems: List) -> float:
        """Simulate multimodal model performance (placeholder)."""
        # In production: run inference with full multimodal model
        return 0.75 + np.random.normal(0, 0.05)  # Simulated improvement
    
    def _calculate_expected_calibration_error(self, predictions: List, confidences: List, ground_truth: List) -> float:
        """Calculate Expected Calibration Error."""
        import numpy as np
        
        # Simplified ECE calculation
        n_bins = 5
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(predictions)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = [(c > bin_lower) and (c <= bin_upper) for c in confidences]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                bin_predictions = [predictions[i] for i, in_b in enumerate(in_bin) if in_b]
                bin_ground_truth = [ground_truth[i] for i, in_b in enumerate(in_bin) if in_b]
                bin_confidences = [confidences[i] for i, in_b in enumerate(in_bin) if in_b]
                
                if bin_predictions:
                    accuracy_in_bin = np.mean([p == gt for p, gt in zip(bin_predictions, bin_ground_truth)])
                    avg_confidence_in_bin = np.mean(bin_confidences)
                    
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


    
    def _map_to_multiobjective_scores(self, metrics: FakeNewsDetectionMetrics, test_problems: List[Dict[str, Any]]) -> MultiObjectiveScores:
        """Map fake news detection metrics to CoralX objectives."""
        
        # For fake news detection, map to CoralX objectives:
        # bugfix -> Detection Accuracy (overall performance)
        # style -> Real News Precision (don't suppress legitimate news)  
        # security -> Fake News Recall (catch dangerous misinformation)
        # runtime -> Efficiency (speed for real-time deployment)
        # syntax -> Cross-Source Robustness + Calibration (reliability)
        
        return MultiObjectiveScores(
            bugfix=metrics.detection_accuracy,         # Overall detection accuracy
            style=metrics.real_news_precision,         # Don't suppress real news
            security=metrics.fake_news_recall,         # Catch fake news
            runtime=metrics.efficiency_score,          # Inference efficiency
            syntax=(metrics.cross_source_robustness + metrics.confidence_calibration) / 2  # Reliability
        )


class MultiModalAISafetyPlugin:
    """Main plugin class for Multi-Modal AI Safety + TinyLlama integration - Application Category."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_id = config.get('run_id')  # Store run_id from config
        self._validate_config()
        
        print(f"Multi-Modal AI Safety + TinyLlama plugin initialized")
        dataset_path = config.get('dataset', {}).get('path') or config.get('dataset', {}).get('dataset_path') or 'not specified'
        print(f"   Dataset: {dataset_path}")
        print(f"   Model: {config.get('model', {}).get('model_name', 'not specified')}")
        print(f"   Objectives: Detection Accuracy, Fake News Recall, Real News Precision, Cross-Source Robustness, Confidence Calibration, Efficiency")
        print(f"   Run ID: {self.run_id}")
    
    def _validate_config(self):
        """Configuration validation with immediate failure on errors."""
        required_sections = ['dataset', 'model']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required config section '{section}' not found")
    
    
    def dataset(self) -> DatasetProvider:
        """Create multi-modal dataset provider from config."""
        return MultiModalAISafetyDatasetProvider(self.config['dataset'])
    
    def model_factory(self) -> Callable[[LoRAConfig], ModelRunner]:
        """Create model factory from config."""
        def create_model(lora_cfg: LoRAConfig, genome: Genome = None) -> ModelRunner:
            return TinyLlamaModelRunner(lora_cfg, self.config, genome=genome)
        return create_model
    
    def fitness_fn(self) -> FitnessFn:
        """Create multi-objective AI safety fitness function from config."""
        eval_config = self.config.get('evaluation', {})
        print(f"Config keys: {list(self.config.keys())}")
        print(f"Evaluation config: {eval_config}")
        return MultiModalAISafetyFitness(eval_config)
    
    def test_evolved_adapters(self, best_genome, config_evaluation=None):
        """Test the best evolved LoRA adapter with comprehensive baseline comparison."""
        print(f"\nCOMPREHENSIVE ADAPTER EVALUATION")
        print(f"   Genome ID: {best_genome.id}")
        print(f"   LoRA Config: r={best_genome.lora_cfg.r}, α={best_genome.lora_cfg.alpha}, dropout={best_genome.lora_cfg.dropout}")
        print(f"   Fitness: {best_genome.fitness:.4f}")
        
        # Check if baseline testing is enabled
        if config_evaluation and hasattr(config_evaluation, 'baseline_testing') and config_evaluation.baseline_testing.enabled:
            return self._comprehensive_baseline_comparison(best_genome, config_evaluation.baseline_testing)
        else:
            # Use simple testing
            return self._simple_adapter_test(best_genome)
    
    def _comprehensive_baseline_comparison(self, best_genome, baseline_config):
        """Run comprehensive baseline vs LoRA comparison."""
        from core.services.baseline_evaluator import BaselineEvaluator
        
        print(f"Running comprehensive baseline vs LoRA comparison...")
        
        # Find the adapter path for this genome
        adapter_path = self._get_adapter_path_for_genome(best_genome)
        if not adapter_path:
            print(f"No adapter found for genome {best_genome.id}")
            return self._simple_adapter_test(best_genome)
        
        # Load base model and tokenizer
        model_name = self.config['model']['model_name']
        print(f"Loading base model: {model_name}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
        )
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        base_model = base_model.to(device)
        base_model.eval()
        
        # Initialize baseline evaluator
        dataset_path = self.config['dataset']['dataset_path']
        evaluator = BaselineEvaluator(baseline_config, dataset_path)
        
        try:
            # Run comprehensive comparison
            comparison = evaluator.compare_baseline_vs_lora(base_model, tokenizer, adapter_path)
            
            # Print detailed results
            evaluator.print_detailed_results(comparison)
            
            # Return the LoRA accuracy for compatibility
            return comparison.lora.accuracy
            
        except Exception as e:
            print(f"Comprehensive evaluation failed: {e}")
            print(f"Falling back to simple test...")
            return self._simple_adapter_test(best_genome)
    
    def _get_adapter_path_for_genome(self, genome):
        """Find the cached adapter path for a specific genome."""
        from infra.adapter_cache import HeavyGenes, CacheConfig, get_adapter_cache
        
        try:
            # Create heavy genes from LoRA config
            heavy_genes = HeavyGenes.from_lora_config(genome.lora_cfg, run_id=self.run_id)
            
            # Get cache
            cache_config = CacheConfig(
                artifacts_dir=self.config.get('cache_dir', './cache'),
                base_checkpoint=self.config['model']['model_name'],
                cache_metadata=True
            )
            cache = get_adapter_cache(cache_config)
            
            # Check if adapter exists
            cache_hash = heavy_genes.to_hash()
            if cache.has_adapter(cache_hash):
                return cache.get_adapter_path(cache_hash)
        except Exception as e:
            print(f"Could not locate adapter for genome: {e}")
        
        return None
    
    def _simple_adapter_test(self, best_genome, test_samples: int = 10):
        """Simple adapter test."""
        print(f"Running simple adapter test...")
        
        # Create model with the best genome's configuration
        model_factory = self.model_factory()
        model_runner = model_factory(best_genome.lora_cfg, best_genome)
        
        # Get test data
        dataset_provider = self.dataset()
        all_problems = list(dataset_provider.problems())
        test_problems = all_problems[-test_samples:]  # Use last samples as test set
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, problem in enumerate(test_problems[:test_samples]):
            prompt = f"""<|system|>
You are a news classification system. Analyze the following news text and determine if it is real news or fake news.

<|user|>
Text: {problem['text'][:200]}...

Is this real news or fake news? Respond with only "real" or "fake".

<|assistant|>
"""
            
            response = model_runner.generate(prompt, max_tokens=5)
            predicted_fake = 1 if 'fake' in response.lower() else 0
            actual_fake = problem['label']
            
            is_correct = predicted_fake == actual_fake
            correct_predictions += is_correct
            total_predictions += 1
            
            if i < 3:  # Show first 3 examples
                status = "PASS" if is_correct else "FAIL"
                actual_label = "fake" if actual_fake else "real"
                pred_label = "fake" if predicted_fake else "real"
                print(f"   {status} Sample {i+1}: Actual={actual_label}, Predicted={pred_label}")
        
        accuracy = correct_predictions / total_predictions
        print(f"   Simple Test Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
        
        if accuracy > 0.6:
            print("   Good performance! The LoRA adapter is working!")
        elif accuracy > 0.4:
            print("   Moderate performance. The adapter shows some learning.")
        else:
            print("   Poor performance. The adapter may need more training or different hyperparameters.")
        
        return accuracy 