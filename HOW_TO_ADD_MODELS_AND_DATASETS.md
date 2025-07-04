# CoralX HOW-TO: Adding New Models and Datasets

## 🏗️ Architecture Overview

CoralX is a functional evolution system built on **Category Theory principles** with a **fail-fast** approach. The architecture is organized into categorical layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Category                             │
│                (Terminal Objects)                          │
├─────────────────────────────────────────────────────────────┤
│                Application Category                         │
│              (Business Logic Functors)                     │
├─────────────────────────────────────────────────────────────┤
│                  Domain Category                           │
│         (Pure Mathematical Objects & Morphisms)            │
├─────────────────────────────────────────────────────────────┤
│                 Ports Category                             │
│              (Abstract Interfaces)                         │
├─────────────────────────────────────────────────────────────┤
│         Infrastructure & Plugins Categories                │
│            (Effectful Functors)                           │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 Plugin System Overview

CoralX uses a **plugin-based architecture** where each experiment is a combination of:
- **Dataset Provider**: Implements `DatasetProvider` protocol
- **Model Runner**: Implements `ModelRunner` protocol  
- **Fitness Function**: Implements `FitnessFn` protocol
- **Plugin Factory**: Orchestrates the above components

## 📊 Adding a New Dataset

### Step 1: Create Dataset Provider Class

Create a new file `adapters/your_dataset_real.py`:

```python
"""
Real YourDataset Adapter for CORAL-X
Uses actual dataset with real evaluation and Modal integration
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable
from dataclasses import dataclass
from coral.ports.interfaces import DatasetProvider


@dataclass
class YourDatasetMetrics:
    """Multi-objective metrics for your dataset evaluation."""
    primary_metric: float      # Main performance metric
    secondary_metric: float    # Secondary performance metric
    efficiency: float          # Runtime efficiency
    quality: float            # Output quality


class YourDatasetRealAdapter:
    """Real dataset adapter - NO FALLBACKS, fail-fast."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Validate required config sections
        experiment_config = config.get('experiment', {})
        dataset_config = experiment_config.get('dataset', {})
        
        if not dataset_config:
            raise ValueError("FAIL-FAST: dataset config section missing")
        
        # Get dataset path from centralized paths
        paths = config.get('paths', {})
        executor = config.get('infra', {}).get('executor', 'local')
        current_paths = paths.get(executor, {})
        
        if 'dataset' not in current_paths:
            raise ValueError(f"FAIL-FAST: dataset path not configured for executor {executor}")
            
        self.dataset_path = Path(current_paths['dataset'])
        
        # Validate dataset exists
        if not self.dataset_path.exists():
            raise ValueError(f"FAIL-FAST: Dataset not found at {self.dataset_path}")
        
        # Load dataset problems
        self.problems = self._load_problems()
        
        # Define training/evaluation split to prevent data leakage
        self.training_problems = set(dataset_config.get('training_problems', []))
        
        print(f"📁 YourDataset loaded: {len(self.problems)} total problems")
        print(f"🎯 Training problems: {len(self.training_problems)} excluded from evaluation")
    
    def _load_problems(self) -> List[Dict[str, Any]]:
        """Load problems from dataset - fail-fast on errors."""
        problems = []
        
        # Example: Load from JSON files
        for problem_file in self.dataset_path.glob("*.json"):
            try:
                with open(problem_file, 'r') as f:
                    problem_data = json.load(f)
                    
                # Validate required fields
                required_fields = ['name', 'prompt', 'expected_output']
                for field in required_fields:
                    if field not in problem_data:
                        raise ValueError(f"FAIL-FAST: Required field '{field}' missing from {problem_file}")
                
                problems.append(problem_data)
                
            except Exception as e:
                raise ValueError(f"FAIL-FAST: Error loading {problem_file}: {e}")
        
        if not problems:
            raise ValueError(f"FAIL-FAST: No problems found in {self.dataset_path}")
        
        return problems
    
    def get_problems(self) -> List[Dict[str, Any]]:
        """Get all problems (excludes training problems)."""
        return [p for p in self.problems if p['name'] not in self.training_problems]
    
    def evaluate_solution(self, problem: Dict[str, Any], solution: str) -> YourDatasetMetrics:
        """Evaluate a solution against a problem - pure function."""
        
        # Your evaluation logic here
        # Example: Compare solution to expected output
        expected = problem['expected_output']
        
        # Implement your evaluation metrics
        primary_score = self._calculate_primary_metric(solution, expected)
        secondary_score = self._calculate_secondary_metric(solution, expected)
        efficiency_score = self._calculate_efficiency(solution)
        quality_score = self._calculate_quality(solution)
        
        return YourDatasetMetrics(
            primary_metric=primary_score,
            secondary_metric=secondary_score,
            efficiency=efficiency_score,
            quality=quality_score
        )
    
    def _calculate_primary_metric(self, solution: str, expected: str) -> float:
        """Calculate primary performance metric."""
        # Implement your primary metric calculation
        # Return value between 0.0 and 1.0
        pass
    
    def _calculate_secondary_metric(self, solution: str, expected: str) -> float:
        """Calculate secondary performance metric."""
        # Implement your secondary metric calculation
        # Return value between 0.0 and 1.0
        pass
    
    def _calculate_efficiency(self, solution: str) -> float:
        """Calculate efficiency metric."""
        # Implement efficiency calculation
        # Return value between 0.0 and 1.0
        pass
    
    def _calculate_quality(self, solution: str) -> float:
        """Calculate quality metric."""
        # Implement quality calculation
        # Return value between 0.0 and 1.0
        pass


class YourDatasetProvider(DatasetProvider):
    """Dataset provider implementing CoralX interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.adapter = YourDatasetRealAdapter(config)
    
    def problems(self) -> Iterable[Dict[str, Any]]:
        """Yield problem dictionaries with prompts and solutions."""
        yield from self.adapter.get_problems()
```

### Step 2: Create Evaluation Domain Function

Create or update `coral/domain/your_dataset_evaluation.py`:

```python
"""
Pure domain functions for YourDataset evaluation.
NO side effects, NO I/O operations.
"""
from typing import Dict, Any, List
from dataclasses import dataclass
from adapters.your_dataset_real import YourDatasetMetrics


@dataclass(frozen=True)
class YourDatasetEvaluationResults:
    """Immutable evaluation results."""
    primary_metric: float
    secondary_metric: float
    efficiency: float
    quality: float
    overall_score: float
    
    @classmethod
    def from_metrics(cls, metrics: YourDatasetMetrics) -> 'YourDatasetEvaluationResults':
        """Create results from metrics."""
        overall = (metrics.primary_metric + metrics.secondary_metric + 
                  metrics.efficiency + metrics.quality) / 4.0
        
        return cls(
            primary_metric=metrics.primary_metric,
            secondary_metric=metrics.secondary_metric,
            efficiency=metrics.efficiency,
            quality=metrics.quality,
            overall_score=overall
        )


def evaluate_your_dataset_code(generated_code: str, 
                              problem: Dict[str, Any], 
                              adapter) -> YourDatasetEvaluationResults:
    """
    Pure evaluation function for YourDataset.
    
    Args:
        generated_code: Generated solution code
        problem: Problem specification
        adapter: Dataset adapter for evaluation
        
    Returns:
        YourDatasetEvaluationResults: Immutable evaluation results
    """
    # Delegate to adapter for evaluation
    metrics = adapter.evaluate_solution(problem, generated_code)
    
    # Convert to domain results
    return YourDatasetEvaluationResults.from_metrics(metrics)
```

### Step 3: Update Dataset Constants

Add to `coral/domain/dataset_constants.py`:

```python
# Your dataset training problems (to prevent data leakage)
YOUR_DATASET_TRAINING_PROBLEMS = {
    'training_problem_1',
    'training_problem_2',
    'training_problem_3',
    # ... list all problems used in training
}
```

## 🤖 Adding a New Model

### Step 1: Create Model Runner Class

Create the model runner in your plugin file:

```python
class YourModelRunner(ModelRunner):
    """Model runner for your specific model."""
    
    def __init__(self, lora_cfg: LoRAConfig, config: Dict[str, Any], genome: Genome = None):
        self.lora_cfg = lora_cfg
        self.config = config
        self.genome = genome  # For cheap knobs access
        self._model_loaded = False
        
        # Validate required config
        model_config = config.get('experiment', {}).get('model', {})
        if not model_config:
            raise ValueError("FAIL-FAST: model config section missing")
        
        self.model_name = model_config.get('name')
        if not self.model_name:
            raise ValueError("FAIL-FAST: model name not specified")
        
        # Initialize model (lazy loading)
        self._setup_model()
    
    def _setup_model(self):
        """Setup model with LoRA adapter."""
        try:
            # Load base model
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Get cache paths
            paths = self.config.get('paths', {})
            executor = self.config.get('infra', {}).get('executor', 'local')
            current_paths = paths.get(executor, {})
            cache_dir = current_paths.get('models')
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Apply LoRA adapter
            self._apply_lora_adapter()
            
            self._model_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"FAIL-FAST: Model setup failed: {e}")
    
    def _apply_lora_adapter(self):
        """Apply LoRA adapter to model."""
        from peft import get_peft_model, LoraConfig
        
        # Convert CoralX LoRAConfig to PEFT LoraConfig
        peft_config = LoraConfig(
            r=self.lora_cfg.rank,
            lora_alpha=self.lora_cfg.alpha,
            lora_dropout=self.lora_cfg.dropout,
            target_modules=self.lora_cfg.target_modules,
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, peft_config)
    
    def generate(self, prompt: str, max_tokens: int = None, cheap_knobs=None) -> str:
        """Generate text using the model."""
        if not self._model_loaded:
            raise RuntimeError("FAIL-FAST: Model not loaded")
        
        # Get generation parameters
        gen_params = self._get_generation_params(max_tokens, cheap_knobs)
        
        # Check if using Modal execution
        infra_config = self.config.get('infra', {})
        executor_type = infra_config.get('executor')
        use_modal = executor_type in ['modal', 'queue_modal']
        
        if use_modal:
            return self._generate_modal(prompt, gen_params)
        else:
            return self._generate_local(prompt, gen_params)
    
    def _get_generation_params(self, max_tokens: int, cheap_knobs) -> Dict[str, Any]:
        """Get generation parameters from config and cheap knobs."""
        gen_config = self.config.get('generation', {})
        
        params = {
            'max_tokens': max_tokens or gen_config.get('max_tokens', 512),
            'temperature': gen_config.get('temperature', 0.7),
            'top_p': gen_config.get('top_p', 0.9),
            'top_k': gen_config.get('top_k', 50),
            'do_sample': gen_config.get('do_sample', True),
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        
        # Apply cheap knobs if available (CA-derived parameters)
        if cheap_knobs:
            params.update(cheap_knobs)
        
        return params
    
    def _generate_local(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate locally."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + params['max_tokens'],
                temperature=params['temperature'],
                top_p=params['top_p'],
                top_k=params['top_k'],
                do_sample=params['do_sample'],
                pad_token_id=params['pad_token_id']
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    
    def _generate_modal(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate using Modal."""
        # Implementation depends on your Modal setup
        # This would call your Modal function
        raise NotImplementedError("Modal generation not implemented yet")
```

### Step 2: Create Modal Service (if using Modal)

Create `infra/modal/your_model_service.py`:

```python
"""
Modal service for YourModel generation.
Clean separation between Modal infrastructure and business logic.
"""
from typing import Dict, Any


def your_model_generate_modal(prompt: str, model_config: Dict[str, Any], 
                             generation_params: Dict[str, Any]) -> str:
    """
    Modal service for YourModel generation.
    This function runs inside Modal environment.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    
    # Load model and tokenizer
    model_name = model_config['name']
    cache_dir = model_config.get('cache_dir', '/cache/models')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA if configured
    if 'lora_config' in model_config:
        lora_cfg = model_config['lora_config']
        peft_config = LoraConfig(
            r=lora_cfg['rank'],
            lora_alpha=lora_cfg['alpha'],
            lora_dropout=lora_cfg['dropout'],
            target_modules=lora_cfg['target_modules'],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + generation_params['max_tokens'],
            temperature=generation_params['temperature'],
            top_p=generation_params['top_p'],
            top_k=generation_params['top_k'],
            do_sample=generation_params['do_sample'],
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()
```

## 🔌 Creating a Complete Plugin

### Step 1: Create Plugin Directory Structure

```
plugins/
└── your_dataset_your_model/
    ├── __init__.py
    └── plugin.py
```

### Step 2: Implement Plugin Class

Create `plugins/your_dataset_your_model/plugin.py`:

```python
"""
YourDataset + YourModel Plugin for CORAL-X
Config-driven, NO FALLBACKS - fail-fast principle
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Dict, Any, Callable, List, Optional
from dataclasses import dataclass
import time

from coral.ports.interfaces import DatasetProvider, ModelRunner, FitnessFn
from coral.domain.mapping import LoRAConfig
from coral.domain.genome import Genome, MultiObjectiveScores
from coral.domain.your_dataset_evaluation import evaluate_your_dataset_code, YourDatasetEvaluationResults
from infra.adapter_cache import HeavyGenes, CacheConfig, get_or_train_adapter
from adapters.your_dataset_real import YourDatasetRealAdapter, YourDatasetProvider

# Your model runner (defined above)
from .model_runner import YourModelRunner


@dataclass
class CAFeatures:
    """CA features extracted from cellular automata evolution."""
    complexity: float
    intensity: float  
    periodicity: float
    convergence: float


class YourDatasetYourModelFitness(FitnessFn):
    """Fitness function for YourDataset + YourModel."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapter = YourDatasetRealAdapter(config)
        
        # Get fitness weights from config
        eval_config = config.get('evaluation', {})
        self.weights = eval_config.get('fitness_weights', {
            'primary_metric': 0.4,
            'secondary_metric': 0.3,
            'efficiency': 0.15,
            'quality': 0.15
        })
    
    def __call__(self, genome: Genome, model: ModelRunner, problems: Iterable[Dict[str, Any]]) -> float:
        """Evaluate fitness of genome."""
        
        scores = []
        for problem in problems:
            # Generate solution
            prompt = problem['prompt']
            generated_code = model.generate(prompt, max_tokens=300)
            
            # Evaluate solution
            result = evaluate_your_dataset_code(generated_code, problem, self.adapter)
            
            # Calculate weighted score
            weighted_score = (
                result.primary_metric * self.weights['primary_metric'] +
                result.secondary_metric * self.weights['secondary_metric'] +
                result.efficiency * self.weights['efficiency'] +
                result.quality * self.weights['quality']
            )
            
            scores.append(weighted_score)
        
        return np.mean(scores) if scores else 0.0


class YourDatasetYourModelPlugin:
    """Main plugin class - config-driven."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Validate required config sections
        if 'experiment' not in config:
            raise ValueError("FAIL-FAST: Required config section 'experiment' not found")
        
        experiment_config = config['experiment']
        required_sections = ['dataset', 'model']
        for section in required_sections:
            if section not in experiment_config:
                raise ValueError(f"FAIL-FAST: Required experiment section '{section}' not found")
        
        print(f"🔌 YourDataset + YourModel plugin initialized")
        print(f"   📁 Dataset: {experiment_config['dataset'].get('path', 'not specified')}")
        print(f"   🤖 Model: {experiment_config['model'].get('name', 'not specified')}")
    
    def get_modal_config(self, coral_config) -> Dict[str, Any]:
        """Get Modal-compatible configuration."""
        return {
            'evo': self.config['evo'],
            'execution': coral_config.execution,
            'experiment': coral_config.experiment,
            'infra': coral_config.infra,
            'cache': coral_config.cache,
            'threshold': {
                'base_thresholds': coral_config.threshold.base_thresholds.__dict__,
                'max_thresholds': coral_config.threshold.max_thresholds.__dict__,
                'schedule': coral_config.threshold.schedule
            },
            'evaluation': coral_config.evaluation,
            'seed': coral_config.seed,
            'adapter_type': getattr(coral_config, 'adapter_type', 'lora'),
            'paths': self.config.get('paths', {}),
            'cheap_knobs': self.config.get('cheap_knobs', {}),
            'training': self.config.get('training', {})
        }
    
    def dataset(self) -> DatasetProvider:
        """Create dataset provider from config."""
        return YourDatasetProvider(self.config)
    
    def model_factory(self) -> Callable[[LoRAConfig], ModelRunner]:
        """Create model factory from config."""
        def create_model(lora_cfg: LoRAConfig, genome: Genome = None) -> ModelRunner:
            return YourModelRunner(lora_cfg, self.config, genome=genome)
        return create_model
    
    def fitness_fn(self) -> FitnessFn:
        """Create fitness function from config."""
        return YourDatasetYourModelFitness(self.config)
```

### Step 3: Update CLI Integration

Add to `cli/coral.py` in the `_load_plugin` function:

```python
def _load_plugin(config):
    """Load the appropriate plugin based on configuration."""
    target = config.experiment.get('target')
    if not target:
        raise ValueError("FAIL-FAST: No plugin target specified in config")
    
    if target == 'quixbugs_codellama':
        from plugins.quixbugs_codellama.plugin import QuixBugsCodeLlamaRealPlugin
        return QuixBugsCodeLlamaRealPlugin(config.experiment)
    elif target == 'your_dataset_your_model':
        from plugins.your_dataset_your_model.plugin import YourDatasetYourModelPlugin
        return YourDatasetYourModelPlugin(config.experiment)
    else:
        raise ValueError(f"FAIL-FAST: Unknown plugin target: {target}")
```

## ⚙️ Creating Configuration Files

### Step 1: Create Base Configuration

Create `config/your_dataset_your_model_config.yaml`:

```yaml
# 📁 CENTRALIZED PATH CONFIGURATION
paths:
  modal:
    cache_root: "/cache"
    adapters: "/cache/adapters"
    models: "/cache/models"
    dataset: "/cache/your_dataset"
    progress: "/cache/evolution_progress.json"
    coralx_root: "/root/coralx"
  local:
    cache_root: "./coral_cache"
    adapters: "./coral_cache/adapters"
    models: "./coral_cache/models"
    dataset: "./coral_cache/your_dataset"
    progress: "./coral_cache/evolution_progress.json"
    coralx_root: "."

cache:
  base_checkpoint: your_org/your_model_name
  cleanup_threshold: 100
  metadata: true
  modal_native: true
  run_id: "your_experiment_v1"

evaluation:
  fitness_weights:
    primary_metric: 0.4
    secondary_metric: 0.3
    efficiency: 0.15
    quality: 0.15

evo:
  alpha_candidates: [4.0, 8.0, 16.0, 32.0, 64.0]
  rank_candidates: [4, 8, 16, 32, 48]
  dropout_candidates: [0.05, 0.1, 0.15, 0.2]
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  ca:
    grid_size: [8, 8]
    initial_density: 0.3
    rule_range: [1, 255]
    steps_range: [5, 25]

adapter_type: "lora"  # or "dora"

execution:
  generations: 20
  population_size: 10
  selection_mode: "pareto"
  output_dir: "./results"

experiment:
  name: "your_dataset_your_model_experiment"
  target: "your_dataset_your_model"
  dataset:
    training_problems:
      - problem_1
      - problem_2
      - problem_3
  model:
    name: "your_org/your_model_name"

cheap_knobs:
  temperature_range: [0.2, 0.8]
  top_p_range: [0.75, 0.92]
  top_k_range: [20, 50]
  repetition_penalty_range: [1.1, 1.3]
  max_tokens_range: [120, 350]

infra:
  executor: "modal"  # or "local" or "queue_modal"
  modal:
    app_name: "your-experiment-app"
    volume_name: "your-experiment-cache"
    functions:
      evaluate_genome:
        gpu: "A100-40GB"
        memory: 32768
        timeout: 1800

training:
  batch_size: 4
  epochs: 5
  learning_rate: 2e-4
  weight_decay: 0.01

seed: 42
system:
  fail_fast: true
  version_check: true
```

## 🔄 Reusable Components

### 1. **Domain Layer Components (100% Reusable)**

These pure functions can be reused across any plugin:

- `coral/domain/ca.py` - Cellular automata evolution
- `coral/domain/genome.py` - Genome representation
- `coral/domain/neat.py` - Population management
- `coral/domain/mapping.py` - Feature to LoRA mapping
- `coral/domain/feature_extraction.py` - CA feature extraction
- `coral/domain/threshold_gate.py` - Threshold gates
- `coral/domain/pareto_selection.py` - Multi-objective selection

### 2. **Application Layer Components (Highly Reusable)**

- `coral/application/evolution_engine.py` - Core evolution engine
- `coral/application/experiment_orchestrator.py` - Experiment coordination

### 3. **Infrastructure Components (Reusable with Configuration)**

- `infra/modal_executor.py` - Modal.com execution
- `infra/adapter_cache.py` - LoRA adapter caching
- `infra/queue_modal_executor.py` - Queue-based Modal execution

### 4. **Configuration System (Fully Reusable)**

- `coral/config/loader.py` - Configuration loading
- `coral/config/path_utils.py` - Path resolution
- `coral/domain/categorical_functors.py` - Configuration transformations

### 5. **CLI Interface (Minimal Changes Required)**

- `cli/coral.py` - Main CLI (add plugin loading)
- `cli/dashboard.py` - Dashboard interface

## 🎯 Integration Patterns

### 1. **Plugin Integration Pattern**

```python
# In your plugin's __init__.py
from .plugin import YourDatasetYourModelPlugin

__all__ = ['YourDatasetYourModelPlugin']
```

### 2. **Modal Integration Pattern**

```python
# In your Modal app file
@app.function(gpu="A100-40GB", memory=32768)
def your_model_generate(prompt: str, config: dict) -> str:
    import sys
    from pathlib import Path
    
    coralx_path = Path("/root/coralx")
    sys.path.insert(0, str(coralx_path))
    
    from infra.modal.your_model_service import your_model_generate_modal
    return your_model_generate_modal(prompt, config)
```

### 3. **Cache Integration Pattern**

```python
# Your model runner can leverage existing cache
from infra.adapter_cache import get_or_train_adapter

class YourModelRunner(ModelRunner):
    def _setup_adapter(self):
        # Reuse existing cache system
        cache_config = CacheConfig(
            cache_dir=self.cache_dir,
            run_id=self.run_id,
            cleanup_threshold=100
        )
        
        heavy_genes = HeavyGenes(
            rank=self.lora_cfg.rank,
            alpha=self.lora_cfg.alpha,
            dropout=self.lora_cfg.dropout,
            target_modules=self.lora_cfg.target_modules
        )
        
        self.adapter = get_or_train_adapter(
            heavy_genes=heavy_genes,
            cache_config=cache_config,
            training_function=self._train_adapter
        )
```

## 🚀 Quick Start Examples

### Example 1: Adding HumanEval Dataset

```python
# adapters/humaneval_real.py
class HumanEvalRealAdapter:
    def __init__(self, config):
        self.problems = self._load_humaneval_problems()
    
    def _load_humaneval_problems(self):
        # Load HumanEval problems from datasets library
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval")
        return dataset["test"]
    
    def evaluate_solution(self, problem, solution):
        # Use code execution to evaluate
        return self._execute_and_score(solution, problem["test"])
```

### Example 2: Adding GPT-4 Model

```python
# In your plugin
class GPT4Runner(ModelRunner):
    def __init__(self, lora_cfg, config, genome=None):
        self.client = OpenAI(api_key=config["openai_api_key"])
        self.model = "gpt-4"
    
    def generate(self, prompt, max_tokens=None, cheap_knobs=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=cheap_knobs.get("temperature", 0.7) if cheap_knobs else 0.7
        )
        return response.choices[0].message.content
```

## 📋 Testing Your Plugin

### Step 1: Create Test Configuration

```yaml
# test_config.yaml
paths:
  local:
    cache_root: "./test_cache"
    dataset: "./test_data"
    models: "./test_models"

execution:
  generations: 2
  population_size: 3

experiment:
  target: "your_dataset_your_model"
  dataset:
    training_problems: []
  model:
    name: "your_test_model"
```

### Step 2: Run Local Test

```bash
# Test locally first
python -m cli.coral --config test_config.yaml --executor local

# Then test with Modal
python -m cli.coral --config test_config.yaml --executor modal
```

### Step 3: Validate Plugin Integration

```python
# test_your_plugin.py
from plugins.your_dataset_your_model.plugin import YourDatasetYourModelPlugin

def test_plugin():
    config = {...}  # Your test config
    plugin = YourDatasetYourModelPlugin(config)
    
    # Test dataset provider
    dataset = plugin.dataset()
    problems = list(dataset.problems())
    assert len(problems) > 0
    
    # Test model factory
    model_factory = plugin.model_factory()
    model = model_factory(test_lora_config)
    result = model.generate("test prompt")
    assert isinstance(result, str)
    
    # Test fitness function
    fitness_fn = plugin.fitness_fn()
    score = fitness_fn(test_genome, model, problems[:1])
    assert 0.0 <= score <= 1.0
```

## 🔍 Debugging Tips

### 1. **Enable Fail-Fast Mode**

```yaml
system:
  fail_fast: true  # Crashes immediately on errors
```

### 2. **Check Configuration Loading**

```python
from coral.config.loader import load_config
config = load_config("your_config.yaml")
print(config.experiment)  # Verify experiment section
```

### 3. **Test Modal Functions Locally**

```python
# Test Modal functions without Modal
from infra.modal.your_model_service import your_model_generate_modal

result = your_model_generate_modal("test prompt", {
    "name": "your_model",
    "cache_dir": "./test_cache"
}, {"temperature": 0.7})
```

### 4. **Validate Plugin Interfaces**

```python
# Check if your classes implement the protocols
from coral.ports.interfaces import DatasetProvider, ModelRunner, FitnessFn

assert issubclass(YourDatasetProvider, DatasetProvider)
assert issubclass(YourModelRunner, ModelRunner)
assert issubclass(YourDatasetFitness, FitnessFn)
```

## 🎓 Best Practices

### 1. **Follow Category Theory Principles**
- Use `@dataclass(frozen=True)` for all data structures
- Keep domain functions pure (no side effects)
- Separate effectful operations in infrastructure layer

### 2. **Implement Fail-Fast Validation**
- Validate all configuration at startup
- Raise explicit errors with context
- No silent fallbacks or defaults

### 3. **Leverage Existing Infrastructure**
- Reuse cache system for adapters
- Use existing Modal integration patterns
- Follow established configuration patterns

### 4. **Test Thoroughly**
- Test plugin locally before Modal
- Validate all protocols are implemented
- Test with minimal configurations first

This guide provides a complete framework for adding new models and datasets to CoralX while maintaining the functional, category-theoretic architecture and fail-fast principles. 