# Extending CORAL-X

## Adding Custom Models

The plugin architecture makes adding new models straightforward. Each model needs a fitness function that evaluates LoRA adapters on your specific tasks.

### Basic Plugin Structure

```
plugins/my_model/
├── __init__.py
├── plugin.py          # Main fitness implementation
└── config.yaml        # Model-specific settings
```

### Implementing the Fitness Function

```python
# plugins/my_model/plugin.py
from coral.ports.interfaces import FitnessFn
from coral.domain.genome import Genome
from dataclasses import dataclass

@dataclass(frozen=True)
class MyModelObjectives:
    task_performance: float
    efficiency: float 
    safety: float
    robustness: float

class MyModelFitness(FitnessFn):
    def __init__(self, config):
        self.model_name = config.experiment.model.name
        self.test_samples = config.evaluation.test_samples
        
    def __call__(self, genome: Genome) -> MyModelObjectives:
        # Load base model
        model = self.load_model()
        
        # Apply LoRA configuration from genome
        adapter = self.train_or_load_adapter(genome)
        model = self.apply_adapter(model, adapter)
        
        # Evaluate on your tasks
        results = self.evaluate_on_tasks(model)
        
        return MyModelObjectives(
            task_performance=results['accuracy'],
            efficiency=1.0 / results['inference_time'],
            safety=results['safety_score'],
            robustness=results['cross_validation_score']
        )
    
    def train_or_load_adapter(self, genome: Genome):
        # Check cache first
        cache_key = self.cache.generate_key(genome)
        if self.cache.exists(cache_key):
            return self.cache.load(cache_key)
        
        # Train new adapter
        return self.train_lora_adapter(genome)
```

### Registering Your Plugin

Add your plugin to the configuration:

```yaml
# config/my_model_config.yaml
experiment:
  target: "my_model"  # Plugin name
  model:
    name: "your/huggingface-model"
    
evaluation:
  test_samples: 50
  fitness_weights:
    task_performance: 0.5
    efficiency: 0.2
    safety: 0.2
    robustness: 0.1
```

## Custom Datasets

### Dataset Structure

```
datasets/my_dataset/
├── train.json         # Training examples
├── test.json          # Test/evaluation examples  
└── metadata.json      # Dataset description
```

### Data Loading

```python
def load_my_dataset(config):
    """Load your custom dataset for training and evaluation."""
    dataset_path = Path(config.experiment.dataset.path)
    
    # Load training data
    with open(dataset_path / "train.json") as f:
        train_data = json.load(f)
    
    # Load test data
    with open(dataset_path / "test.json") as f:
        test_data = json.load(f)
    
    return {
        'train': format_for_training(train_data),
        'test': format_for_evaluation(test_data)
    }
```

### Evaluation Functions

```python
def evaluate_my_task(model, test_data):
    """Evaluate model performance on your specific task."""
    correct = 0
    total = len(test_data)
    inference_times = []
    
    for example in test_data:
        start_time = time.time()
        prediction = model.generate(example['input'])
        inference_time = time.time() - start_time
        
        if prediction == example['expected_output']:
            correct += 1
        inference_times.append(inference_time)
    
    return {
        'accuracy': correct / total,
        'avg_inference_time': np.mean(inference_times),
        'total_examples': total
    }
```

## Custom Evolution Strategies

### Alternative Selection Methods

```python
# coral/services/custom_selection.py
from coral.services.population_manager import PopulationManager

class TournamentSelection(PopulationManager):
    def __init__(self, tournament_size=3):
        super().__init__()
        self.tournament_size = tournament_size
    
    def select_survivors(self, population, fitness_scores):
        survivors = []
        for _ in range(self.target_population_size):
            # Random tournament
            tournament = random.sample(
                list(zip(population, fitness_scores)), 
                self.tournament_size
            )
            # Pick winner
            winner = max(tournament, key=lambda x: x[1])
            survivors.append(winner[0])
        return survivors
```

### Custom Mutation Operators

```python
# coral/domain/custom_mutations.py
from coral.domain.genome import Genome

def adaptive_mutation(genome: Genome, generation: int) -> Genome:
    """Mutation rate decreases with generation for convergence."""
    base_mutation_rate = 0.1
    current_rate = base_mutation_rate * (0.9 ** generation)
    
    if random() < current_rate:
        # Mutate heavy genes (structural)
        new_heavy = mutate_heavy_genes(genome.heavy_genes)
        return genome.with_heavy_genes(new_heavy)
    else:
        # Mutate light genes (optimization)  
        new_light = mutate_light_genes(genome.light_genes)
        return genome.with_light_genes(new_light)
```

### Custom CA Rules

```python
# coral/domain/custom_ca.py
def custom_ca_rule(grid, rule_number=None):
    """Define your own cellular automata evolution rules."""
    if rule_number == 999:  # Your custom rule
        return your_special_ca_logic(grid)
    else:
        return standard_ca_evolution(grid, rule_number)

def your_special_ca_logic(grid):
    """Example: Conway's Game of Life variant."""
    new_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = count_neighbors(grid, i, j)
            # Your custom rules here
            if grid[i,j] == 1 and neighbors in [2, 3]:
                new_grid[i,j] = 1
            elif grid[i,j] == 0 and neighbors == 3:
                new_grid[i,j] = 1
    return new_grid
```

## Infrastructure Extensions

### Custom Executors

```python
# coral/infra/kubernetes_executor.py
from coral.ports.interfaces import Executor

class KubernetesExecutor(Executor):
    def __init__(self, namespace="coralx"):
        self.namespace = namespace
        self.k8s_client = self.setup_kubernetes_client()
    
    def execute_batch(self, genomes: List[Genome]) -> List[float]:
        # Create Kubernetes jobs for each genome
        jobs = []
        for genome in genomes:
            job = self.create_training_job(genome)
            jobs.append(job)
        
        # Wait for completion and collect results
        results = self.wait_for_jobs(jobs)
        return [r.fitness_score for r in results]
```

### Custom Cache Backends

```python
# coral/infra/s3_cache.py
from coral.infra.adapter_cache import AdapterCache

class S3AdapterCache(AdapterCache):
    def __init__(self, bucket_name, region="us-west-2"):
        self.s3_client = boto3.client('s3', region_name=region)
        self.bucket = bucket_name
    
    def store_adapter(self, cache_key: str, adapter_path: str):
        # Upload to S3
        self.s3_client.upload_file(
            adapter_path, 
            self.bucket, 
            f"adapters/{cache_key}.tar.gz"
        )
    
    def load_adapter(self, cache_key: str) -> str:
        # Download from S3
        local_path = f"/tmp/{cache_key}"
        self.s3_client.download_file(
            self.bucket,
            f"adapters/{cache_key}.tar.gz", 
            local_path
        )
        return local_path
```

## Advanced Configurations

### Multi-Modal Evolution

```yaml
# Different CA rules for different aspects
evo:
  ca_rules:
    structural:       # Heavy gene generation
      rule_range: [30, 60]
      grid_size: [8, 8]
    optimization:     # Light gene generation  
      rule_range: [150, 200]
      grid_size: [4, 4]
      
  evolution_stages:
    - name: "exploration"
      generations: 10
      mutation_rate: 0.3
      crossover_rate: 0.7
    - name: "exploitation"  
      generations: 10
      mutation_rate: 0.1
      crossover_rate: 0.9
```

### Curriculum Learning

```yaml
# Progressive difficulty increase
experiment:
  curriculum:
    enabled: true
    stages:
      - name: "easy"
        generations: 5
        dataset_filter: "difficulty < 3"
      - name: "medium"
        generations: 5  
        dataset_filter: "difficulty < 7"
      - name: "hard"
        generations: 10
        dataset_filter: "all"
```

### Transfer Learning

```yaml
# Warm-start from related task
cache:
  transfer_learning:
    enabled: true
    source_task: "similar_task_cache/"
    similarity_threshold: 0.8  # How similar before reuse
```

## Testing Your Extensions

### Unit Testing

```python
# tests/test_my_plugin.py
def test_my_fitness_function():
    config = load_test_config()
    fitness_fn = MyModelFitness(config)
    
    # Create test genome
    genome = create_test_genome(rank=8, alpha=16)
    
    # Should return valid objectives
    result = fitness_fn(genome)
    assert 0.0 <= result.task_performance <= 1.0
    assert result.efficiency > 0.0
```

### Integration Testing

```python
def test_end_to_end_evolution():
    config = create_minimal_config(
        generations=2,
        population_size=2
    )
    
    orchestrator = EvolutionOrchestrator(config)
    result = orchestrator.run_evolution()
    
    # Should complete without errors
    assert result.status == "completed"
    assert len(result.final_population) > 0
```

## Best Practices

### Performance
- Cache aggressively - training is expensive
- Use lazy evaluation - don't train until you need the fitness
- Batch operations when possible
- Monitor memory usage and adjust batch sizes

### Debugging
- Log at appropriate levels (INFO for progress, DEBUG for details)
- Use structured logging for easy parsing
- Include context in error messages
- Make failures reproducible with seeds

### Configuration  
- Validate early with Pydantic schemas
- Use environment variables for secrets
- Provide sensible defaults
- Document configuration options clearly

### Evolution Tuning
- Start with diverse populations 
- Balance exploration vs exploitation
- Monitor population diversity
- Use multi-objective optimization for real-world trade-offs

---

---

**Research Note**: As an experimental framework, CORAL-X is designed to facilitate research into evolutionary ML optimization. The extension points described here allow researchers to explore different models, datasets, and evolutionary strategies while leveraging the core infrastructure.

*Contributions that advance the research goals of evolutionary hyperparameter optimization are particularly welcome.*