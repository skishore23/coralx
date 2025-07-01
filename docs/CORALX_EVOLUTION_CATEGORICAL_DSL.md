# CoralX Evolution Categorical DSL
## A Domain-Specific Language for NEAT/CA Evolution with Category Theory

### Executive Summary

This DSL focuses specifically on the categorical structures present in CoralX's evolution loop: NEAT genetic operations, Cellular Automata dynamics, and the Feature→LoRA mapping pipeline. Unlike general category theory frameworks, this DSL is tailored for evolutionary computation with CA-based genome representations.

---

## 🧬 Core Evolution DSL Syntax

### 1. Evolution Category Definition

```haskell
evolution_category CoralXEvolution where
  -- Core evolution objects
  objects = {
    CASeed        :: (Grid, Rule, Steps),
    CAHistory     :: [CAState],
    CAFeatures    :: (Complexity, Intensity, Periodicity, Convergence), 
    LoRAConfig    :: (Rank, Alpha, Dropout, Modules),
    Genome        :: (CASeed, LoRAConfig, Fitness),
    Population    :: [Genome]
  }
  
  -- Primary evolution morphisms
  morphisms = {
    ca_evolve     :: CASeed ⟶ CAHistory,
    extract_feats :: CAHistory ⟶ CAFeatures,
    feat_to_lora  :: CAFeatures × Config ⟶ LoRAConfig,
    make_genome   :: CASeed × LoRAConfig ⟶ Genome,
    evaluate      :: Genome ⟶ Genome,  -- adds fitness
    
    -- NEAT genetic operations
    mutate        :: Genome ⟶ Genome,
    crossover     :: Genome × Genome ⟶ Genome,
    select        :: Population × Int ⟶ Population
  }
  
  -- Composition laws for CORAL-X pipeline
  compose {
    coral_pipeline :: CASeed ⟶ LoRAConfig = 
      feat_to_lora ∘ extract_feats ∘ ca_evolve
      
    full_evolution :: CASeed ⟶ Genome =
      evaluate ∘ make_genome ∘ (id × coral_pipeline)
  }
```

### 2. Cellular Automata Time Category

```haskell
ca_time_category CADynamics where
  -- CA state space and time
  objects = {
    CAState       :: Grid[Int],
    TimeStep      :: Nat,
    CARule        :: Int,
    Neighborhood  :: CAState ⟶ [Int]
  }
  
  -- CA evolution morphisms
  morphisms = {
    next_step     :: CAState × CARule ⟶ CAState,
    moore_hood    :: (Int, Int) ⟶ [(Int, Int)],
    apply_rule    :: [Int] × CARule ⟶ Int,
    evolve_n      :: CAState × CARule × Nat ⟶ [CAState]
  }
  
  -- Time evolution functor
  time_functor :: DiscreteTime ⟶ CAState where
    fmap(+1) :: CAState ⟶ CAState = λstate. next_step(state, rule)
    
  -- Compositionality laws
  evolution_laws {
    -- Time additivity: evolving n+m steps = evolving n then m steps
    evolve_n(state, rule, n+m) ≡ evolve_n(evolve_n(state, rule, n)[last], rule, m)
    
    -- Determinism: same initial conditions → same result
    ∀seed1, seed2. seed1 = seed2 ⟹ ca_evolve(seed1) = ca_evolve(seed2)
  }
```

### 3. Feature Space Category

```haskell
feature_category CAFeatureSpace where
  -- Feature measurement spaces
  objects = {
    ComplexitySpace   :: [0, 1],
    IntensitySpace    :: [0, 1], 
    PeriodicitySpace  :: [0, 1],
    ConvergenceSpace  :: [0, 1],
    FeatureSpace      :: ComplexitySpace × IntensitySpace × PeriodicitySpace × ConvergenceSpace
  }
  
  -- Measurement functors
  morphisms = {
    measure_complexity  :: [CAState] ⟶ Float,
    measure_intensity   :: [CAState] ⟶ Float,
    measure_periodicity :: [CAState] ⟶ Float,
    measure_convergence :: [CAState] ⟶ Float,
    
    -- Combined measurement functor
    extract_features :: [CAState] ⟶ FeatureSpace = 
      ⟨measure_complexity, measure_intensity, measure_periodicity, measure_convergence⟩
  }
  
  -- Feature extraction laws
  measurement_laws {
    -- Monotonicity: more complex CA → higher complexity measure
    ∀ca1, ca2. more_complex(ca1, ca2) ⟹ measure_complexity(ca1) ≥ measure_complexity(ca2)
    
    -- Boundedness: all measurements in [0,1]
    ∀ca. 0 ≤ measure_complexity(ca) ≤ 1
    
    -- Independence: measurements are orthogonal
    measure_complexity ⊥ measure_intensity ⊥ measure_periodicity ⊥ measure_convergence
  }
```

### 4. NEAT Genetic Operations as Functors

```haskell
neat_functors GeneticOperations where
  -- Mutation functor
  mutation_functor :: EvolutionCategory ⟶ EvolutionCategory where
    fmap(mutate) :: Genome ⟶ Genome
    fmap(mutate) :: Population ⟶ Population
    
    -- Two mutation types
    ca_mutation :: CASeed ⟶ CASeed = random_grid_flip ∘ random_rule_change ∘ random_steps_change
    lora_mutation :: LoRAConfig ⟶ LoRAConfig = random_rank ∘ random_alpha ∘ random_dropout
    
    genome_mutation :: Genome ⟶ Genome = 
      if random() < 0.7 then
        make_genome ∘ (ca_mutation × (coral_pipeline ∘ ca_mutation))
      else
        make_genome ∘ (id × lora_mutation)
  
  -- Crossover functor (binary operation)
  crossover_functor :: EvolutionCategory × EvolutionCategory ⟶ EvolutionCategory where
    fmap(crossover) :: Genome × Genome ⟶ Genome
    
    -- CA crossover: combine grid from one parent, rule/steps from other
    ca_crossover :: CASeed × CASeed ⟶ CASeed = 
      λ(seed1, seed2). if random() < 0.5 then
        CASeed(seed1.grid, seed2.rule, seed2.steps)
      else
        CASeed(seed2.grid, seed1.rule, seed1.steps)
    
    genome_crossover :: Genome × Genome ⟶ Genome =
      make_genome ∘ (ca_crossover × (coral_pipeline ∘ ca_crossover))
  
  -- Selection functor  
  selection_functor :: EvolutionCategory ⟶ EvolutionCategory where
    fmap(select) :: Population ⟶ Population
    
    fitness_ordering :: Population ⟶ Population = sort_by_fitness(descending)
    take_best :: Population × Int ⟶ Population = λ(pop, k). take(k, pop)
    
    selection :: Population × Int ⟶ Population = take_best ∘ (fitness_ordering × id)
    
  -- Functor laws verification
  genetic_laws {
    -- Identity preservation
    fmap(id) = id
    
    -- Composition preservation  
    fmap(g ∘ f) = fmap(g) ∘ fmap(f)
    
    -- Structure preservation
    ∀genome. type(fmap(mutate)(genome)) = type(genome)
    ∀pop. size(fmap(select)(pop)) ≤ size(pop)
  }
```

---

## 🔄 Evolution Pipeline DSL

### Complete Evolution Step Definition

```haskell
evolution_step :: Population ⟶ Population where
  step_composition = 
    population_reconstruction ∘ 
    reproduction_operations ∘ 
    selection_operation ∘ 
    evaluation_operation
    
  evaluation_operation :: Population ⟶ Population = 
    map(full_evolution)  -- Apply CA→LoRA→evaluation to each genome
    
  selection_operation :: Population ⟶ Population =
    select(_, survival_count)
    
  reproduction_operations :: Population ⟶ Population = coproduct {
    mutation_branch   :: Population ⟶ [Genome] = map(mutate),
    crossover_branch  :: Population ⟶ [Genome] = pairwise(crossover)
  }
  
  population_reconstruction :: [Genome] ⟶ Population =
    Population ∘ take(population_size)
```

### Example DSL Usage

```haskell
experiment QuixBugsEvolution extends CoralXEvolution where
  
  -- CA parameters for code generation
  ca_config = {
    grid_size = (12, 12),
    rule_range = (50, 200),
    steps_range = (10, 30),
    initial_density = 0.4
  }
  
  -- LoRA configuration
  lora_config = {
    rank_candidates = [8, 16, 32, 64],
    alpha_candidates = [4.0, 8.0, 16.0],
    dropout_candidates = [0.1, 0.2, 0.3],
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
  }
  
  -- Evolution pipeline
  run_evolution :: () ⟶ EvolutionResults =
    extract_results ∘ evolution_loop ∘ initialize_population
```

---

## 🧮 Categorical Structures in Current CoralX Code

### Actual Implementation Evidence

The DSL formalizes structures already present in CoralX:

**Evolution Category Objects**:
```python
# From coral/domain/ca.py
@dataclass(frozen=True)
class CASeed:
    grid: NDArray[np.int_]
    rule: int
    steps: int

# From coral/domain/feature_extraction.py
@dataclass(frozen=True)
class CAFeatures:
    complexity: float
    intensity: float
    periodicity: float
    convergence: float
```

**Evolution Category Morphisms**:
```python
# CA evolution morphism: CASeed → CAStateHistory
def evolve(seed: CASeed) -> CAStateHistory:
    """Pure arrow: Seed ──▶ History."""
    
# Feature extraction morphism: CAStateHistory → CAFeatures
def extract_features(hist: CAStateHistory) -> CAFeatures:
    """History ──▶ Features. Pure & vectorisable."""
    
# Feature-to-LoRA morphism: CAFeatures × Config → LoRAConfig
def map_features_to_lora_config(features: CAFeatures, config: Dict[str, Any]) -> AdapterConfig:
    """Categorical functor: FeatureSpace ⟶ ConfigurationSpace"""
```

**NEAT Genetic Functors**:
```python
# Mutation functor: Genome → Genome
def mutate(genome: Genome, evo_cfg: EvolutionConfig, rng: Random) -> Genome:
    """Categorical morphism: Genome ⟶ Genome"""
    
# Crossover functor: Genome × Genome → Genome  
def crossover(p1: Genome, p2: Genome, evo_cfg: EvolutionConfig, rng: Random) -> Genome:
    """Categorical product operation: Genome × Genome ⟶ Genome"""
    
# Selection functor: Population → Population
def select(pop: Population, k: int) -> Population:
    """Categorical morphism: Population ⟶ Population"""
```

---

## 🎯 Benefits of Evolution-Focused DSL

### 1. Mathematical Rigor for Evolution
- **Categorical laws ensure correctness** of genetic operations
- **Composition guarantees** for complex evolution pipelines  
- **Type safety** prevents invalid evolutionary steps
- **Automatic verification** of evolution properties

### 2. CA-Specific Categorical Structure
- **Time evolution as functor** captures CA dynamics mathematically
- **Feature extraction as measurement functor** ensures consistent feature computation
- **Configuration mapping preserves mathematical relationships** between features and parameters

### 3. NEAT Operations as Functors
- **Mutation and crossover preserve genome structure** categorically
- **Selection maintains population ordering properties**
- **Parallel genetic operations** guaranteed to be safe through functorial structure

### 4. Performance and Optimization
- **Categorical fusion laws** enable automatic pipeline optimization
- **Functor laws guarantee parallelization safety**
- **Cache coherence through categorical limits** ensures distributed consistency

### 5. Extensibility for New Evolution Algorithms
- **Categorical framework** easily extended to new genetic operators
- **Functor composition** enables modular algorithm design
- **Property verification** automatically applies to extensions

This DSL transforms CoralX's evolution loop from imperative code into a mathematically rigorous categorical framework while maintaining all the practical benefits of the existing implementation.

```mermaid
graph TB
    subgraph "1. Evolution Category - Core CORAL-X Pipeline"
        CASeed["CASeed<br/>(Grid, Rule, Steps)"]
        CAHistory["CAStateHistory<br/>[CAState]"]
        CAFeatures["CAFeatures<br/>(Complexity, Intensity,<br/>Periodicity, Convergence)"]
        LoRAConfig["LoRAConfig<br/>(Rank, Alpha, Dropout,<br/>Modules)"]
        Genome["Genome<br/>(CASeed, LoRAConfig, Fitness)"]
        Population["Population<br/>[Genome]"]
    end

    subgraph "2. CA Time Category - Discrete Dynamics"
        CAState1["CAState(t)"]
        CAState2["CAState(t+1)"]
        CAState3["CAState(t+2)"]
        TimeStep["TimeStep"]
        CARule["CA Rule"]
    end

    subgraph "3. Feature Space Category - Measurement Functors"
        ComplexitySpace["ComplexitySpace<br/>[0,1]"]
        IntensitySpace["IntensitySpace<br/>[0,1]"]
        PeriodicitySpace["PeriodicitySpace<br/>[0,1]"]
        ConvergenceSpace["ConvergenceSpace<br/>[0,1]"]
        ProductSpace["FeatureSpace<br/>= Product Category"]
    end

    subgraph "4. NEAT Genetic Functors"
        MutationF["Mutation Functor<br/>Genome → Genome"]
        CrossoverF["Crossover Functor<br/>Genome × Genome → Genome"]
        SelectionF["Selection Functor<br/>Population → Population"]
        EvolutionStep["Evolution Step<br/>Population → Population"]
    end

    %% Core evolution morphisms
    CASeed -->|"ca_evolve"| CAHistory
    CAHistory -->|"extract_features"| CAFeatures
    CAFeatures -->|"feat_to_lora + Config"| LoRAConfig
    CASeed -.->|"id"| CASeed
    LoRAConfig -.->|"make_genome"| Genome
    Genome -->|"evaluate"| Genome

    %% CA time evolution
    CAState1 -->|"next_step + rule"| CAState2
    CAState2 -->|"next_step + rule"| CAState3
    TimeStep -.->|"time functor"| CAState2
    CARule -.->|"apply rule"| CAState2

    %% Feature measurements
    ComplexitySpace --> ProductSpace
    IntensitySpace --> ProductSpace
    PeriodicitySpace --> ProductSpace
    ConvergenceSpace --> ProductSpace
    CAHistory -.->|"measurement functor"| ProductSpace

    %% Genetic operations
    Genome -->|"mutation"| MutationF
    Genome -->|"crossover"| CrossoverF
    Population -->|"selection"| SelectionF
    MutationF --> EvolutionStep
    CrossoverF --> EvolutionStep
    SelectionF --> EvolutionStep
    EvolutionStep -->|"categorical composition"| Population

    %% Composition laws
    CASeed -.->|"coral_pipeline = feat_to_lora ∘ extract_features ∘ ca_evolve"| LoRAConfig

    %% Styling
    classDef evolutionCategory fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef caTimeCategory fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef featureCategory fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef geneticFunctors fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef composition fill:#fce4ec,stroke:#880e4f,stroke-width:3px

    class CASeed,CAHistory,CAFeatures,LoRAConfig,Genome,Population evolutionCategory
    class CAState1,CAState2,CAState3,TimeStep,CARule caTimeCategory
    class ComplexitySpace,IntensitySpace,PeriodicitySpace,ConvergenceSpace,ProductSpace featureCategory
    class MutationF,CrossoverF,SelectionF,EvolutionStep geneticFunctors
    ```

    image.png