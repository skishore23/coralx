# 🏗️ **Unified Config Architecture Refactoring**

## 🔍 **Problem: Impedance Mismatch**

**Current Issue**: Converting between structured configs (`CoralConfig`) and raw dicts everywhere:

```python
# ❌ CURRENT ANTI-PATTERN
def __init__(self, cfg: CoralConfig, raw_config: Dict[str, Any] = None):
    self.cfg = cfg              # Structured for type safety
    self.raw_config = raw_config # Raw dict for Modal functions
    
def _get_full_config_dict(self) -> Dict[str, Any]:
    return self.raw_config  # Constant conversion!

# Complex type detection everywhere
if hasattr(evo_raw, 'rank_candidates'):  # EvolutionConfig object
    evo_cfg = evo_raw
else:  # Raw dictionary
    evo_cfg = EvolutionConfig(...)
```

## 🎯 **Solution: Unified Config Object**

### **Single Source of Truth with Dual Access Patterns**

```python
@dataclass(frozen=True)
class UnifiedConfig:
    """
    Category-theoretic configuration with dual access patterns.
    Eliminates impedance mismatch between structured and dict access.
    """
    # Raw YAML data (source of truth)
    _raw_data: Dict[str, Any]
    
    # Lazy-loaded structured fields
    _evo: Optional[EvolutionConfig] = None
    _threshold: Optional[ThresholdConfig] = None
    
    def __post_init__(self):
        """Validate structure on creation."""
        self._validate_config(self._raw_data)
    
    # 🧮 CATEGORICAL DICT ACCESS (for legacy compatibility)
    def __getitem__(self, key: str):
        """Direct dict access: config['evo']"""
        return self._raw_data[key]
    
    def __contains__(self, key: str) -> bool:
        """'key' in config"""
        return key in self._raw_data
    
    def get(self, key: str, default=None):
        """config.get('key', default)"""
        return self._raw_data.get(key, default)
    
    def keys(self):
        return self._raw_data.keys()
    
    def items(self):
        return self._raw_data.items()
    
    # 🧮 CATEGORICAL STRUCTURED ACCESS (for type safety)
    @property
    def evo(self) -> EvolutionConfig:
        """Lazy-loaded structured evolution config."""
        if self._evo is None:
            evo_raw = self._raw_data['evo']
            object.__setattr__(self, '_evo', EvolutionConfig(
                rank_candidates=tuple(evo_raw['rank_candidates']),
                alpha_candidates=tuple(evo_raw['alpha_candidates']),
                dropout_candidates=tuple(evo_raw['dropout_candidates']),
                target_modules=tuple(evo_raw['target_modules'])
            ))
        return self._evo
    
    @property 
    def threshold(self) -> ThresholdConfig:
        """Lazy-loaded structured threshold config."""
        if self._threshold is None:
            # Build from raw data...
            pass
        return self._threshold
    
    # 🧮 CATEGORICAL FUNCTORS (Modal serialization)
    def serialize_for_modal(self) -> Dict[str, Any]:
        """Natural transformation: UnifiedConfig → Modal Dict"""
        return self._raw_data.copy()  # Raw data is already Modal-ready
    
    def serialize_for_executor(self, executor_type: str) -> Dict[str, Any]:
        """Functorial transformation for different execution contexts."""
        config = self._raw_data.copy()
        
        # Apply context-specific transformations
        if executor_type == 'modal':
            config['infra']['executor'] = 'modal'
            # Use modal paths if available
            if 'paths' in config and 'modal' in config['paths']:
                config['current_paths'] = config['paths']['modal']
        elif executor_type == 'local':
            config['infra']['executor'] = 'local'
            if 'paths' in config and 'local' in config['paths']:
                config['current_paths'] = config['paths']['local']
                
        return config
    
    # 🧮 CATEGORICAL MORPHISMS (Pure transformations)
    def with_population_size(self, size: int) -> 'UnifiedConfig':
        """Immutable update - returns new config."""
        new_raw = self._raw_data.copy()
        new_raw['execution']['population_size'] = size
        return UnifiedConfig(new_raw)
    
    def with_executor(self, executor_type: str) -> 'UnifiedConfig':
        """Immutable executor change."""
        new_raw = self._raw_data.copy()
        new_raw['infra']['executor'] = executor_type
        return UnifiedConfig(new_raw)


# 🧮 CATEGORICAL LOADER
def load_unified_config(path: str) -> UnifiedConfig:
    """Pure functor: YAML Path → UnifiedConfig"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"FAIL-FAST: Config file not found: {path}")
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    return UnifiedConfig(raw_config)
```

## 🔧 **Refactored EvolutionEngine**

```python
class EvolutionEngine:
    def __init__(self,
                 config: UnifiedConfig,  # ✅ SINGLE CONFIG OBJECT
                 fitness_fn: FitnessFn,
                 executor: Executor,
                 model_factory: Callable,
                 dataset: DatasetProvider,
                 run_id: str = None):
        
        self.config = config  # ✅ NO MORE cfg + raw_config
        self.fitness_fn = fitness_fn
        self.executor = executor
        # ...rest unchanged
    
    def _train_adapters_parallel(self, unique_heavy_genes):
        """No more _get_full_config_dict() needed!"""
        for genes_hash, heavy_genes in training_needed.items():
            # ✅ CLEAN: Direct Modal serialization
            training_config = self.config.serialize_for_modal()
            
            future = self.executor.submit_training(
                base_model=base_model,
                heavy_genes=heavy_genes, 
                save_path=save_path,
                config=training_config  # ✅ CLEAN: No conversion needed
            )
    
    def _apply_threshold_gate(self, pop, gen):
        """No more complex config access!"""
        # ✅ CLEAN: Structured access for type safety
        current_thresholds = calculate_dynamic_thresholds(
            gen, self.max_generations, self.config.threshold
        )
        
        # ✅ CLEAN: Dict access for legacy compatibility  
        population_size = self.config['execution']['population_size']
```

## 🔧 **Refactored Monadic Functions**

```python
def map_features_to_lora_config_monadic(
    features: CAFeatures, 
    config: UnifiedConfig,  # ✅ SINGLE TYPE
    diversity_strength: float = 1.0, 
    genome_index: int = 0
) -> 'Result[AdapterConfig, str]':
    """No more type detection needed!"""
    
    # ✅ CLEAN: Structured access
    evo_cfg = config.evo
    
    # ✅ CLEAN: Dict access  
    adapter_type = config.get('adapter_type', 'lora')
    
    # No more complex validation chains!
    rank = _map_with_enhanced_diversity(features, evo_cfg.rank_candidates, ...)
    alpha = _map_with_enhanced_diversity(features, evo_cfg.alpha_candidates, ...)
    dropout = _map_with_enhanced_diversity(features, evo_cfg.dropout_candidates, ...)
    
    return success(AdapterConfig(
        r=rank, alpha=alpha, dropout=dropout,
        target_modules=evo_cfg.target_modules,
        adapter_type=adapter_type
    ))
```

## 🧪 **Simplified Tests**

```python
def test_monadic_pipeline_success(self):
    """No more dict vs EvolutionConfig complexity!"""
    config = load_unified_config("config/test.yaml")
    
    result = compose_ca_pipeline_monadic(self.test_seed, config)
    self.assertTrue(result.is_success())
    
    adapter_config = result.unwrap()
    
    # ✅ CLEAN: No more type detection
    self.assertIn(adapter_config.r, config.evo.rank_candidates)
    self.assertIn(adapter_config.alpha, config.evo.alpha_candidates)
```

## 🎯 **Migration Strategy**

### **Phase 1: Create UnifiedConfig**
1. Create `UnifiedConfig` class
2. Add dual access patterns (dict + structured)
3. Test with existing code (should be compatible)

### **Phase 2: Replace CoralConfig Usage**
1. Update `load_config()` to return `UnifiedConfig`
2. Update `EvolutionEngine` constructor
3. Remove `raw_config` parameter everywhere

### **Phase 3: Simplify Code**
1. Remove `_get_full_config_dict()` methods
2. Remove type detection logic
3. Simplify Modal serialization
4. Clean up tests

### **Phase 4: Remove Legacy**
1. Delete `CoralConfig` class
2. Remove dict conversion methods
3. Update documentation

## 🚀 **Benefits**

### **✅ Eliminates Impedance Mismatch**
- Single source of truth
- No more dual config storage
- No more type conversion

### **✅ Maintains Compatibility** 
- Dict access still works: `config['evo']`
- Structured access available: `config.evo.rank_candidates`
- Existing code continues working

### **✅ Categorical Elegance**
- Pure functors for context switching
- Natural transformations for serialization
- Immutable morphisms for config updates

### **✅ Engineering Benefits**
- Reduced complexity
- Better performance (no copying)
- Easier maintenance
- Clear responsibility boundaries

 