# 🏗️ **Enhanced CoralConfig Architecture Refactoring**

## 🔍 **Problem: Impedance Mismatch**

**Current Issue**: `CoralConfig` + `raw_config` dual storage pattern creates conversion complexity:

```python
# ❌ CURRENT ANTI-PATTERN
def __init__(self, cfg: CoralConfig, raw_config: Dict[str, Any] = None):
    self.cfg = cfg              # Structured for type safety
    self.raw_config = raw_config # Raw dict for Modal functions
    
def _get_full_config_dict(self) -> Dict[str, Any]:
    return self.raw_config  # Why do we need this?
```

## 🎯 **Solution: Enhanced CoralConfig**

### **Single Source of Truth with Dual Access Patterns**

```python
@dataclass(frozen=True)
class CoralConfig:
    """
    Enhanced configuration container with categorical dual access.
    Eliminates impedance mismatch by storing raw YAML as source of truth.
    """
    # Raw YAML data (source of truth)
    _raw_data: Dict[str, Any]
    
    # Lazy-loaded structured fields
    _evo: Optional[EvolutionConfig] = None
    _threshold: Optional[ThresholdConfig] = None
    
    def __post_init__(self):
        """Validate structure on creation."""
        self._validate_config(self._raw_data)
    
    # 🧮 CATEGORICAL DICT ACCESS (existing functionality enhanced)
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
    
    def copy(self):
        """Return raw dict copy - no more conversion needed!"""
        return self._raw_data.copy()
    
    # 🧮 CATEGORICAL STRUCTURED ACCESS (lazy-loaded for performance)
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
            threshold_raw = self._raw_data['threshold']
            base_thresh = threshold_raw['base_thresholds']
            max_thresh = threshold_raw['max_thresholds']
            
            object.__setattr__(self, '_threshold', ThresholdConfig(
                base_thresholds=ObjectiveThresholds(
                    bugfix=base_thresh['bugfix'],
                    style=base_thresh['style'],
                    security=base_thresh['security'],
                    runtime=base_thresh['runtime'],
                    syntax=base_thresh.get('syntax', 0.3)
                ),
                max_thresholds=ObjectiveThresholds(
                    bugfix=max_thresh['bugfix'],
                    style=max_thresh['style'],
                    security=max_thresh['security'],
                    runtime=max_thresh['runtime'],
                    syntax=max_thresh.get('syntax', 0.9)
                ),
                schedule=threshold_raw['schedule']
            ))
        return self._threshold
    
    @property
    def seed(self) -> int:
        """Direct access to seed."""
        return self._raw_data['seed']
    
    @property
    def execution(self) -> Dict[str, Any]:
        """Direct access to execution config."""
        return self._raw_data['execution']
    
    @property
    def infra(self) -> Dict[str, Any]:
        """Direct access to infra config."""
        return self._raw_data['infra']
    
    @property
    def experiment(self) -> Dict[str, Any]:
        """Direct access to experiment config."""
        return self._raw_data['experiment']
    
    @property
    def cache(self) -> Dict[str, Any]:
        """Direct access to cache config."""
        return self._raw_data['cache']
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Direct access to evaluation config."""
        return self._raw_data['evaluation']
    
    @property
    def adapter_type(self) -> str:
        """Direct access to adapter type."""
        return self._raw_data.get('adapter_type', 'lora')
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Direct access to paths config."""
        return self._raw_data.get('paths', {})
    
    # 🧮 CATEGORICAL FUNCTORS (Modal serialization - now trivial!)
    def serialize_for_modal(self) -> Dict[str, Any]:
        """Natural transformation: CoralConfig → Modal Dict"""
        return self._raw_data.copy()  # Raw data is already Modal-ready!
    
    def serialize_for_executor(self, executor_type: str) -> Dict[str, Any]:
        """Functorial transformation for different execution contexts."""
        config = self._raw_data.copy()
        
        # Apply context-specific transformations
        if executor_type == 'modal':
            config['infra']['executor'] = 'modal'
            if 'paths' in config and 'modal' in config['paths']:
                config['current_paths'] = config['paths']['modal']
        elif executor_type == 'local':
            config['infra']['executor'] = 'local'
            if 'paths' in config and 'local' in config['paths']:
                config['current_paths'] = config['paths']['local']
                
        return config
    
    # 🧮 CATEGORICAL MORPHISMS (Pure transformations)
    def with_population_size(self, size: int) -> 'CoralConfig':
        """Immutable update - returns new config."""
        new_raw = self._raw_data.copy()
        new_raw['execution']['population_size'] = size
        return CoralConfig(new_raw)
    
    def with_executor(self, executor_type: str) -> 'CoralConfig':
        """Immutable executor change."""
        new_raw = self._raw_data.copy()
        new_raw['infra']['executor'] = executor_type
        return CoralConfig(new_raw)


# 🧮 ENHANCED LOADER (same function name)
def load_config(path: str, env: Dict[str, str] = None) -> CoralConfig:
    """Enhanced loader: YAML Path → CoralConfig with raw data storage."""
    if env is None:
        env = os.environ
    
    # Load YAML
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"FAIL-FAST: Config file not found: {path}")
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    if not raw_config:
        raise ValueError(f"FAIL-FAST: Config file is empty or invalid: {path}")
    
    # Apply environment overrides
    raw_config = _apply_env_overrides(raw_config, env)
    
    # Create enhanced CoralConfig with raw data as source of truth
    return CoralConfig(raw_config)
```

## 🔧 **Simplified EvolutionEngine**

```python
class EvolutionEngine:
    def __init__(self,
                 config: CoralConfig,  # ✅ SINGLE CONFIG OBJECT
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
    
    # ✅ DELETE: _get_full_config_dict() method completely removed!
    
    def _apply_threshold_gate(self, pop, gen):
        """Clean config access!"""
        # ✅ CLEAN: Structured access for type safety
        current_thresholds = calculate_dynamic_thresholds(
            gen, self.max_generations, self.config.threshold
        )
        
        # ✅ CLEAN: Dict access for legacy compatibility  
        population_size = self.config['execution']['population_size']
```

## 🔧 **Simplified Monadic Functions**

```python
def map_features_to_lora_config_monadic(
    features: CAFeatures, 
    config: CoralConfig,  # ✅ SAME TYPE as everywhere else
    diversity_strength: float = 1.0, 
    genome_index: int = 0
) -> 'Result[AdapterConfig, str]':
    """No more type detection needed!"""
    
    # ✅ CLEAN: Structured access (lazy-loaded)
    evo_cfg = config.evo
    
    # ✅ CLEAN: Dict access  
    adapter_type = config.get('adapter_type', 'lora')
    
    # ✅ DELETE: All the complex validation and type detection!
    
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
    config = load_config("config/test.yaml")  # Same function name
    
    result = compose_ca_pipeline_monadic(self.test_seed, config)
    self.assertTrue(result.is_success())
    
    adapter_config = result.unwrap()
    
    # ✅ CLEAN: No more type detection
    self.assertIn(adapter_config.r, config.evo.rank_candidates)
    self.assertIn(adapter_config.alpha, config.evo.alpha_candidates)
```

## 🎯 **Migration Strategy**

### **Phase 1: Enhance CoralConfig**
1. Modify `CoralConfig` to store `_raw_data` as source of truth
2. Make structured fields lazy-loaded properties
3. Test with existing code (should be compatible)

### **Phase 2: Remove raw_config Parameter**
1. Update `EvolutionEngine` to use single `config` parameter
2. Remove `_get_full_config_dict()` methods everywhere
3. Use `config.serialize_for_modal()` instead

### **Phase 3: Simplify Code**
1. Remove all type detection logic in monadic functions
2. Simplify Modal serialization everywhere
3. Clean up tests

### **Phase 4: Remove Legacy Methods**
1. Remove old dataclass fields from `CoralConfig`
2. Remove `create_config_from_dict()` if no longer needed
3. Update documentation

## 🚀 **Benefits**

### **✅ Keeps Familiar Names**
- `CoralConfig` remains the main config class
- `EvolutionConfig` remains for evolution parameters  
- `load_config()` function name unchanged
- Everyone continues using familiar APIs

### **✅ Eliminates Impedance Mismatch**
- Single source of truth (`_raw_data`)
- No more dual config storage (`cfg + raw_config`)
- No more type conversion methods

### **✅ Maintains Full Compatibility** 
- Dict access still works: `config['evo']`
- Structured access available: `config.evo.rank_candidates`
- Existing code continues working unchanged

### **✅ Performance & Simplicity**
- Lazy loading for better performance
- Remove complex type detection everywhere
- Much simpler Modal serialization
- Cleaner, more maintainable code

This approach enhances what we already have instead of creating new abstractions! 