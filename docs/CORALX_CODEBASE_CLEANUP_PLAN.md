# CORAL-X Codebase Cleanup Plan

**Goal**: Clean, maintainable codebase with 40% size reduction while preserving all functionality  
**Impact**: ~45 files deleted, ~300KB+ code removed, ~150MB+ artifacts cleaned

---

## 🚀 **PHASE 1: IMMEDIATE CLEANUP (Low Risk)**

### ✅ **Step 1.1: Reorganize Documentation (Smart Cleanup)**

**CONTEXT**: Future roadmap includes category theory refactoring, Modal queues, dynamic meta-optimization, and multi-LLM/dataset work.

#### **🗂️ Create Directory Structure**
```bash
mkdir -p docs/{active,reference,archive}
```

#### **📋 KEEP (Essential for Roadmap)** - Move to `docs/active/`
```bash
# Category Theory Refactoring (PRIORITY 1)
mv CATEGORY_THEORY_ANALYSIS.md docs/active/          # 27KB, June 21 - Essential CT analysis
mv ct.md docs/active/category_theory_reference.md    # 3KB, June 21 - CT reference (rename)

# Modal Queues Implementation (PRIORITY 1) 
mv COMPREHENSIVE_QUEUE_REFACTORING_PLAN.md docs/active/    # 40KB, June 21 - Complete implementation
mv CORAL_X_QUEUE_REFACTORING_PLAN.md docs/active/         # 13KB, June 21 - Queue plan
mv MODAL_INFRASTRUCTURE_ARCHITECTURE.md docs/active/      # 43KB, June 21 - Modal architecture  
mv DETAILED_IMPLEMENTATION_MAP.md docs/active/            # 27KB, June 21 - Implementation guide

# Dynamic Meta-Optimization (PRIORITY 2)
mv CORAL_X_DYNAMIC_META_OPTIMIZATION.md docs/active/      # 18KB, June 20 - Meta-optimization guide

# Current Analysis (PRIORITY 2)
mv CORAL_X_ARCHITECTURE_ANALYSIS.md docs/active/         # 14KB, June 21 - Recent analysis
```

#### **📚 REFERENCE (Useful but not critical)** - Move to `docs/reference/`
```bash
# Overview documentation  
mv CORAL_X_HOW_IT_WORKS.md docs/reference/               # 18KB, June 20 - Good overview
mv EMERGENT_BEHAVIOR_DETECTION_GUIDE.md docs/reference/  # 12KB, June 20 - Emergent behavior guide
mv SIMPLE_EMERGENT_TRACKING_GUIDE.md docs/reference/     # 5KB, June 20 - Simple guide
```

#### **🗄️ ARCHIVE (Superseded/Outdated)** - Move to `docs/archive/`
```bash
# Superseded by newer analysis
mv CORAL_X_ARCHITECTURE.md docs/archive/                 # 40KB, June 21 19:05 - Superseded by analysis

# Less relevant for multi-dataset future  
mv QUIXBUGS_USAGE_ANALYSIS.md docs/archive/              # 14KB, June 21 00:11 - Single dataset focus
```

#### **🗑️ DELETE (Consider removing after archiving)**
```bash
# After confirming content is captured elsewhere:
# rm docs/archive/CORAL_X_ARCHITECTURE.md              # Content captured in analysis doc
# rm docs/archive/QUIXBUGS_USAGE_ANALYSIS.md           # Less relevant for multi-dataset roadmap
```

#### **📋 Documentation Organization Strategy**

**🎯 RATIONALE**: Organized by **relevance to future roadmap** rather than creation date:

1. **`docs/active/`** - **Essential for current roadmap**
   - **Category Theory Refactoring**: Direct implementation guides
   - **Modal Queues**: Complete plans and architecture 
   - **Dynamic Meta-Optimization**: Next evolution phase
   - **Current Analysis**: Recent architectural insights

2. **`docs/reference/`** - **Useful but not blocking**
   - System overviews and educational content
   - Emergent behavior guides (useful for any dataset)
   - Implementation references

3. **`docs/archive/`** - **Historical/Superseded**
   - Older architecture docs (superseded by analysis)
   - Single-dataset focused docs (multi-dataset future)

**🔄 FUTURE**: As you implement category theory → Modal queues → dynamic meta-optimization, documentation will naturally flow: `active/` → `reference/` → `archive/`

**📊 SIZE IMPACT**:
- **Active docs**: 147KB (essential roadmap docs)
- **Reference docs**: 35KB (useful references)  
- **Archive docs**: 54KB (superseded content)
- **Potential deletion**: 54KB after validation

### ✅ **Step 1.2: Delete Cache Artifacts**
```bash
# Python cache (regenerable)
rm -rf __pycache__/

# Test artifacts (regenerable)
rm -rf test_output/

# WandB logs (optional - keep if analyzing past runs)
# rm -rf wandb/
```

### ✅ **Step 1.3: Delete Backup/Legacy Files**
```bash
rm coral_x_codellama_config.yaml.backup
rm benchmark_verification_1750558653.json
rm coral_x_with_auto_benchmark_config.yaml    # Only 18 lines - merge into main
rm test_coral_x_config.yaml                   # Test-only config
```

### ✅ **Step 1.4: Delete Test/Debug Files** (20 files)
```bash
# Debug/reproduction scripts
rm test_cache_coordination_debug.py          # 22KB - debugging only
rm reproduce_cache_error.py                  # 12KB - reproduction script
rm test_cache_coordination.py                # 2.7KB - basic test
rm test_modal_volume_race_condition.py       # 14KB - race condition testing
rm quick_race_condition_test.py              # 2.9KB - quick test

# Integration tests
rm test_integration_fix.py                   # 11KB - integration testing
rm test_benchmark_architecture_claims.py     # 14KB - architecture validation
rm test_local_generation.py                  # 7.9KB - local testing
rm test_lora_training_debug.py               # 4.4KB - LoRA debugging
rm test_genome_reconstruction.py             # 4.7KB - genome testing

# Pipeline tests
rm test_quick_ca_pipeline.py                 # 3.6KB - pipeline testing
rm test_cheap_knobs_fix.py                   # 11KB - knobs testing
rm test_emergent_tracking.py                 # 5.8KB - emergent testing
rm test_genetic_tracking.py                  # 7.6KB - genetic tracking
rm test_genome_ids.py                        # 7.1KB - genome ID testing

# Parameter/config tests
rm test_parameter_loading.py                 # 4.1KB - parameter testing
rm test_real_benchmark.py                    # 4.0KB - benchmark testing
rm test_coral_x_quick.py                     # 6.5KB - quick testing
rm test_cache_mode.py                        # 2.7KB - cache testing
rm test_lora_local.py                        # 8.1KB - local LoRA testing
```

### ✅ **Step 1.5: Delete Utility Scripts** (3 files)
```bash
rm analyze_cache_groups.py                   # 4.9KB - cache analysis
rm monitor_coral_evolution.py                # 4.0KB - monitoring script  
rm quick_real_benchmark.py                   # 5.6KB - quick benchmark
```

### ✅ **Step 1.6: Delete Duplicate Modal Apps** (2 files)
```bash
rm coral_modal_benchmark_app.py              # 7.3KB - duplicate functionality
rm run_modal_benchmark.py                    # 4.8KB - simple benchmark runner
```

---

## 🔧 **PHASE 2: CODE CLEANUP (Medium Risk)**

### ✅ **Step 2.1: Remove Unused Modal Functions**

**File**: `coral_modal_app.py`

**DELETE these functions** (7 functions, ~500 lines):

1. **test_modal_functions()** (Lines ~1481-1488)
   - @app.local_entrypoint() for testing deployed functions

2. **test_cache_coordination_modal()** (Lines ~2058-2150) 
   - @app.function() for cache debugging

3. **test_adapter_functionality()** (Lines ~1628-1761)
   - @app.function() for adapter testing

4. **test_emergent_behavior_tracking()** (Lines ~1855-1932)
   - @app.function() for emergent behavior testing

5. **check_dora_availability()** (Lines ~1806-1854)
   - @app.function() for PEFT version checking

6. **cleanup_corrupted_adapters()** (Lines ~1491-1617)
   - @app.function() for one-time maintenance

7. **clear_all_adapters()** (Lines ~1762-1805)
   - @app.function() for emergency cache clearing

8. **_get_dir_size_mb()** (Helper function)
   - Used only by cleanup functions

### ✅ **Step 2.2: Remove Deprecated Functions**

**File**: `adapters/quixbugs_real.py`

**DELETE**: `threshold_gate()` function (Lines 479-541, 63 lines)
```python
def threshold_gate(scores: QuixBugsMetrics, sigma: float, thresholds_config: Dict[str, Any]) -> bool:
    """Apply threshold gate with σ-wave dynamics."""
    # DEPRECATED: System now uses NSGA-II Pareto selection
    # This function is no longer called in current evolution pipeline
```

**File**: `coral/domain/ca.py`

**DELETE**: `_apply_rule()` function (if exists and marked deprecated)
```python
def _apply_rule(grid: np.ndarray, rule: int) -> np.ndarray:
    """DEPRECATED: Original broken implementation."""
```

**File**: `run_coral_x_evolution.py`

**DELETE**: `_run_post_evolution_benchmarks()` function (Lines 770-838)
```python
def _run_post_evolution_benchmarks(result, config):
    """DEPRECATED: Post-evolution benchmarking removed."""
```

### ✅ **Step 2.3: Clean Legacy Code Patterns**

**File**: `adapters/quixbugs_real.py`

**DELETE**: Legacy dataset paths (Lines 38-39)
```python
"/data/quixbugs_dataset",            # Legacy path
"../quixbugs_dataset",               # Legacy relative
```

**File**: `reproduce_cache_error.py`

**DELETE**: Old serialization simulation section (Lines 64-152, 88+ lines)
```python
# Phase 2: Simulate OLD serialization/deserialization (BROKEN)
# ... entire section that reproduces fixed bugs
```

### ✅ **Step 2.4: Review Legacy Cache Mode**

**File**: `cli/coral.py`

**REVIEW**: Check if legacy cache mode is still needed (Lines 311, 322)
```python
print(f"🔧 Using cache-friendly population (legacy mode)")
def create_population_with_cache_groups(config):
    """Create population with pre-assigned cache groups (legacy mode)."""
```

---

## 📁 **PHASE 3: CONFIG CONSOLIDATION (Careful)**

### ✅ **Step 3.1: Analyze Config Overlap**

**Current configs** (9 remaining):
- `coral_x_codellama_config.yaml` (Main - 8.7KB)
- `coral_x_codellama_config_benchmark.yaml` (Benchmark - 5.7KB)  
- `coral_x_clean_config.yaml` (Clean - 3.9KB)
- `coral_x_modal_config.yaml` (Modal-specific - 5.8KB)
- `coral_x_real_config.yaml` (Real evaluation - 1.7KB)
- `coral_x_scale_config.yaml` (Scale - 3.6KB)
- `coral_x_test_config.yaml` (Test - 4.4KB)
- `coral_x_emergent_config.yaml` (Emergent - 1.3KB)

### ✅ **Step 3.2: Plan Consolidation**

**Target structure**:
1. `coral_x_main_config.yaml` - Core settings with conditional sections
2. `coral_x_development_config.yaml` - Development/testing overrides  
3. `coral_x_benchmark_config.yaml` - Benchmark-specific (if needed)

---

## 🎯 **PHASE 4: VALIDATION**

### ✅ **Step 4.1: Test Core Functionality**
```bash
# Test evolution pipeline
python run_coral_x_evolution.py --config coral_x_codellama_config.yaml --quick-test

# Test Modal deployment  
modal deploy coral_modal_app.py::app --name coral-x-production
```

### ✅ **Step 4.2: Verify Modal Functions**
```bash
# Check deployed functions
modal app list
modal function list coral-x-production
```

### ✅ **Step 4.3: Run Integration Test**
```bash
# Quick integration test
python -c "
import coral
from coral.application.evolution_engine import EvolutionEngine
print('✅ Core imports working')
"
```

---

## 📊 **CLEANUP SUMMARY**

### **Files Organized/Deleted by Phase**
- **Phase 1**: 45 files deleted + docs organized (~250KB code + 150MB artifacts)
  - **Smart docs organization**: 13 files → structured by roadmap relevance
  - **Active docs**: 8 files (147KB) - essential for category theory/queues/meta-optimization
  - **Archive/delete**: 2-3 files (54KB) - superseded content
- **Phase 2**: 7-9 Modal functions (~500 lines)  
- **Phase 3**: 2-4 config files (consolidation)

### **Total Impact**
- **40% smaller codebase** with **organized documentation**
- **Faster Modal deployments** (fewer functions)
- **Cleaner architecture** aligned with roadmap
- **Better maintainability** and **easier navigation**
- **Documentation strategy** supports category theory → queues → meta-optimization progression

### **Risk Mitigation**
- ✅ **All test files** - Safe to delete (not imported by production)
- ✅ **Cache artifacts** - Regenerable
- ✅ **Deprecated functions** - Already marked for removal
- ⚠️ **Config consolidation** - Requires testing
- ⚠️ **Legacy code patterns** - Verify not in use

---

## 🚨 **ROLLBACK PLAN**

If issues arise:
```bash
# Restore from git
git checkout HEAD -- <deleted_file>

# Restore Modal functions
git checkout HEAD -- coral_modal_app.py
modal deploy coral_modal_app.py::app --name coral-x-production
```

---

## 🛣️ **FUTURE DOCUMENTATION STRATEGY**

### **Roadmap-Aligned Documentation Maintenance**

Given your roadmap: **Category Theory → Modal Queues → Dynamic Meta-Optimization → Multi-LLM/Dataset**, the documentation structure should evolve:

#### **🔄 Documentation Lifecycle**
```
PLANNING → ACTIVE → REFERENCE → ARCHIVE → DELETE
```

#### **📋 Ongoing Strategy**
```bash
# As you implement each phase:

# 1. CATEGORY THEORY REFACTORING
# Active: CATEGORY_THEORY_ANALYSIS.md, category_theory_reference.md
# Create: category_theory_implementation_log.md (track progress)
# Archive: Old architecture docs after CT refactoring

# 2. MODAL QUEUES IMPLEMENTATION  
# Active: COMPREHENSIVE_QUEUE_REFACTORING_PLAN.md, MODAL_INFRASTRUCTURE_ARCHITECTURE.md
# Create: modal_queues_implementation_log.md (track progress)  
# Archive: Old Modal infrastructure docs after queue implementation

# 3. DYNAMIC META-OPTIMIZATION
# Active: CORAL_X_DYNAMIC_META_OPTIMIZATION.md
# Create: meta_optimization_experiments.md (track LLM/dataset experiments)
# Reference: Implementation guides become references

# 4. MULTI-LLM/DATASET EXPANSION
# Create: multi_llm_integration_guide.md, dataset_expansion_strategy.md
# Archive: Single-dataset focused documentation
```

#### **🎯 Documentation Principles Going Forward**
1. **Living docs in `active/`** - Update as you implement
2. **Implementation logs** - Track what works/doesn't work  
3. **Progressive archiving** - Move completed implementations to `reference/`
4. **Delete superseded content** - Keep codebase lean
5. **Roadmap alignment** - Docs should support current work phase

This cleanup creates a **maintainable documentation foundation** for your ambitious roadmap while eliminating current bloat.

---

**Execute phases sequentially. Test after each phase before proceeding.** 