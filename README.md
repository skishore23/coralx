# 🪸 CORAL-X: Functional Evolution System

A cost-optimized functional evolution system combining Cellular Automata, LoRA adaptation, and NEAT-style evolution for code generation.

## 🚀 Quick Start

```bash
# Show status and dashboard
./coralx                    # Default rich dashboard
./coralx rich              # Rich dashboard (static)
./coralx live              # Live streaming dashboard
./coralx status            # Simple status check

# Run evolution
python scripts/run_coral_x_evolution.py --config config/main.yaml

# Run benchmarks
python scripts/run_held_out_benchmark.py --config config/main.yaml
```

## 📁 Project Structure

```
coralx/
├── coralx                       # 🎯 Main CLI entry point
├── coral_rich_dashboard.py     # 📊 Primary dashboard (only dashboard in root)
├── coral_modal_app.py          # ☁️  Cost-optimized Modal app (main implementation)
├── config/                     # ⚙️  Configuration files
│   ├── main.yaml              # Main config (cost-optimized)
│   └── coral_x_codellama_config.yaml  # Legacy config
├── scripts/                    # 🧪 Execution scripts
│   ├── run_coral_x_evolution.py     # Main evolution runner
│   ├── run_held_out_benchmark.py    # Held-out benchmark
│   └── run_realtime_benchmarks.py   # Real-time monitoring
├── tools/                      # 🔧 Utility tools
│   ├── real_inference_benchmark.py  # Inference benchmarking
│   └── deploy_*.py                  # Deployment scripts
├── archive/                    # 📦 Archived/unused files
├── coral/                      # 🧮 Core domain logic (pure functions)
├── infra/                      # 🏗️  Infrastructure (Modal, caching)
├── plugins/                    # 🔌 Experiment plugins
└── docs/                       # 📚 Documentation
```

## 💰 Cost Optimization

This system is **cost-optimized** with 60-80% reduction in Modal compute costs:

- **CPU-only functions**: JSON operations use minimal resources (99% cost reduction)
- **A10G for inference**: Code generation uses A10G instead of A100 (50% cost reduction)  
- **Reduced memory**: All functions use right-sized memory allocations (30-50% reduction)
- **Optimized timeouts**: Faster failure detection (50-80% reduction)

## 🧮 Architecture Principles

### Category Theory Foundation
- **Objects**: Immutable data structures (`@dataclass(frozen=True)`)
- **Morphisms**: Pure functions (no side effects)
- **Functors**: Clean boundaries between layers
- **Composition**: Small functions combined into pipelines

### Fail-Fast Philosophy
- **NO fallbacks** - explicit errors over silent failures
- **NO defensive programming** - crash early with clear messages
- **NO hardcoded values** - everything configurable
- **NO mixed concerns** - pure functions stay pure

### Two-Loop Architecture
- **Heavy genes**: LoRA parameters (rank, alpha, dropout) → require training → cached
- **Cheap knobs**: CA-derived parameters (temperature, top_k, etc.) → inference only → recomputed

## 🎯 Key Features

- **Cellular Automata Evolution**: CA features drive generation parameters
- **Cost-Optimized Modal**: Right-sized GPU/CPU resources 
- **Real-time Monitoring**: Live dashboard with progress tracking
- **Held-out Benchmarks**: Separate validation with neutral parameters
- **Cache-Clone System**: 10-60x speedup through intelligent caching
- **Multi-objective Optimization**: Bugfix, style, security, runtime, syntax

## 🧪 Usage Examples

### Basic Evolution Run
```bash
# 20 generations, population 10 (production settings)
python scripts/run_coral_x_evolution.py --config config/main.yaml
```

### Quick Test Run  
```bash
# Modify config/main.yaml:
# execution:
#   generations: 5
#   population_size: 4
python scripts/run_coral_x_evolution.py --config config/main.yaml
```

### Monitoring Progress
```bash
./coralx live          # Live streaming dashboard
./coralx rich          # Static rich dashboard  
./coralx status        # Simple text status
```

### Benchmarking
```bash
# Held-out benchmark (scientifically valid)
python scripts/run_held_out_benchmark.py --config config/main.yaml

# Real-time benchmarking during evolution
python scripts/run_realtime_benchmarks.py --config config/main.yaml
```

## ⚙️ Configuration

Main configuration in `config/main.yaml`:

```yaml
# Cost-optimized Modal resources
infra:
  modal:
    functions:
      generate_code:
        gpu: A10G          # Cost-optimized (vs A100)  
        memory: 8192       # Right-sized
        timeout: 600       # Optimized
      get_progress:
        cpu: 1             # CPU-only for JSON
        memory: 512        # Minimal
        timeout: 30        # Fast failure
```

## 🚀 Deployment

The system uses Modal for distributed execution:

```bash
# Deploy Modal app
modal deploy coral_modal_app.py

# Verify deployment  
modal app list | grep coral-x-production
```

## 📊 Results

Evolution results are saved to:
- `results/evolution_results_[timestamp].json` - Main results
- `results/genetic_tracking/` - Crossover/mutation data  
- `results/realtime_benchmarks/` - Real-time monitoring

## 🔧 Development

### Adding New Experiments
1. Create plugin in `plugins/my_experiment/`
2. Implement required interfaces
3. Add configuration section
4. Test locally before Modal deployment

### Modifying CA Features
1. Edit `coral/domain/ca.py` - pure functions only
2. Update feature extraction in `coral/domain/feature_extraction.py`
3. Modify mapping in `coral/domain/mapping.py`

### Cost Optimization
- Monitor costs in Modal dashboard
- Adjust resource allocations in `config/main.yaml`
- Use CPU-only functions for non-ML operations

---
