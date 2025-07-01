# Configuration Guide

## Main Configuration

**Primary config:** `coral_x_codellama_config.yaml` (in root directory)  
This is the main configuration file with all settings.

## Usage

```bash
# Main evolution
python run_coral_x_evolution.py --config coral_x_codellama_config.yaml

# Held-out benchmark  
python run_held_out_benchmark.py --config coral_x_codellama_config.yaml

# Real-time benchmark
python run_realtime_benchmarks.py
```

## Test Case Management

All test cases are centrally managed in `coral/domain/dataset_constants.py`:

- **`QUIXBUGS_TRAINING_PROBLEMS`** - Used for LoRA training (24 problems)
- **`QUIXBUGS_CLEAN_TEST_PROBLEMS`** - Used for evaluation (8 problems)  

**No hardcoding!** All components should import from `dataset_constants.py`

## Config Backup

A clean copy is available here: `config/main.yaml` (same as root config) 