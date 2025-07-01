# CORAL-X Evolution Dashboard

Real-time monitoring dashboard for CORAL-X evolution experiments following category theory principles.

## Quick Start

```bash
# Install dependencies
pip install rich

# Run live dashboard
python cli/dashboard.py

# Or use the launcher script
./coral_dashboard

# Snapshot mode (single view)
python cli/dashboard.py --snapshot

# Custom config
python cli/dashboard.py -c my_config.yaml

# Simple text mode (no Rich)
python cli/dashboard.py --simple
```

## Features

The dashboard displays real-time information about your CORAL-X evolution:

### 🧬 Evolution Progress
- Current generation and completion percentage
- Population size and runtime
- Progress bar visualization

### 📊 Performance Metrics  
- Multi-objective scores (Bugfix, Style, Security, Runtime, Syntax)
- Letter grades (A+ to D)
- Visual score bars

### 🎛️ DoRA Adapters
- Adapters trained count
- Cache hit rate and training rate
- Parameter space coverage

### 🌊 CA Metrics
- Complexity, Intensity, Periodicity, Convergence
- Population-wide averages

### 🧬 Genetic Operations
- Crossover and mutation counts
- Success rates for genetic operations

### 🎛️ Cheap Knobs Configuration
- Temperature, Top-P, Top-K ranges
- Real-time parameter monitoring

### 🌟 Emergent Behavior
- Detection status and behavior count
- Detection rate and recent behaviors

### 💻 Infrastructure
- Models and dataset cache status
- Modal app and volume status
- GPU monitoring

## Architecture

The dashboard follows CORAL-X's category theory principles:

- **Domain Layer**: Pure functions for state calculations (`coral/domain/dashboard_state.py`)
- **Application Layer**: Orchestration logic (`coral/application/dashboard_orchestrator.py`) 
- **Ports Layer**: Abstract interfaces (`coral/ports/dashboard_interfaces.py`)
- **Infrastructure Layer**: File I/O and Rich rendering (`infra/dashboard/`)
- **CLI Layer**: User interface (`cli/dashboard.py`)

## Configuration

The dashboard reads from your existing CORAL-X configuration file and monitoring outputs:

- Progress tracking files
- Genetic operations tracking
- Emergent behavior logs
- Modal volume status
- Configuration parameters

## Data Sources

The dashboard automatically detects and reads from:

- Evolution progress files
- Genetic tracking directory  
- Modal volume contents
- Emergent behavior logs
- Configuration files

All data collection follows the fail-fast principle with no fallback defaults.

## Troubleshooting

```bash
# Test components
python cli/dashboard.py --test

# Use simple mode if Rich is unavailable
python cli/dashboard.py --simple

# Debug with verbose output
python cli/dashboard.py --snapshot
```

The dashboard will automatically fall back to simple text mode if Rich is not installed. 