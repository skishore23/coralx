"""Test CLI functionality."""

import pytest
import subprocess
import sys
from pathlib import Path
import tempfile
import yaml

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_cli_help():
    """Test that CLI shows help without errors."""
    result = subprocess.run([
        sys.executable, "-m", "core.cli.main", "--help"
    ], capture_output=True, text=True, cwd=project_root)

    assert result.returncode == 0
    assert "CORAL-X Evolution Framework" in result.stdout
    assert "Commands" in result.stdout


def test_cli_run_help():
    """Test that run command shows help."""
    result = subprocess.run([
        sys.executable, "-m", "core.cli.main", "run", "--help"
    ], capture_output=True, text=True, cwd=project_root)

    assert result.returncode == 0
    assert "Config file path" in result.stdout
    assert "--config" in result.stdout


def test_cli_invalid_command():
    """Test that invalid commands show appropriate error."""
    result = subprocess.run([
        sys.executable, "-m", "core.cli.main", "invalid_command"
    ], capture_output=True, text=True, cwd=project_root)

    assert result.returncode == 2  # argparse returns 2 for invalid arguments


def test_cli_missing_config():
    """Test that missing config file shows appropriate error."""
    result = subprocess.run([
        sys.executable, "-m", "core.cli.main", "run", "--config", "nonexistent.yaml"
    ], capture_output=True, text=True, cwd=project_root)

    assert result.returncode == 1
    assert "Config file not found" in result.stdout


def test_cli_invalid_config():
    """Test that invalid config file shows appropriate error."""
    # Create a temporary invalid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_config = f.name

    try:
        result = subprocess.run([
            sys.executable, "-m", "core.cli.main", "run", "--config", temp_config
        ], capture_output=True, text=True, cwd=project_root)

        assert result.returncode == 1
        assert "Error" in result.stdout
    finally:
        Path(temp_config).unlink(missing_ok=True)


def test_cli_valid_config_parsing():
    """Test that valid config file can be parsed."""
    # Create a minimal valid config
    valid_config = {
        "execution": {
            "generations": 1,
            "population_size": 2,
            "output_dir": "./results/test",
            "selection_mode": "pareto",
            "survival_rate": 0.5,
            "crossover_rate": 0.7
        },
        "evo": {
            "rank_candidates": [4, 8],
            "alpha_candidates": [8, 16],
            "dropout_candidates": [0.05, 0.1],
            "target_modules": ["q_proj", "v_proj"]
        },
        "experiment": {
            "target": "fakenews_tinyllama",
            "name": "test_parsing",
            "dataset": {
                "path": "./datasets",
                "max_samples": 5,
                "datasets": ["fake_news"]
            },
            "model": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_seq_length": 512
            }
        },
        "evaluation": {
            "test_samples": 2,
            "fitness_weights": {
                "bugfix": 0.3,
                "style": 0.15,
                "security": 0.25,
                "runtime": 0.1,
                "syntax": 0.2
            }
        },
        "infra": {
            "executor": "local"
        },
        "cache": {
            "artifacts_dir": "./cache/test",
            "base_checkpoint": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        },
        "threshold": {
            "base_thresholds": {
                "bugfix": 0.1,
                "style": 0.1,
                "security": 0.1,
                "runtime": 0.1,
                "syntax": 0.1
            },
            "max_thresholds": {
                "bugfix": 0.8,
                "style": 0.8,
                "security": 0.8,
                "runtime": 0.8,
                "syntax": 0.8
            }
        },
        "seed": 42
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        temp_config = f.name

    try:
        # Test that config can be loaded without errors
        result = subprocess.run([
            sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{project_root}')
from core.common.config_loader import load_config
from pathlib import Path
config = load_config(Path('{temp_config}'))
print(f"Config loaded: {{config.experiment.name}}")
print(f"Generations: {{config.execution.generations}}")
print(f"Population size: {{config.execution.population_size}}")
print(f"Executor: {{config.infra.executor}}")
"""
        ], capture_output=True, text=True, cwd=project_root)

        assert result.returncode == 0
        assert "Config loaded: test_parsing" in result.stdout
        assert "Generations: 1" in result.stdout
        assert "Population size: 2" in result.stdout
        assert "Executor: ExecutorType.LOCAL" in result.stdout

    finally:
        Path(temp_config).unlink(missing_ok=True)


def test_cli_dry_run():
    """Test that CLI can perform a dry run without full execution."""
    # Use the existing smoke config but with very minimal settings
    smoke_config_path = project_root / "config" / "examples" / "smoke.yaml"

    if not smoke_config_path.exists():
        pytest.skip("Smoke config not found")

    # Test that the CLI can at least start the experiment
    # We'll use a timeout to prevent it from running too long
    try:
        result = subprocess.run([
            sys.executable, "-m", "core.cli.main", "run", "--config", str(smoke_config_path)
        ], capture_output=True, text=True, cwd=project_root, timeout=10)  # 10 second timeout

        # The command might timeout or complete, but it should not fail with import errors
        # We're mainly testing that the CLI can start and parse the config
        assert "Loading config" in result.stdout or "Starting experiment" in result.stdout
    except subprocess.TimeoutExpired:
        # Timeout is expected for this test - we just want to make sure it starts
        pass


if __name__ == "__main__":
    pytest.main([__file__])
