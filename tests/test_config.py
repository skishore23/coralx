"""Test configuration validation and loading."""

import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError

from core.common.config import (
    CoralConfig, ExecutionConfig, EvolutionConfig, ObjectiveThresholds, FitnessWeights
)
from core.common.config_loader import load_config, ConfigManager
from core.common.exceptions import ConfigurationError


def test_execution_config_validation():
    """Test ExecutionConfig validation."""
    # Valid config
    valid_config = {
        "generations": 10,
        "population_size": 50,
        "output_dir": "./results",
        "selection_mode": "pareto",
        "survival_rate": 0.5,
        "crossover_rate": 0.7
    }

    config = ExecutionConfig(**valid_config)
    assert config.generations == 10
    assert config.population_size == 50
    assert config.selection_mode.value == "pareto"

    # Invalid config - negative generations
    with pytest.raises(ValidationError):
        ExecutionConfig(generations=-1, population_size=50, output_dir="./results")

    # Invalid config - invalid selection mode
    with pytest.raises(ValidationError):
        ExecutionConfig(
            generations=10,
            population_size=50,
            output_dir="./results",
            selection_mode="invalid_mode"
        )


def test_evolution_config_validation():
    """Test EvolutionConfig validation."""
    # Valid config
    valid_config = {
        "rank_candidates": [4, 8, 16, 32],
        "alpha_candidates": [8, 16, 32],
        "dropout_candidates": [0.05, 0.1, 0.15],
        "target_modules": ["q_proj", "v_proj"]
    }

    config = EvolutionConfig(**valid_config)
    assert config.rank_candidates == [4, 8, 16, 32]
    assert config.alpha_candidates == [8, 16, 32]

    # Invalid config - negative rank candidates
    with pytest.raises(ValidationError):
        EvolutionConfig(
            rank_candidates=[-1, 8],
            alpha_candidates=[8, 16],
            dropout_candidates=[0.05, 0.1],
            target_modules=["q_proj", "v_proj"]
        )

    # Invalid config - dropout out of range
    with pytest.raises(ValidationError):
        EvolutionConfig(
            rank_candidates=[4, 8],
            alpha_candidates=[8, 16],
            dropout_candidates=[1.5, 0.1],  # 1.5 > 1.0
            target_modules=["q_proj", "v_proj"]
        )


def test_fitness_weights_validation():
    """Test FitnessWeights validation."""
    # Valid config
    valid_weights = {
        "bugfix": 0.3,
        "style": 0.15,
        "security": 0.25,
        "runtime": 0.1,
        "syntax": 0.2
    }

    weights = FitnessWeights(**valid_weights)
    assert weights.bugfix == 0.3
    # Test that weights sum to 1.0 (validated by Pydantic)
    total = weights.bugfix + weights.style + weights.security + weights.runtime + weights.syntax
    assert abs(total - 1.0) < 0.01

    # Invalid config - weights don't sum to 1.0
    with pytest.raises(ValidationError):
        FitnessWeights(
            bugfix=0.5,
            style=0.5,
            security=0.5,
            runtime=0.5,
            syntax=0.5
        )


def test_objective_thresholds_validation():
    """Test ObjectiveThresholds validation."""
    # Valid config
    valid_thresholds = {
        "bugfix": 0.1,
        "style": 0.1,
        "security": 0.1,
        "runtime": 0.1,
        "syntax": 0.1
    }

    thresholds = ObjectiveThresholds(**valid_thresholds)
    assert thresholds.bugfix == 0.1

    # Test to_dict method
    threshold_dict = thresholds.to_dict()
    assert threshold_dict["bugfix"] == 0.1
    assert len(threshold_dict) == 5


def test_complete_config_validation():
    """Test complete CoralConfig validation."""
    # Valid complete config
    valid_config = {
        "execution": {
            "generations": 5,
            "population_size": 10,
            "output_dir": "./results",
            "selection_mode": "pareto",
            "survival_rate": 0.5,
            "crossover_rate": 0.7
        },
        "evo": {
            "rank_candidates": [4, 8, 16],
            "alpha_candidates": [8, 16, 32],
            "dropout_candidates": [0.05, 0.1, 0.15],
            "target_modules": ["q_proj", "v_proj"]
        },
        "experiment": {
            "target": "fakenews_tinyllama",
            "name": "test_experiment",
            "dataset": {
                "path": "./datasets",
                "max_samples": 100,
                "datasets": ["fake_news"]
            },
            "model": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_seq_length": 512
            }
        },
        "evaluation": {
            "test_samples": 10,
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
            "artifacts_dir": "./cache",
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

    config = CoralConfig(**valid_config)
    assert config.seed == 42
    assert config.execution.generations == 5
    assert config.experiment.name == "test_experiment"
    assert config.infra.executor.value == "local"


def test_config_loader():
    """Test configuration loading from file."""
    # Create a temporary config file
    config_data = {
        "execution": {
            "generations": 3,
            "population_size": 8,
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
            "name": "loader_test",
            "dataset": {
                "path": "./datasets",
                "max_samples": 50,
                "datasets": ["fake_news"]
            },
            "model": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_seq_length": 512
            }
        },
        "evaluation": {
            "test_samples": 5,
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
        "seed": 123
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = Path(f.name)

    try:
        # Test loading config
        config = load_config(temp_config_path)
        assert config.seed == 123
        assert config.execution.generations == 3
        assert config.experiment.name == "loader_test"
        assert config.infra.executor.value == "local"

    finally:
        temp_config_path.unlink(missing_ok=True)


def test_config_loader_missing_file():
    """Test configuration loading with missing file."""
    with pytest.raises(ConfigurationError):
        load_config(Path("nonexistent_config.yaml"))


def test_config_loader_invalid_yaml():
    """Test configuration loading with invalid YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_config_path = Path(f.name)

    try:
        with pytest.raises(ConfigurationError):
            load_config(temp_config_path)
    finally:
        temp_config_path.unlink(missing_ok=True)


def test_config_manager():
    """Test ConfigManager functionality."""
    # Create a temporary config file
    config_data = {
        "execution": {
            "generations": 2,
            "population_size": 4,
            "output_dir": "./results/manager_test",
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
            "name": "manager_test",
            "dataset": {
                "path": "./datasets",
                "max_samples": 20,
                "datasets": ["fake_news"]
            },
            "model": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_seq_length": 512
            }
        },
        "evaluation": {
            "test_samples": 3,
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
            "artifacts_dir": "./cache/manager_test",
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
        "seed": 456
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = Path(f.name)

    try:
        # Test ConfigManager
        manager = ConfigManager("test")
        config = manager.load_config(temp_config_path)

        assert config.seed == 456
        assert config.execution.generations == 2
        assert config.experiment.name == "manager_test"

    finally:
        temp_config_path.unlink(missing_ok=True)


def test_environment_overrides():
    """Test environment variable overrides."""
    import os

    # Set environment variables
    os.environ["CORAL_POPULATION_SIZE"] = "20"
    os.environ["CORAL_GENERATIONS"] = "5"
    os.environ["CORAL_SEED"] = "999"

    try:
        # Create a base config
        config_data = {
            "execution": {
                "generations": 1,
                "population_size": 10,
                "output_dir": "./results",
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
                "name": "env_test",
                "dataset": {
                    "path": "./datasets",
                    "max_samples": 20,
                    "datasets": ["fake_news"]
                },
                "model": {
                    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "max_seq_length": 512
                }
            },
            "evaluation": {
                "test_samples": 5,
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
                "artifacts_dir": "./cache/env_test",
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

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = Path(f.name)

        try:
            # Test that environment overrides work
            manager = ConfigManager("test")
            config = manager.load_config(temp_config_path)

            # Environment variables should override base config
            assert config.execution.population_size == 20
            assert config.execution.generations == 5
            assert config.seed == 999

        finally:
            temp_config_path.unlink(missing_ok=True)

    finally:
        # Clean up environment variables
        for key in ["CORAL_POPULATION_SIZE", "CORAL_GENERATIONS", "CORAL_SEED"]:
            os.environ.pop(key, None)


if __name__ == "__main__":
    pytest.main([__file__])
