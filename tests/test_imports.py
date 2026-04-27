"""Test that all core modules can be imported without errors."""

import sys
from pathlib import Path

import pytest

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_core_imports():
    """Test that core modules can be imported."""
    # Test domain imports

    # Test common imports

    # Test services imports

    # Test application imports

    # Test infrastructure imports

    # Test CLI imports

    # Test plugin imports


def test_config_validation():
    """Test that configuration validation works."""
    from core.common.config import CoralConfig

    # Test valid configuration
    valid_config = {
        "execution": {
            "generations": 5,
            "population_size": 10,
            "output_dir": "./results",
            "selection_mode": "pareto",
            "survival_rate": 0.5,
            "crossover_rate": 0.7,
        },
        "evo": {
            "rank_candidates": [4, 8, 16],
            "alpha_candidates": [8, 16, 32],
            "dropout_candidates": [0.05, 0.1, 0.15],
            "target_modules": ["q_proj", "v_proj"],
        },
        "experiment": {
            "target": "quixbugs_mini",
            "name": "test_experiment",
            "dataset": {
                "path": "./datasets",
                "max_samples": 100,
                "datasets": ["quixbugs_mini"],
            },
            "model": {
                "name": "mock_model_for_tiny_run",
                "max_seq_length": 512,
            },
        },
        "evaluation": {
            "test_samples": 10,
            "fitness_weights": {
                "bugfix": 0.3,
                "style": 0.15,
                "security": 0.25,
                "runtime": 0.1,
                "syntax": 0.2,
            },
        },
        "infra": {"executor": "local"},
        "cache": {
            "artifacts_dir": "./cache",
            "base_checkpoint": "mock_model_for_tiny_run",
        },
        "threshold": {
            "base_thresholds": {
                "bugfix": 0.1,
                "style": 0.1,
                "security": 0.1,
                "runtime": 0.1,
                "syntax": 0.1,
            },
            "max_thresholds": {
                "bugfix": 0.8,
                "style": 0.8,
                "security": 0.8,
                "runtime": 0.8,
                "syntax": 0.8,
            },
        },
        "seed": 42,
    }

    # Should not raise any exceptions
    config = CoralConfig(**valid_config)
    assert config.seed == 42
    assert config.execution.generations == 5
    assert config.experiment.name == "test_experiment"


def test_domain_objects():
    """Test that domain objects can be created."""
    import numpy as np

    from core.domain.ca import CASeed
    from core.domain.genome import Genome, MultiObjectiveScores
    from core.domain.mapping import LoRAConfig

    # Test CA seed creation
    grid = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    ca_seed = CASeed(grid=grid, rule=30, steps=5)
    assert ca_seed.rule == 30
    assert ca_seed.steps == 5

    # Test LoRA config creation
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=("q_proj", "v_proj"),
        adapter_type="lora",
    )
    assert lora_config.r == 8
    assert lora_config.alpha == 16

    # Test multi-objective scores
    scores = MultiObjectiveScores(
        bugfix=0.8, style=0.6, security=0.9, runtime=0.7, syntax=0.85
    )
    assert scores.overall_fitness() > 0.0

    # Test genome creation
    genome = Genome(
        seed=ca_seed,
        lora_cfg=lora_config,
        id="test_genome",
        fitness=0.75,
        multi_scores=scores,
    )
    assert genome.id == "test_genome"
    assert genome.fitness == 0.75
    assert genome.is_evaluated() is True


def test_executor_creation():
    """Test that executors can be created."""
    from infra.executors.local import LocalExecutor, LocalExecutorConfig

    # Test local executor creation
    config = LocalExecutorConfig(max_workers=2, default_timeout=60.0)
    executor = LocalExecutor(config)

    assert executor.is_available is True
    assert executor.config.max_workers == 2
    result = executor.submit(lambda value: value + 1, 41)
    assert result.is_successful()
    assert result.result == 42

    # Test context manager
    with LocalExecutor() as exec:
        assert exec.is_available is True

    # Cleanup
    executor.shutdown()


def test_plugin_imports():
    """Test that plugins can be imported and initialized."""
    from plugins.quixbugs_mini.plugin import QuixBugsMiniPlugin

    # Test plugin initialization with minimal config
    plugin_config = {
        "experiment": {
            "name": "test_experiment",
            "target": "quixbugs_mini",
            "dataset": {
                "path": "./datasets",
                "max_samples": 3,
                "datasets": ["quixbugs_mini"],
            },
            "model": {
                "name": "mock_model_for_tiny_run",
                "max_seq_length": 512,
            },
        },
        "dataset": {
            "dataset_path": "./datasets",
            "max_samples": 3,
            "datasets": ["quixbugs_mini"],
        },
        "model": {
            "model_name": "mock_model_for_tiny_run",
            "max_seq_length": 512,
        },
        "evaluation": {"test_samples": 2},
    }

    plugin = QuixBugsMiniPlugin(plugin_config)
    assert plugin is not None

    # Test that plugin components can be created
    model_factory = plugin.model_factory()
    dataset_provider = plugin.dataset()
    fitness_fn = plugin.fitness_fn()

    assert model_factory is not None
    assert dataset_provider is not None
    assert fitness_fn is not None


if __name__ == "__main__":
    pytest.main([__file__])
