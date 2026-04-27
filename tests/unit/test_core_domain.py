"""Tests for core domain modules."""

import pytest


def test_adapter_config_import():
    """Test adapter config module imports."""
    from core.domain.adapter_config import AdapterEnvironment, AdapterParameters

    assert AdapterParameters is not None
    assert AdapterEnvironment is not None


def test_mapping_import():
    """Test mapping module imports."""
    from core.domain.mapping import LoRAConfig, map_features_to_lora_config

    assert map_features_to_lora_config is not None
    assert LoRAConfig is not None


def test_experiment_import():
    """Test experiment module imports."""
    from core.domain.experiment import ExperimentConfig, create_experiment_config

    assert create_experiment_config is not None
    assert ExperimentConfig is not None


def test_feature_extraction_import():
    """Test feature extraction imports."""
    from core.domain.feature_extraction import CAFeatures, extract_features

    assert CAFeatures is not None
    assert extract_features is not None


def test_genome_import():
    """Test genome module imports."""
    from core.domain.genome import Genome, MultiObjectiveScores

    assert Genome is not None
    assert MultiObjectiveScores is not None


def test_adapter_parameters_validation():
    """Test adapter parameter validation."""
    from core.domain.adapter_config import AdapterParameters

    # Valid parameters
    params = AdapterParameters(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=("q_proj", "v_proj"),
        adapter_type="lora",
    )
    assert params.rank == 8
    assert params.alpha == 16.0

    # Invalid rank
    with pytest.raises(ValueError, match="Invalid rank"):
        AdapterParameters(
            rank=-1,
            alpha=16.0,
            dropout=0.1,
            target_modules=("q_proj",),
            adapter_type="lora",
        )


def test_multi_objective_scores():
    """Test multi-objective scores creation."""
    from core.domain.genome import MultiObjectiveScores

    scores = MultiObjectiveScores(
        bugfix=0.8, style=0.7, security=0.9, runtime=0.6, syntax=0.85
    )

    assert scores.bugfix == 0.8
    assert scores.overall_fitness() > 0.0

    scores_dict = scores.to_dict()
    assert "bugfix" in scores_dict
    assert scores_dict["bugfix"] == 0.8


def test_tournament_select_returns_unique_survivors_with_numpy_genomes():
    """Tournament survivors should not duplicate genome IDs or ndarray equality."""
    from random import Random

    import numpy as np

    from core.domain.ca import CASeed
    from core.domain.genome import Genome
    from core.domain.mapping import LoRAConfig
    from core.domain.neat import Population, tournament_select

    lora = LoRAConfig(
        r=4,
        alpha=8,
        dropout=0.05,
        target_modules=("q_proj", "v_proj"),
        adapter_type="lora",
    )
    genomes = []
    for index in range(6):
        grid = np.full((2, 2), index % 2, dtype=int)
        genome = Genome(
            seed=CASeed(grid=grid, rule=30 + index, steps=5),
            lora_cfg=lora,
            id=f"genome_{index}",
            fitness=float(index),
        )
        genomes.append(genome)

    selected = tournament_select(Population(tuple(genomes)), 4, rng=Random(42))
    selected_ids = [genome.id for genome in selected.genomes]

    assert len(selected_ids) == 4
    assert len(selected_ids) == len(set(selected_ids))
