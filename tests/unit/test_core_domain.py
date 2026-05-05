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
    """Test target-neutral multi-objective scores creation."""
    from core.domain.genome import MultiObjectiveScores

    scores = MultiObjectiveScores(
        task_score=0.8,
        quality_score=0.7,
        risk_score=0.9,
        efficiency_score=0.6,
        validity_score=0.85,
    )

    assert scores.task_score == 0.8
    assert scores.overall_fitness() > 0.0

    scores_dict = scores.to_dict()
    assert "task_score" in scores_dict
    assert "bugfix" not in scores_dict
    assert scores_dict["task_score"] == 0.8


def test_multi_objective_scores_accept_legacy_plugin_names_as_aliases():
    """Legacy plugins can still pass code-repair names at the boundary."""
    from core.domain.genome import MultiObjectiveScores

    scores = MultiObjectiveScores(
        bugfix=0.8, style=0.7, security=0.9, runtime=0.6, syntax=0.85
    )

    assert scores.task_score == 0.8
    assert scores.quality_score == 0.7
    assert scores.risk_score == 0.9
    assert scores.efficiency_score == 0.6
    assert scores.validity_score == 0.85
    assert scores.bugfix == scores.task_score


def test_objective_vector_supports_target_neutral_scores():
    """Core objective handling should not require code-repair field names."""
    from core.domain.objectives import ObjectiveVector

    scores = ObjectiveVector.from_mapping(
        values={
            "semantic_match": 0.8,
            "background_plainness": 0.7,
            "aesthetic_quality": 0.6,
        },
        weights={
            "semantic_match": 0.5,
            "background_plainness": 0.3,
            "aesthetic_quality": 0.2,
        },
    )

    assert scores.keys() == (
        "semantic_match",
        "background_plainness",
        "aesthetic_quality",
    )
    assert scores.to_dict()["semantic_match"] == 0.8
    assert scores.weighted_fitness() == pytest.approx(0.73)


def test_objective_vector_fails_fast_on_missing_weight():
    """Objective weights must be explicit for each target objective."""
    from core.domain.objectives import ObjectiveVector

    with pytest.raises(ValueError, match="missing weights"):
        ObjectiveVector.from_mapping(
            values={"exact_accuracy": 0.9, "formatting": 0.8},
            weights={"exact_accuracy": 1.0},
        )


def test_legacy_multi_objective_scores_expose_generic_vector():
    """Legacy score objects should bridge into the generic objective contract."""
    from core.domain.genome import MultiObjectiveScores

    scores = MultiObjectiveScores(
        bugfix=0.8, style=0.7, security=0.9, runtime=0.6, syntax=0.85
    )
    vector = scores.as_objective_vector(
        labels={
            "task_score": "Task success",
            "quality_score": "Output quality",
            "risk_score": "Risk control",
            "efficiency_score": "Runtime efficiency",
            "validity_score": "Format validity",
        }
    )

    assert vector.label_for("task_score") == "Task success"
    assert vector.weighted_fitness() == pytest.approx(scores.overall_fitness())


def test_proof_quality_gate_requires_all_comparative_records():
    """Strong proof reports should require base, fixed, random, evolved, and held-out data."""
    from core.domain.proof import validate_proof_quality

    with pytest.raises(ValueError, match="held_out"):
        validate_proof_quality(
            {
                "base": {"fitness": 0.4},
                "fixed": {"fitness": 0.5},
                "random": {"fitness": 0.6},
                "evolved": {"fitness": 0.7},
            }
        )


def test_proof_quality_gate_accepts_evolved_held_out_win():
    """A credible proof needs evolved performance to beat fixed and random controls."""
    from core.domain.proof import validate_proof_quality

    verdict = validate_proof_quality(
        {
            "base": {"fitness": 0.4},
            "fixed": {"fitness": 0.5},
            "random": {"fitness": 0.6},
            "evolved": {"fitness": 0.7},
            "held_out": {"fitness": 0.65},
            "seeds": [11, 13, 17],
        }
    )

    assert verdict.passes
    assert verdict.evolved_beats_fixed
    assert verdict.evolved_beats_random
    assert verdict.has_multi_seed_support


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
