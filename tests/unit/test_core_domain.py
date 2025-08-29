"""Tests for core domain modules."""

import pytest
from pathlib import Path

def test_adapter_config_import():
    """Test adapter config module imports."""
    from core.domain.adapter_config import AdapterParameters, AdapterEnvironment
    assert AdapterParameters is not None
    assert AdapterEnvironment is not None

def test_mapping_import():
    """Test mapping module imports."""
    from core.domain.mapping import map_features_to_lora_config, LoRAConfig
    assert map_features_to_lora_config is not None
    assert LoRAConfig is not None

def test_experiment_import():
    """Test experiment module imports."""
    from core.domain.experiment import create_experiment_config, ExperimentConfig
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
        adapter_type="lora"
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
            adapter_type="lora"
        )

def test_multi_objective_scores():
    """Test multi-objective scores creation."""
    from core.domain.genome import MultiObjectiveScores
    
    scores = MultiObjectiveScores(
        bugfix=0.8,
        style=0.7,
        security=0.9,
        runtime=0.6,
        syntax=0.85
    )
    
    assert scores.bugfix == 0.8
    assert scores.overall_fitness() > 0.0
    
    scores_dict = scores.to_dict()
    assert "bugfix" in scores_dict
    assert scores_dict["bugfix"] == 0.8