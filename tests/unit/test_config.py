"""Unit tests for configuration system."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from core.common.config import (
    CoralConfig, ExecutionConfig, EvolutionConfig, ExperimentConfig,
    DatasetConfig, ModelConfig, FitnessWeights, EvaluationConfig,
    InfrastructureConfig, CacheConfig, ThresholdConfig, ObjectiveThresholds
)
from core.common.config_loader import ConfigManager
from core.common.exceptions import ConfigurationError


class TestCoralConfig:
    """Test CoralConfig Pydantic model."""
    
    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config_data = {
            'execution': {
                'generations': 5,
                'population_size': 4,
                'output_dir': './results'
            },
            'evo': {
                'rank_candidates': [4, 8, 16],
                'alpha_candidates': [8, 16, 32],
                'dropout_candidates': [0.05, 0.1, 0.15],
                'target_modules': ['q_proj', 'v_proj']
            },
            'experiment': {
                'target': 'test_target',
                'name': 'test_experiment',
                'dataset': {
                    'path': './data',
                    'datasets': ['test_dataset']
                },
                'model': {
                    'name': 'test_model'
                }
            },
            'evaluation': {
                'test_samples': 50,
                'fitness_weights': {
                    'bugfix': 0.3,
                    'style': 0.2,
                    'security': 0.3,
                    'runtime': 0.1,
                    'syntax': 0.1
                }
            },
            'infra': {
                'executor': 'local'
            },
            'cache': {
                'artifacts_dir': './cache',
                'base_checkpoint': 'test_model'
            },
            'threshold': {
                'base_thresholds': {
                    'bugfix': 0.3,
                    'style': 0.3,
                    'security': 0.4,
                    'runtime': 0.3,
                    'syntax': 0.2
                },
                'max_thresholds': {
                    'bugfix': 0.8,
                    'style': 0.7,
                    'security': 0.9,
                    'runtime': 0.7,
                    'syntax': 0.8
                }
            }
        }
        
        config = CoralConfig(**config_data)
        
        assert config.execution.generations == 5
        assert config.execution.population_size == 4
        assert config.evo.rank_candidates == [4, 8, 16]
        assert config.experiment.name == 'test_experiment'
        assert config.infra.executor.value == 'local'
    
    def test_invalid_generations(self):
        """Test validation fails for invalid generations."""
        config_data = {
            'execution': {'generations': 0, 'population_size': 4, 'output_dir': './results'},
            'evo': {'rank_candidates': [4], 'alpha_candidates': [8], 'dropout_candidates': [0.1], 'target_modules': ['q_proj']},
            'experiment': {'target': 'test', 'name': 'test', 'dataset': {'path': './data', 'datasets': ['test']}, 'model': {'name': 'test'}},
            'evaluation': {'test_samples': 50, 'fitness_weights': {'bugfix': 0.2, 'style': 0.2, 'security': 0.2, 'runtime': 0.2, 'syntax': 0.2}},
            'infra': {'executor': 'local'},
            'cache': {'artifacts_dir': './cache', 'base_checkpoint': 'test'},
            'threshold': {'base_thresholds': {'bugfix': 0.3, 'style': 0.3, 'security': 0.4, 'runtime': 0.3, 'syntax': 0.2}, 'max_thresholds': {'bugfix': 0.8, 'style': 0.7, 'security': 0.9, 'runtime': 0.7, 'syntax': 0.8}}
        }
        
        with pytest.raises(ValidationError):
            CoralConfig(**config_data)
    
    def test_fitness_weights_sum_validation(self):
        """Test that fitness weights must sum to 1.0."""
        config_data = {
            'execution': {'generations': 5, 'population_size': 4, 'output_dir': './results'},
            'evo': {'rank_candidates': [4], 'alpha_candidates': [8], 'dropout_candidates': [0.1], 'target_modules': ['q_proj']},
            'experiment': {'target': 'test', 'name': 'test', 'dataset': {'path': './data', 'datasets': ['test']}, 'model': {'name': 'test'}},
            'evaluation': {
                'test_samples': 50,
                'fitness_weights': {
                    'bugfix': 0.5,  # Sum = 1.5 > 1.0
                    'style': 0.5,
                    'security': 0.3,
                    'runtime': 0.1,
                    'syntax': 0.1
                }
            },
            'infra': {'executor': 'local'},
            'cache': {'artifacts_dir': './cache', 'base_checkpoint': 'test'},
            'threshold': {'base_thresholds': {'bugfix': 0.3, 'style': 0.3, 'security': 0.4, 'runtime': 0.3, 'syntax': 0.2}, 'max_thresholds': {'bugfix': 0.8, 'style': 0.7, 'security': 0.9, 'runtime': 0.7, 'syntax': 0.8}}
        }
        
        with pytest.raises(ValidationError):
            CoralConfig(**config_data)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager('test')
        assert manager.environment == 'test'
    
    def test_parse_env_value(self):
        """Test environment value parsing."""
        manager = ConfigManager()
        
        assert manager._parse_env_value('42') == 42
        assert manager._parse_env_value('3.14') == 3.14
        assert manager._parse_env_value('true') == True
        assert manager._parse_env_value('false') == False
        assert manager._parse_env_value('hello') == 'hello'
    
    def test_set_nested_value(self):
        """Test setting nested configuration values."""
        manager = ConfigManager()
        config = {}
        
        manager._set_nested_value(config, ('a', 'b', 'c'), 'value')
        
        assert config == {'a': {'b': {'c': 'value'}}}
    
    def test_merge_configs(self):
        """Test configuration merging."""
        manager = ConfigManager()
        base = {'a': {'b': 1}, 'c': 2}
        overrides = {'a': {'d': 3}, 'e': 4}
        
        merged = manager._merge_configs(base, overrides)
        
        expected = {'a': {'b': 1, 'd': 3}, 'c': 2, 'e': 4}
        assert merged == expected


if __name__ == '__main__':
    pytest.main([__file__])