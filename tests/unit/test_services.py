"""Unit tests for service layer."""

import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path

from core.common.config import CoralConfig, ExecutionConfig, EvolutionConfig, ExperimentConfig, DatasetConfig, ModelConfig, EvaluationConfig, FitnessWeights, InfrastructureConfig, CacheConfig, ThresholdConfig, ObjectiveThresholds
from core.services.population_manager import PopulationManager
from core.services.genetic_operations import GeneticOperationsService
from core.services.progress_tracker import ProgressTracker
from core.domain.neat import Population
from core.domain.genome import Genome
from core.common.exceptions import EvolutionError


def create_test_config():
    """Create a test configuration."""
    return CoralConfig(
        execution=ExecutionConfig(
            generations=5,
            population_size=4,
            output_dir=Path('./test_results')
        ),
        evo=EvolutionConfig(
            rank_candidates=[4, 8],
            alpha_candidates=[8, 16],
            dropout_candidates=[0.1],
            target_modules=['q_proj']
        ),
        experiment=ExperimentConfig(
            target='test_target',
            name='test_experiment',
            dataset=DatasetConfig(
                path=Path('./test_data'),
                datasets=['test']
            ),
            model=ModelConfig(name='test_model')
        ),
        evaluation=EvaluationConfig(
            test_samples=50,
            fitness_weights=FitnessWeights(
                bugfix=0.2,
                style=0.2,
                security=0.2,
                runtime=0.2,
                syntax=0.2
            )
        ),
        infra=InfrastructureConfig(executor='local'),
        cache=CacheConfig(
            artifacts_dir=Path('./test_cache'),
            base_checkpoint='test_model'
        ),
        threshold=ThresholdConfig(
            base_thresholds=ObjectiveThresholds(
                bugfix=0.3, style=0.3, security=0.4, runtime=0.3, syntax=0.2
            ),
            max_thresholds=ObjectiveThresholds(
                bugfix=0.8, style=0.7, security=0.9, runtime=0.7, syntax=0.8
            )
        ),
        seed=42
    )


def create_test_population(size=4):
    """Create a test population."""
    import numpy as np
    from core.domain.ca import CASeed
    from core.domain.mapping import AdapterConfig
    
    genomes = []
    for i in range(size):
        # Create test CA seed
        test_grid = np.random.randint(0, 2, size=(10, 10))
        ca_seed = CASeed(grid=test_grid, rule=30, steps=15)
        
        # Create test LoRA config
        lora_cfg = AdapterConfig(
            r=8,
            alpha=16.0,
            dropout=0.1,
            target_modules=("q_proj", "v_proj"),
            adapter_type="lora"
        )
        
        # Create test scores for Pareto selection
        from core.domain.genome import MultiObjectiveScores
        test_scores = MultiObjectiveScores(
            bugfix=0.5 + i * 0.1,
            style=0.6 + i * 0.05,
            security=0.7,
            runtime=0.8 - i * 0.1,
            syntax=0.75
        )
        
        genome = Genome(
            seed=ca_seed,
            lora_cfg=lora_cfg,
            id=f'genome_{i}',
            multi_scores=test_scores,
            run_id='test_run'
        )
        genomes.append(genome)
    
    return Population(tuple(genomes))


class TestPopulationManager:
    """Test PopulationManager service."""
    
    def test_initialization(self):
        """Test PopulationManager initialization."""
        config = create_test_config()
        manager = PopulationManager(config, 42)
        
        assert manager.config == config
        assert manager.current_generation == 0
        assert manager.random.getstate() is not None
    
    def test_validate_population_success(self):
        """Test population validation with valid population."""
        config = create_test_config()
        manager = PopulationManager(config)
        population = create_test_population(4)
        
        # Should not raise any exception
        manager.validate_population(population)
    
    def test_validate_population_empty(self):
        """Test population validation with empty population."""
        config = create_test_config()
        manager = PopulationManager(config)
        empty_population = Population(tuple())
        
        with pytest.raises(EvolutionError, match="Population is empty"):
            manager.validate_population(empty_population)
    
    def test_select_survivors(self):
        """Test survivor selection."""
        config = create_test_config()
        manager = PopulationManager(config)
        population = create_test_population(4)
        
        survivors = manager.select_survivors(population)
        
        # Should return smaller population
        assert survivors.size() <= population.size()
        assert survivors.size() > 0
    
    def test_calculate_diversity_metrics(self):
        """Test diversity metrics calculation."""
        config = create_test_config()
        manager = PopulationManager(config)
        population = create_test_population(4)
        
        metrics = manager.calculate_diversity_metrics(population)
        
        assert 'fitness_diversity' in metrics
        assert 'genetic_diversity' in metrics
        assert 'phenotype_diversity' in metrics
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
    
    def test_record_generation_stats(self):
        """Test generation statistics recording."""
        config = create_test_config()
        manager = PopulationManager(config)
        population = create_test_population(4)
        
        initial_gen = manager.current_generation
        manager.record_generation_stats(population)
        
        assert manager.current_generation == initial_gen + 1
        assert len(manager.generation_history['best_fitness']) == 1
        assert len(manager.generation_history['diversity_scores']) == 1


class TestGeneticOperationsService:
    """Test GeneticOperationsService."""
    
    def test_initialization(self):
        """Test GeneticOperationsService initialization."""
        config = create_test_config()
        service = GeneticOperationsService(config, 42)
        
        assert service.config == config
        assert service.crossover_rate == config.execution.crossover_rate
        assert service.population_size == config.execution.population_size
    
    def test_calculate_diversity_strength(self):
        """Test diversity strength calculation."""
        config = create_test_config()
        service = GeneticOperationsService(config)
        population = create_test_population(4)
        
        strength = service.calculate_diversity_strength(population)
        
        assert isinstance(strength, float)
        assert 0.0 <= strength <= 2.0
    
    def test_adjust_genetic_parameters(self):
        """Test genetic parameter adjustment."""
        config = create_test_config()
        service = GeneticOperationsService(config)
        population = create_test_population(4)
        
        original_rate = service.crossover_rate
        service.adjust_genetic_parameters(population, 1)
        
        # Rate may have changed based on diversity
        assert isinstance(service.crossover_rate, float)
        assert 0.0 <= service.crossover_rate <= 1.0
    
    def test_get_generation_summary_empty(self):
        """Test generation summary with no operations."""
        config = create_test_config()
        service = GeneticOperationsService(config)
        
        summary = service.get_generation_summary()
        
        assert 'message' in summary
        assert 'No genetic operations performed yet' in summary['message']


class TestProgressTracker:
    """Test ProgressTracker service."""
    
    def test_initialization(self):
        """Test ProgressTracker initialization."""
        config = create_test_config()
        tracker = ProgressTracker(config, 'test_run')
        
        assert tracker.run_id == 'test_run'
        assert tracker.max_generations == config.execution.generations
        assert tracker.progress_file_path.exists()
    
    def test_update_status(self):
        """Test status update."""
        config = create_test_config()
        tracker = ProgressTracker(config)
        
        # Should not raise exception
        tracker.update_status('running', 'Test message')
        
        # Check progress file was updated
        progress_data = tracker.get_current_progress()
        assert progress_data['status'] == 'running'
        assert progress_data['message'] == 'Test message'
    
    def test_update_generation_progress(self):
        """Test generation progress update."""
        config = create_test_config()
        tracker = ProgressTracker(config)
        population = create_test_population(4)
        
        # Should not raise exception
        tracker.update_generation_progress(1, population)
        
        # Check progress was updated
        progress_data = tracker.get_current_progress()
        assert 'population_stats' in progress_data
        assert 'generation_history' in progress_data
    
    def test_update_cache_stats(self):
        """Test cache stats update."""
        config = create_test_config()
        tracker = ProgressTracker(config)
        
        tracker.update_cache_stats(0.8, 10, 100.5)
        
        progress_data = tracker.get_current_progress()
        assert progress_data['cache_stats']['hit_rate'] == 0.8
        assert progress_data['cache_stats']['total_adapters'] == 10
        assert progress_data['cache_stats']['cache_size_mb'] == 100.5


if __name__ == '__main__':
    pytest.main([__file__])