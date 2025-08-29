"""Progress tracking service for CORAL-X evolution."""

import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from ..domain.genome import Genome
from ..domain.neat import Population
from ..common.config import CoralConfig
from ..common.logging import LoggingMixin
from ..common.exceptions import InfrastructureError


class ProgressTracker(LoggingMixin):
    """Tracks and persists evolution progress with structured data."""
    
    def __init__(self, config: CoralConfig, run_id: Optional[str] = None):
        super().__init__()
        self.config = config
        self.run_id = run_id or f"run_{int(time.time())}"
        
        # Initialize tracking state
        self.start_time = time.time()
        self.current_generation = 0
        self.max_generations = config.execution.generations
        self.population_size = config.execution.population_size
        
        # Set up progress file path
        self.progress_file_path = self._get_progress_file_path()
        
        # Initialize progress tracking
        self._initialize_progress_file()
        
        self.logger.info(f"Progress tracker initialized: run_id={self.run_id}, file={self.progress_file_path}, max_generations={self.max_generations}")
    
    def _get_progress_file_path(self) -> Path:
        """Get the path to the progress tracking file.
        
        Returns:
            Path to progress file
            
        Raises:
            InfrastructureError: If progress path cannot be determined
        """
        # Use output directory for progress tracking
        progress_file = self.config.execution.output_dir / "evolution_progress.json"
        
        # Ensure directory exists
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        return progress_file
    
    def _initialize_progress_file(self) -> None:
        """Initialize the progress tracking file with initial state."""
        initial_progress = {
            'status': 'starting',
            'message': 'Evolution starting...',
            'run_id': self.run_id,
            'current_generation': 0,
            'max_generations': self.max_generations,
            'population_size': self.population_size,
            'start_time': self.start_time,
            'start_time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)),
            'last_update': self.start_time,
            'elapsed_time': 0.0,
            'progress_percent': 0.0,
            'best_fitness': 0.0,
            'best_scores': {
                'bugfix': 0.0,
                'style': 0.0,
                'security': 0.0,
                'runtime': 0.0,
                'syntax': 0.0
            },
            'cache_stats': {
                'hit_rate': 0.0,
                'total_adapters': 0,
                'cache_size_mb': 0
            },
            'training_stats': {
                'adapters_trained': 0,
                'training_rate': 0.0,
                'current_adapter': 'Initializing...'
            },
            'population_stats': {
                'evaluated_count': 0,
                'evaluation_rate': 0.0,
                'diversity_score': 0.0
            },
            'infrastructure_stats': {
                'executor_type': self.config.infra.executor.value,
                'model_name': self.config.experiment.model.name
            },
            'generation_history': []
        }
        
        self._save_progress(initial_progress)
        
        self.logger.info(f"Progress file initialized: {self.progress_file_path}")
    
    def update_status(self, status: str, message: str, 
                     best_genome: Optional[Genome] = None) -> None:
        """Update the evolution status and progress.
        
        Args:
            status: Current status ('starting', 'evolving', 'completed', 'failed')
            message: Status message
            best_genome: Current best genome (optional)
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        progress_data = {
            'status': status,
            'message': message,
            'run_id': self.run_id,
            'current_generation': self.current_generation,
            'max_generations': self.max_generations,
            'population_size': self.population_size,
            'start_time': self.start_time,
            'start_time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)),
            'last_update': current_time,
            'last_update_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
            'elapsed_time': elapsed_time,
            'progress_percent': (self.current_generation / self.max_generations) * 100.0 if self.max_generations > 0 else 0.0
        }
        
        # Add best genome information if available
        if best_genome:
            progress_data['best_fitness'] = best_genome.fitness
            if best_genome.has_multi_scores():
                scores = best_genome.multi_scores
                progress_data['best_scores'] = {
                    'bugfix': scores.bugfix,
                    'style': scores.style,
                    'security': scores.security,
                    'runtime': scores.runtime,
                    'syntax': getattr(scores, 'syntax', 0.0)
                }
        else:
            progress_data['best_fitness'] = 0.0
            progress_data['best_scores'] = {
                'bugfix': 0.0,
                'style': 0.0,
                'security': 0.0,
                'runtime': 0.0,
                'syntax': 0.0
            }
        
        # Load existing progress to preserve additional data
        try:
            existing_progress = self._load_progress()
            # Preserve fields that aren't updated in this call
            for key in ['cache_stats', 'training_stats', 'population_stats', 
                       'infrastructure_stats', 'generation_history']:
                if key in existing_progress:
                    progress_data[key] = existing_progress[key]
        except:
            pass
        
        self._save_progress(progress_data)
        
        self.logger.info(f"Progress updated: status={status}, generation={self.current_generation}, message='{message}', elapsed_time={elapsed_time:.2f}s")
    
    def update_generation_progress(self, generation: int, population: Population) -> None:
        """Update progress for a specific generation.
        
        Args:
            generation: Current generation number
            population: Current population
        """
        self.current_generation = generation
        
        # Calculate population statistics
        evaluated_count = len([g for g in population.genomes if g.is_evaluated()])
        evaluation_rate = evaluated_count / population.size() if population.size() > 0 else 0.0
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(population)
        
        # Get best genome
        best_genome = None
        try:
            if population.size() > 0:
                best_genome = population.best()
        except:
            pass
        
        # Create generation entry for history
        generation_entry = {
            'generation': generation,
            'timestamp': time.time(),
            'population_size': population.size(),
            'evaluated_count': evaluated_count,
            'evaluation_rate': evaluation_rate,
            'diversity_score': diversity_score,
            'best_fitness': best_genome.fitness if best_genome else 0.0
        }
        
        # Load existing progress
        try:
            existing_progress = self._load_progress()
        except:
            existing_progress = {}
        
        # Update population stats
        existing_progress['population_stats'] = {
            'evaluated_count': evaluated_count,
            'evaluation_rate': evaluation_rate,
            'diversity_score': diversity_score
        }
        
        # Update generation history
        if 'generation_history' not in existing_progress:
            existing_progress['generation_history'] = []
        existing_progress['generation_history'].append(generation_entry)
        
        # Keep only last 50 generation entries to prevent file bloat
        if len(existing_progress['generation_history']) > 50:
            existing_progress['generation_history'] = existing_progress['generation_history'][-50:]
        
        self._save_progress(existing_progress)
        
        self.logger.info(f"Generation progress updated: generation={generation}, population_size={population.size()}, evaluated_count={evaluated_count}, diversity_score={diversity_score}")
    
    def update_cache_stats(self, hit_rate: float, total_adapters: int, 
                          cache_size_mb: float) -> None:
        """Update cache statistics.
        
        Args:
            hit_rate: Cache hit rate (0.0 to 1.0)
            total_adapters: Total number of cached adapters
            cache_size_mb: Cache size in megabytes
        """
        try:
            existing_progress = self._load_progress()
            existing_progress['cache_stats'] = {
                'hit_rate': hit_rate,
                'total_adapters': total_adapters,
                'cache_size_mb': cache_size_mb
            }
            self._save_progress(existing_progress)
            
            self.logger.debug(f"Cache stats updated: hit_rate={hit_rate}, total_adapters={total_adapters}, cache_size_mb={cache_size_mb}")
        except Exception as e:
            self.logger.error(f"Cache stats update failed: {e}")
    
    def update_training_stats(self, adapters_trained: int, training_rate: float,
                            current_adapter: str) -> None:
        """Update training statistics.
        
        Args:
            adapters_trained: Number of adapters trained
            training_rate: Training rate (adapters per minute)
            current_adapter: Currently training adapter
        """
        try:
            existing_progress = self._load_progress()
            existing_progress['training_stats'] = {
                'adapters_trained': adapters_trained,
                'training_rate': training_rate,
                'current_adapter': current_adapter
            }
            self._save_progress(existing_progress)
            
            self.logger.debug(f"Training stats updated: adapters_trained={adapters_trained}, training_rate={training_rate}, current_adapter={current_adapter}")
        except Exception as e:
            self.logger.error(f"Training stats update failed: {e}")
    
    def _calculate_diversity_score(self, population: Population) -> float:
        """Calculate a simple diversity score for the population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if population.size() <= 1:
            return 0.0
        
        # Count unique fitness values
        fitness_values = [g.fitness for g in population.genomes if g.is_evaluated()]
        if len(fitness_values) <= 1:
            return 0.0
        
        unique_fitness = len(set(fitness_values))
        diversity_score = unique_fitness / len(fitness_values)
        
        return min(1.0, diversity_score)
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress data from file.
        
        Returns:
            Progress data dictionary
            
        Raises:
            InfrastructureError: If progress file cannot be loaded
        """
        try:
            if not self.progress_file_path.exists():
                return {}
            
            with open(self.progress_file_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            raise InfrastructureError(f"Failed to load progress file: {e}",
                                    context={"progress_file": str(self.progress_file_path)})
    
    def _save_progress(self, progress_data: Dict[str, Any]) -> None:
        """Save progress data to file.
        
        Args:
            progress_data: Progress data to save
            
        Raises:
            InfrastructureError: If progress file cannot be saved
        """
        try:
            with open(self.progress_file_path, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            raise InfrastructureError(f"Failed to save progress file: {e}",
                                    context={"progress_file": str(self.progress_file_path)})
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current progress data.
        
        Returns:
            Current progress data
        """
        return self._load_progress()
    
    def cleanup_old_progress_files(self, max_files: int = 10) -> None:
        """Clean up old progress files to prevent disk bloat.
        
        Args:
            max_files: Maximum number of progress files to keep
        """
        try:
            progress_dir = self.progress_file_path.parent
            progress_files = list(progress_dir.glob("evolution_progress_*.json"))
            
            if len(progress_files) > max_files:
                # Sort by modification time, oldest first
                progress_files.sort(key=lambda p: p.stat().st_mtime)
                files_to_remove = progress_files[:-max_files]
                
                for file_path in files_to_remove:
                    file_path.unlink()
                    
                self.logger.info(f"Old progress files cleaned: removed_count={len(files_to_remove)}, remaining_count={len(progress_files) - len(files_to_remove)}")
        except Exception as e:
            self.logger.error(f"Progress cleanup failed: {e}")