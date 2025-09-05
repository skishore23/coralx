"""
JSONL Logger for M1 - End-to-End Tiny Run
Logs candidate data with genes, metrics, and seeds in JSONL format
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from ..domain.genome import Genome
from ..common.logging import LoggingMixin


class JSONLLogger(LoggingMixin):
    """JSONL logger for evolution candidates and metrics."""

    def __init__(self, log_file: Path):
        super().__init__()
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file with header
        self._write_header()

        self.logger.info(f"JSONL logger initialized: {log_file}")

    def _write_header(self):
        """Write header comment to JSONL file."""
        header = {
            "_comment": "CORAL-X M1 Evolution Log",
            "_format": "JSONL - one JSON object per line",
            "_timestamp": time.time(),
            "_version": "1.0"
        }

        with open(self.log_file, 'w') as f:
            f.write(f"# {json.dumps(header)}\n")

    def log_candidate(self,
                     genome: Genome,
                     generation: int,
                     evaluation_time: float,
                     additional_data: Optional[Dict[str, Any]] = None):
        """Log a single candidate with all relevant data."""

        # Extract genome data
        genome_data = {
            "genome_id": genome.id,
            "generation": generation,
            "timestamp": time.time(),
            "evaluation_time": evaluation_time,
            "fitness": genome.fitness if genome.is_evaluated() else None,
        }

        # Add CA seed data
        if hasattr(genome, 'seed') and genome.seed:
            genome_data["ca_seed"] = {
                "rule": genome.seed.rule,
                "steps": genome.seed.steps,
                "grid_shape": genome.seed.grid.shape if hasattr(genome.seed, 'grid') else None,
                "grid_sum": int(genome.seed.grid.sum()) if hasattr(genome.seed, 'grid') else None
            }

        # Add LoRA configuration
        if hasattr(genome, 'lora_cfg') and genome.lora_cfg:
            genome_data["lora_config"] = {
                "r": genome.lora_cfg.r,
                "alpha": genome.lora_cfg.alpha,
                "dropout": genome.lora_cfg.dropout,
                "target_modules": list(genome.lora_cfg.target_modules),
                "adapter_type": getattr(genome.lora_cfg, 'adapter_type', 'lora')
            }

        # Add CA features
        if hasattr(genome, 'ca_features') and genome.ca_features:
            genome_data["ca_features"] = {
                "complexity": genome.ca_features.complexity,
                "intensity": genome.ca_features.intensity,
                "periodicity": genome.ca_features.periodicity,
                "convergence": genome.ca_features.convergence
            }

        # Add multi-objective scores
        if genome.has_multi_scores():
            scores = genome.multi_scores
            genome_data["multi_objective_scores"] = {
                "bugfix": scores.bugfix,
                "style": scores.style,
                "security": scores.security,
                "runtime": scores.runtime,
                "syntax": scores.syntax,
                "overall_fitness": scores.overall_fitness()
            }

        # Add run ID if available
        if hasattr(genome, 'run_id') and genome.run_id:
            genome_data["run_id"] = genome.run_id

        # Add any additional data
        if additional_data:
            genome_data.update(additional_data)

        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(genome_data) + '\n')

        self.logger.debug(f"Logged candidate {genome.id} to {self.log_file}")

    def log_generation_summary(self,
                              generation: int,
                              population_size: int,
                              best_fitness: float,
                              avg_fitness: float,
                              diversity_metrics: Dict[str, float],
                              selection_info: Dict[str, Any]):
        """Log generation summary data."""

        summary = {
            "type": "generation_summary",
            "generation": generation,
            "timestamp": time.time(),
            "population_size": population_size,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "diversity_metrics": diversity_metrics,
            "selection_info": selection_info
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')

        self.logger.info(f"Logged generation {generation} summary")

    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start with configuration."""

        start_data = {
            "type": "experiment_start",
            "timestamp": time.time(),
            "config": {
                "generations": config.get('generations', 1),
                "population_size": config.get('population_size', 8),
                "selection_mode": config.get('selection_mode', 'tournament'),
                "survival_rate": config.get('survival_rate', 0.5),
                "executor": config.get('executor', 'local'),
                "experiment_name": config.get('experiment_name', 'm1_test')
            }
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(start_data) + '\n')

        self.logger.info("Logged experiment start")

    def log_experiment_end(self,
                          total_time: float,
                          final_best_fitness: float,
                          total_candidates: int):
        """Log experiment end with final statistics."""

        end_data = {
            "type": "experiment_end",
            "timestamp": time.time(),
            "total_time": total_time,
            "final_best_fitness": final_best_fitness,
            "total_candidates": total_candidates
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(end_data) + '\n')

        self.logger.info("Logged experiment end")

    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about the log file."""
        if not self.log_file.exists():
            return {"error": "Log file does not exist"}

        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()

            # Count different types of entries
            candidate_count = 0
            generation_count = 0
            start_count = 0
            end_count = 0

            for line in lines:
                if line.strip().startswith('#'):
                    continue  # Skip header comments

                try:
                    data = json.loads(line.strip())
                    entry_type = data.get('type', 'candidate')

                    if entry_type == 'candidate':
                        candidate_count += 1
                    elif entry_type == 'generation_summary':
                        generation_count += 1
                    elif entry_type == 'experiment_start':
                        start_count += 1
                    elif entry_type == 'experiment_end':
                        end_count += 1
                except json.JSONDecodeError:
                    continue

            return {
                "total_lines": len(lines),
                "candidate_entries": candidate_count,
                "generation_summaries": generation_count,
                "experiment_starts": start_count,
                "experiment_ends": end_count,
                "file_size_bytes": self.log_file.stat().st_size
            }

        except Exception as e:
            return {"error": f"Failed to read log file: {e}"}
