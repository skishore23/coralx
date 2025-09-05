"""
Reproducibility utilities for M1 - End-to-End Tiny Run
Pins seeds and writes repro.lock with datasets, checkpoints, versions
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..common.logging import LoggingMixin


@dataclass(frozen=True)
class ReproducibilityInfo:
    """Immutable reproducibility information."""
    experiment_id: str
    timestamp: float
    coralx_version: str
    python_version: str
    seed: int
    config_hash: str
    datasets: Dict[str, str]
    checkpoints: Dict[str, str]
    dependencies: Dict[str, str]
    environment: Dict[str, str]


class ReproducibilityManager(LoggingMixin):
    """Manages reproducibility for M1 experiments."""

    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repro_file = output_dir / "repro.lock"

        self.logger.info(f"Reproducibility manager initialized: {self.repro_file}")

    def create_repro_lock(self,
                         experiment_id: str,
                         seed: int,
                         config: Dict[str, Any],
                         datasets: Optional[Dict[str, str]] = None,
                         checkpoints: Optional[Dict[str, str]] = None) -> ReproducibilityInfo:
        """Create repro.lock file with all necessary information for reproducibility."""

        # Get system information
        import sys
        import platform

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Get CoralX version
        coralx_version = self._get_coralx_version()

        # Create config hash
        config_hash = self._hash_config(config)

        # Default datasets if not provided
        if datasets is None:
            datasets = {
                "quixbugs_mini": "3_problems_mock",
                "fakenews_mini": "6_samples_mock"
            }

        # Default checkpoints if not provided
        if checkpoints is None:
            checkpoints = {
                "base_model": "mock_model_for_m1",
                "tokenizer": "mock_tokenizer_for_m1"
            }

        # Get dependencies
        dependencies = self._get_dependencies()

        # Get environment info
        environment = {
            "platform": platform.platform(),
            "python_version": python_version,
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor()
        }

        # Create reproducibility info
        repro_info = ReproducibilityInfo(
            experiment_id=experiment_id,
            timestamp=time.time(),
            coralx_version=coralx_version,
            python_version=python_version,
            seed=seed,
            config_hash=config_hash,
            datasets=datasets,
            checkpoints=checkpoints,
            dependencies=dependencies,
            environment=environment
        )

        # Write repro.lock file
        self._write_repro_lock(repro_info)

        self.logger.info(f"Created repro.lock: {self.repro_file}")
        self.logger.info(f"   Experiment ID: {experiment_id}")
        self.logger.info(f"   Seed: {seed}")
        self.logger.info(f"   Config hash: {config_hash[:16]}...")
        self.logger.info(f"   Datasets: {list(datasets.keys())}")
        self.logger.info(f"   Checkpoints: {list(checkpoints.keys())}")

        return repro_info

    def _get_coralx_version(self) -> str:
        """Get CoralX version."""
        try:
            # Try to get version from VERSION file
            version_file = Path(__file__).parent.parent.parent / "VERSION"
            if version_file.exists():
                return version_file.read_text().strip()

            # Fallback to git if available
            import subprocess
            try:
                result = subprocess.run(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
                )
                if result.returncode == 0:
                    return f"git-{result.stdout.strip()}"
            except FileNotFoundError:
                pass

            return "unknown"
        except Exception:
            return "unknown"

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create hash of configuration for reproducibility."""
        # Convert config to JSON string for hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _get_dependencies(self) -> Dict[str, str]:
        """Get dependency versions."""
        dependencies = {}

        # Core dependencies
        try:
            import numpy
            dependencies["numpy"] = numpy.__version__
        except ImportError:
            dependencies["numpy"] = "not_installed"

        try:
            import scipy
            dependencies["scipy"] = scipy.__version__
        except ImportError:
            dependencies["scipy"] = "not_installed"

        try:
            import torch
            dependencies["torch"] = torch.__version__
        except ImportError:
            dependencies["torch"] = "not_installed"

        try:
            import transformers
            dependencies["transformers"] = transformers.__version__
        except ImportError:
            dependencies["transformers"] = "not_installed"

        try:
            import peft
            dependencies["peft"] = peft.__version__
        except ImportError:
            dependencies["peft"] = "not_installed"

        try:
            import modal
            dependencies["modal"] = modal.__version__
        except ImportError:
            dependencies["modal"] = "not_installed"

        return dependencies

    def _write_repro_lock(self, repro_info: ReproducibilityInfo):
        """Write repro.lock file."""

        # Convert to dictionary for JSON serialization
        repro_dict = {
            "experiment_id": repro_info.experiment_id,
            "timestamp": repro_info.timestamp,
            "coralx_version": repro_info.coralx_version,
            "python_version": repro_info.python_version,
            "seed": repro_info.seed,
            "config_hash": repro_info.config_hash,
            "datasets": repro_info.datasets,
            "checkpoints": repro_info.checkpoints,
            "dependencies": repro_info.dependencies,
            "environment": repro_info.environment,
            "reproducibility_notes": [
                "This file contains all information needed to reproduce the experiment",
                "Use the same seed and config_hash to get identical results",
                "Dependencies should match the versions listed above",
                "Environment differences may cause slight variations"
            ]
        }

        with open(self.repro_file, 'w') as f:
            json.dump(repro_dict, f, indent=2)

    def load_repro_lock(self) -> Optional[ReproducibilityInfo]:
        """Load reproducibility information from repro.lock file."""
        if not self.repro_file.exists():
            self.logger.warning(f"repro.lock file not found: {self.repro_file}")
            return None

        try:
            with open(self.repro_file, 'r') as f:
                repro_dict = json.load(f)

            return ReproducibilityInfo(
                experiment_id=repro_dict["experiment_id"],
                timestamp=repro_dict["timestamp"],
                coralx_version=repro_dict["coralx_version"],
                python_version=repro_dict["python_version"],
                seed=repro_dict["seed"],
                config_hash=repro_dict["config_hash"],
                datasets=repro_dict["datasets"],
                checkpoints=repro_dict["checkpoints"],
                dependencies=repro_dict["dependencies"],
                environment=repro_dict["environment"]
            )

        except Exception as e:
            self.logger.error(f"Failed to load repro.lock: {e}")
            return None

    def verify_reproducibility(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Verify current environment matches repro.lock."""
        repro_info = self.load_repro_lock()
        if not repro_info:
            return {"status": "no_repro_lock", "message": "No repro.lock file found"}

        # Check config hash
        current_config_hash = self._hash_config(current_config)
        config_matches = current_config_hash == repro_info.config_hash

        # Check dependencies
        current_deps = self._get_dependencies()
        dep_matches = {}
        for dep, version in repro_info.dependencies.items():
            current_version = current_deps.get(dep, "not_installed")
            dep_matches[dep] = {
                "expected": version,
                "current": current_version,
                "matches": version == current_version
            }

        all_deps_match = all(info["matches"] for info in dep_matches.values())

        verification = {
            "status": "verified" if config_matches and all_deps_match else "mismatch",
            "config_hash_matches": config_matches,
            "dependencies_match": all_deps_match,
            "dependency_details": dep_matches,
            "experiment_id": repro_info.experiment_id,
            "original_timestamp": repro_info.timestamp,
            "seed": repro_info.seed
        }

        if verification["status"] == "verified":
            self.logger.info("Reproducibility verification passed")
        else:
            self.logger.warning("Reproducibility verification failed - results may differ")

        return verification
