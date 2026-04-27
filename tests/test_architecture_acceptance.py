"""Architecture and documentation acceptance tests."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from core.cli.proof import _best_evolution_candidate_by_exact_accuracy
from core.common.config import CoralConfig
from core.domain.ca import CASeed
from core.domain.experiment import create_experiment_config, create_initial_population
from core.domain.genome import Genome
from core.domain.mapping import LoRAConfig
from infra.adapter_cache import HeavyGenes
from plugins.gsm8k_lora.plugin import (
    GSM8KLoRAFitness,
    GSM8KLoRAMetrics,
    GSM8KLoRARunner,
    _answers_match,
    _extract_final_answer,
)
from plugins.registry import create_plugin, supported_targets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
M1_CONFIG = PROJECT_ROOT / "config" / "examples" / "m1_tiny.yaml"
GSM8K_CONFIG = PROJECT_ROOT / "config" / "examples" / "gsm8k_lora_micro.yaml"


def _load_m1_config_dict() -> dict:
    return yaml.safe_load(M1_CONFIG.read_text())


def _load_gsm8k_config_dict() -> dict:
    return yaml.safe_load(GSM8K_CONFIG.read_text())


def _tiny_lora_config() -> LoRAConfig:
    return LoRAConfig(
        r=2,
        alpha=4,
        dropout=0.05,
        target_modules=("q_proj", "v_proj"),
        adapter_type="lora",
    )


def _tiny_genome() -> Genome:
    return Genome(
        seed=CASeed(grid=np.zeros((2, 2), dtype=int), rule=30, steps=1),
        lora_cfg=_tiny_lora_config(),
        id="gsm8k-test-genome",
        run_id="gsm8k-test-run",
    )


def _tiny_gsm8k_problem() -> dict:
    return {
        "name": "gsm8k_lora",
        "train": [
            {
                "question": "If you have 1 apple and get 1 more, how many?",
                "answer": "#### 2",
                "final_answer": "2",
            }
        ],
        "eval": [
            {
                "question": "What is 2 + 2?",
                "answer": "#### 4",
                "final_answer": "4",
            }
        ],
    }


def test_plugin_registry_resolves_supported_targets():
    """Every supported target should resolve to a plugin object."""
    raw_config = _load_m1_config_dict()

    for target in supported_targets():
        raw_config["experiment"]["target"] = target
        config = CoralConfig.model_validate(raw_config)
        plugin = create_plugin(config)

        assert callable(plugin.dataset)
        assert callable(plugin.model_factory)
        assert callable(plugin.fitness_fn)


def test_gsm8k_lora_plugin_is_import_safe_without_ml_downloads():
    """The real-LoRA plugin should not import ML stacks until evaluation."""
    raw_config = _load_m1_config_dict()
    raw_config["experiment"]["target"] = "gsm8k_lora"
    raw_config["experiment"]["model"]["name"] = "Qwen/Qwen2.5-0.5B-Instruct"

    config = CoralConfig.model_validate(raw_config)
    plugin = create_plugin(config)

    assert callable(plugin.dataset)
    assert callable(plugin.model_factory)
    assert callable(plugin.fitness_fn)


def test_gsm8k_answer_extraction_matches_final_numeric_answer():
    """GSM8K exact-answer scoring should compare the final numeric answer."""
    assert _extract_final_answer("Reasoning... #### 1,234") == "1234"
    assert _extract_final_answer(r"We get \boxed{42}.") == "42"
    assert _extract_final_answer(r"First \boxed{7}, final \boxed{1,234}.") == "1234"
    assert _extract_final_answer("The answer is 3.5") == "3.5"
    assert _extract_final_answer("no numeric answer") is None
    assert _answers_match("1234", "1,234")
    assert not _answers_match("1235", "1,234")


def test_gsm8k_cache_key_changes_when_fitness_settings_change(tmp_path):
    """GSM8K cache keys should include data and settings that affect fitness."""
    raw_config = _load_gsm8k_config_dict()
    raw_config["execution"]["output_dir"] = str(tmp_path / "base-output")
    raw_config["cache"]["artifacts_dir"] = str(tmp_path / "base-cache")

    problem = _tiny_gsm8k_problem()
    genome = _tiny_genome()
    base_runner = GSM8KLoRARunner(_tiny_lora_config(), raw_config, genome=genome)
    equivalent_runner = GSM8KLoRARunner(_tiny_lora_config(), raw_config, genome=genome)

    changed_config = yaml.safe_load(yaml.safe_dump(raw_config))
    changed_config["experiment"]["evaluation"]["loss_weight"] = 0.25
    changed_runner = GSM8KLoRARunner(_tiny_lora_config(), changed_config, genome=genome)

    changed_training_config = yaml.safe_load(yaml.safe_dump(raw_config))
    changed_training_config["training"]["learning_rate"] = 1e-4
    changed_training_runner = GSM8KLoRARunner(
        _tiny_lora_config(), changed_training_config, genome=genome
    )

    changed_prompt_config = yaml.safe_load(yaml.safe_dump(raw_config))
    changed_prompt_config["experiment"]["evaluation"]["answer_format"] = "boxed"
    changed_prompt_runner = GSM8KLoRARunner(
        _tiny_lora_config(), changed_prompt_config, genome=genome
    )

    changed_problem = _tiny_gsm8k_problem()
    changed_problem["eval"][0]["question"] = "What is 3 + 3?"

    assert base_runner._cache_key(problem) == equivalent_runner._cache_key(problem)
    assert base_runner._cache_key(problem) != changed_runner._cache_key(problem)
    assert base_runner._cache_key(problem) != changed_training_runner._cache_key(
        problem
    )
    assert base_runner._cache_key(problem) != changed_prompt_runner._cache_key(problem)
    assert base_runner._cache_key(problem) != base_runner._cache_key(changed_problem)


def test_gsm8k_fitness_combines_answer_accuracy_and_loss_score(monkeypatch, tmp_path):
    """GSM8K fitness should plumb exact-answer and loss scores into objectives."""
    raw_config = _load_gsm8k_config_dict()
    raw_config["execution"]["output_dir"] = str(tmp_path / "output")
    raw_config["cache"]["artifacts_dir"] = str(tmp_path / "cache")
    raw_config["experiment"]["evaluation"]["loss_weight"] = 0.25

    genome = _tiny_genome()
    problem = _tiny_gsm8k_problem()
    runner = GSM8KLoRARunner(_tiny_lora_config(), raw_config, genome=genome)

    def fake_train_and_evaluate(problem_arg):
        assert problem_arg == problem
        return GSM8KLoRAMetrics(
            exact_accuracy=0.5,
            formatted_answer_rate=1.0,
            train_loss=0.8,
            eval_loss=0.4,
            loss_score=0.9,
            train_seconds=0.01,
            eval_seconds=0.01,
            predictions=(),
            cache_key="test-cache-key",
        )

    monkeypatch.setattr(runner, "train_and_evaluate", fake_train_and_evaluate)

    scores = GSM8KLoRAFitness().evaluate_multi_objective(genome, runner, [problem])
    expected = 0.5 * 0.75 + 0.9 * 0.25

    assert scores.bugfix == expected
    assert scores.style == expected
    assert scores.security == expected
    assert scores.runtime == expected
    assert scores.syntax == expected
    assert math.isclose(scores.overall_fitness(), expected)


def test_proof_exact_candidate_scan_ignores_previous_runs(tmp_path):
    """Proof reports should only consider candidates from the current run."""
    candidate_jsonl = tmp_path / "candidate_evaluations.jsonl"
    stale_record = {
        "genome_id": "stale-best",
        "generation": 0,
        "fitness": 0.99,
        "lora": {"r": 2, "alpha": 4, "dropout": 0.0},
        "metadata": {"evaluation": {"exact_accuracy": 0.99}},
    }
    candidate_jsonl.write_text(json.dumps(stale_record) + "\n")
    start_offset = candidate_jsonl.stat().st_size

    current_records = [
        {
            "genome_id": "current-low",
            "generation": 0,
            "fitness": 0.1,
            "lora": {"r": 2, "alpha": 4, "dropout": 0.0},
            "metadata": {"evaluation": {"exact_accuracy": 0.1}},
        },
        {
            "genome_id": "current-best",
            "generation": 0,
            "fitness": 0.2,
            "lora": {"r": 4, "alpha": 8, "dropout": 0.05},
            "metadata": {"evaluation": {"exact_accuracy": 0.2}},
        },
    ]
    with candidate_jsonl.open("a") as handle:
        for record in current_records:
            handle.write(json.dumps(record) + "\n")

    best = _best_evolution_candidate_by_exact_accuracy(
        candidate_jsonl, start_offset=start_offset
    )

    assert best["genome_id"] == "current-best"


def test_gsm8k_config_dry_run_does_not_import_ml_stacks(tmp_path):
    """Dry-run should validate GSM8K wiring without importing downloader stacks."""
    raw_config = _load_gsm8k_config_dict()
    raw_config["execution"]["output_dir"] = str(tmp_path / "output")
    raw_config["cache"]["artifacts_dir"] = str(tmp_path / "cache")

    config_path = tmp_path / "gsm8k_dry_run.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))

    script = f"""
import sys
from pathlib import Path

sys.path.insert(0, {str(PROJECT_ROOT)!r})
from core.cli.main import _run_dry_validation
from core.common.config_loader import load_config

for name in ('torch', 'transformers', 'datasets', 'peft'):
    assert name not in sys.modules, name

config = load_config(Path({str(config_path)!r}))
_run_dry_validation(config)

for name in ('torch', 'transformers', 'datasets', 'peft'):
    print(f"{{name}}={{name in sys.modules}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Plugin: GSM8KLoRAPlugin" in result.stdout
    assert "Population: 3 deterministic genomes" in result.stdout
    assert "torch=False" in result.stdout
    assert "transformers=False" in result.stdout
    assert "datasets=False" in result.stdout
    assert "peft=False" in result.stdout


def test_mock_mini_services_load_only_selected_plugin():
    """The canonical mini path should load only its selected plugin."""
    script = f"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, {str(PROJECT_ROOT)!r})
from core.application.services import create_evolution_services
from core.common.config import CoralConfig

raw = yaml.safe_load(Path({str(M1_CONFIG)!r}).read_text())
config = CoralConfig.model_validate(raw)
assert 'plugins.fakenews_mini.plugin' not in sys.modules
services = create_evolution_services(config)
assert services.dataset_provider is not None
assert services.model_factory is not None
print('IMPORTED=' + str('plugins.fakenews_mini.plugin' in sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "IMPORTED=False" in result.stdout


def test_population_generation_stable_across_python_hash_seeds():
    """Initial population generation must not depend on Python's salted hash()."""
    script = f"""
import json
import sys
import yaml
from pathlib import Path

sys.path.insert(0, {str(PROJECT_ROOT)!r})
from core.domain.experiment import create_experiment_config, create_initial_population

raw = yaml.safe_load(Path({str(M1_CONFIG)!r}).read_text())
config = create_experiment_config(raw)
population = create_initial_population(config, raw_config=raw, run_id='hashseed-test')
payload = []
for genome in population.genomes:
    payload.append({{
        'id': genome.id,
        'rule': genome.seed.rule,
        'steps': genome.seed.steps,
        'grid': genome.seed.grid.tolist(),
        'lora': {{
            'r': genome.lora_cfg.r,
            'alpha': genome.lora_cfg.alpha,
            'dropout': genome.lora_cfg.dropout,
            'target_modules': list(genome.lora_cfg.target_modules),
            'adapter_type': genome.lora_cfg.adapter_type,
        }},
    }})
print('PAYLOAD=' + json.dumps(payload, sort_keys=True))
"""

    outputs = []
    for hash_seed in ("1", "987654"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        payload_line = next(
            line for line in result.stdout.splitlines() if line.startswith("PAYLOAD=")
        )
        outputs.append(json.loads(payload_line.removeprefix("PAYLOAD=")))

    assert outputs[0] == outputs[1]


def test_cache_hash_is_structural_and_stable():
    """Equivalent heavy genes should hash the same; structural changes should not."""
    base = HeavyGenes(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=("q_proj", "v_proj"),
        adapter_type="lora",
        run_id="test-run",
    )
    equivalent = HeavyGenes(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=("q_proj", "v_proj"),
        adapter_type="lora",
        run_id="test-run",
    )
    changed = HeavyGenes(
        rank=16,
        alpha=16.0,
        dropout=0.1,
        target_modules=("q_proj", "v_proj"),
        adapter_type="lora",
        run_id="test-run",
    )

    assert base.to_hash() == equivalent.to_hash()
    assert base.to_hash() != changed.to_hash()


def test_configured_ca_ranges_are_honored():
    """Population creation should honor CA grid, rule, steps, and density config."""
    raw_config = _load_m1_config_dict()
    raw_config["execution"]["population_size"] = 4
    raw_config["evo"]["ca"] = {
        "grid_size": [3, 5],
        "rule_range": [44, 44],
        "steps_range": [7, 7],
        "initial_density": 0.0,
    }

    config = create_experiment_config(raw_config)
    population = create_initial_population(config, raw_config=raw_config)

    for genome in population.genomes:
        assert genome.seed.grid.shape == (3, 5)
        assert int(genome.seed.grid.sum()) == 0
        assert genome.seed.rule == 44
        assert genome.seed.steps == 7


def test_docs_and_metadata_match_supported_behavior():
    """License and documentation should match the supported architecture."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    assert 'license = {text = "MIT"}' in pyproject
    assert "License :: OSI Approved :: MIT License" in pyproject

    public_docs = "\n".join(
        path.read_text().lower()
        for path in [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "architecture.md",
            PROJECT_ROOT / "docs" / "technical-deep-dive.md",
        ]
    )

    banned_phrases = [
        "mathematical guarantees",
        "mathematical guarantee",
        "identical adapter",
        "identical results",
        "same outputs",
        "transfer perfectly",
    ]
    for phrase in banned_phrases:
        assert phrase not in public_docs
