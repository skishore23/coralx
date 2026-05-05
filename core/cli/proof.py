"""Proof-report helpers for GSM8K LoRA runs."""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from core.application.evolution_orchestrator import EvolutionOrchestrator
from core.application.services import create_evolution_services
from core.common.config import CoralConfig
from core.domain.ca import CASeed
from core.domain.genome import Genome
from core.domain.mapping import LoRAConfig
from core.domain.proof import (
    validate_proof_execution_policy,
    validate_proof_quality,
)
from plugins.gsm8k_lora.plugin import GSM8KLoRARunner, score_gsm8k_metrics
from plugins.registry import create_plugin


def run_gsm8k_lora_proof(
    config: CoralConfig,
    random_trials: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run all configured proof seeds and write an aggregate GSM8K LoRA report."""
    started_at = time.time()
    seed_reports = [
        _run_gsm8k_lora_single_seed_proof(seed_config, random_trials)
        for seed_config in _proof_seed_configs(config)
    ]
    report = _aggregate_gsm8k_lora_seed_reports(config, seed_reports, started_at)

    destination = output_path or (config.execution.output_dir / "proof_report.json")
    report["artifacts"]["proof_report"] = str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def _run_gsm8k_lora_single_seed_proof(
    config: CoralConfig,
    random_trials: int | None = None,
) -> dict[str, Any]:
    """Run evolution plus fixed/random controls for one GSM8K LoRA proof seed."""
    if config.experiment.target != "gsm8k_lora":
        raise ValueError(
            "The proof command currently supports experiment.target='gsm8k_lora' only"
        )

    started_at = time.time()
    candidate_jsonl = config.execution.output_dir / "candidate_evaluations.jsonl"
    candidate_jsonl_start = _file_size(candidate_jsonl)
    services = create_evolution_services(config, plugin=create_plugin(config))
    orchestrator = EvolutionOrchestrator(services)
    evolution_result = asyncio.run(orchestrator.run_evolution())
    if evolution_result.status != "completed":
        raise RuntimeError(f"Evolution failed with status={evolution_result.status}")

    if not services.dataset_provider or not services.model_factory:
        raise RuntimeError("Proof requires dataset provider and model factory")

    problem = next(iter(services.dataset_provider.problems()))
    budget = (
        random_trials or config.execution.population_size * config.execution.generations
    )

    base_record = _evaluate_candidate(
        config,
        services.model_factory,
        problem,
        _base_model_config(),
        candidate_id="base_model",
    )

    fixed_lora = _fixed_lora_config(config)
    fixed_record = _evaluate_candidate(
        config,
        services.model_factory,
        problem,
        fixed_lora,
        candidate_id="fixed_baseline",
    )

    random_records = []
    for index, lora_cfg in enumerate(_sample_lora_configs(config, budget)):
        random_records.append(
            _evaluate_candidate(
                config,
                services.model_factory,
                problem,
                lora_cfg,
                candidate_id=f"random_{index:04d}",
            )
        )

    evolution_best = _best_genome_record(evolution_result.best_genome)
    evolution_best_exact = _best_evolution_candidate_by_exact_accuracy(
        candidate_jsonl,
        start_offset=candidate_jsonl_start,
    )
    random_best = max(random_records, key=lambda row: row["fitness"], default=None)
    if evolution_best is None:
        raise RuntimeError("FAIL-FAST: proof requires an evolved best genome")
    held_out_record = _evaluate_candidate(
        config,
        services.model_factory,
        _held_out_problem(problem),
        _lora_config_from_record(evolution_best["lora"]),
        candidate_id="held_out_evolved_best",
    )

    report = {
        "target": config.experiment.target,
        "experiment": config.experiment.name,
        "model": config.experiment.model.name,
        "dataset": "openai/gsm8k",
        "started_at": started_at,
        "total_seconds": time.time() - started_at,
        "scope": (
            "This report covers real LoRA train/eval wiring, structural caching, "
            "fitness differentiation, selection retention, and same-budget random "
            "controls for a GSM8K run. It also includes an untrained base-model "
            "baseline for judging whether adapter search improves exact-answer "
            "math behavior."
        ),
        "fitness": {
            "primary": (
                "(1 - loss_weight) * exact_answer_accuracy "
                "+ loss_weight * exp(-held_out_answer_token_loss)"
            ),
            "exact_accuracy_reported": True,
            "loss_weight": (
                config.experiment.evaluation.get("loss_weight", 1.0)
                if config.experiment.evaluation
                else 1.0
            ),
        },
        "budget": {
            "evolution_generations": config.execution.generations,
            "evolution_population_size": config.execution.population_size,
            "random_trials": len(random_records),
        },
        "seeds": [config.seed],
        "evolution": {
            "status": evolution_result.status,
            "generations_completed": evolution_result.generations_completed,
            "total_seconds": evolution_result.total_time,
            "best_by_fitness": evolution_best,
            "best_by_exact_accuracy": evolution_best_exact,
            "best": evolution_best,
            "final_population": [
                _best_genome_record(genome)
                for genome in evolution_result.final_population.genomes
            ],
        },
        "base_model_baseline": base_record,
        "fixed_baseline": fixed_record,
        "random_baseline": {
            "best": random_best,
            "median_fitness": _median(
                [row["fitness"] for row in random_records if row["fitness"] is not None]
            ),
            "trials": random_records,
        },
        "held_out": held_out_record,
        "interpretation": _interpret(
            evolution_best, base_record, fixed_record, random_best
        ),
        "artifacts": {"candidate_jsonl": str(candidate_jsonl)},
    }

    destination = config.execution.output_dir / "proof_report.json"
    report["artifacts"]["proof_report"] = str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def _aggregate_gsm8k_lora_seed_reports(
    config: CoralConfig,
    seed_reports: list[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    """Aggregate single-seed proof reports into one framework-level proof report."""
    if not seed_reports:
        raise ValueError("FAIL-FAST: proof aggregation requires at least one seed report")

    base = _mean_fitness_record(
        [report["base_model_baseline"] for report in seed_reports],
        "base_model_mean",
    )
    fixed = _mean_fitness_record(
        [report["fixed_baseline"] for report in seed_reports],
        "fixed_baseline_mean",
    )
    random_best = _mean_fitness_record(
        [report["random_baseline"]["best"] for report in seed_reports],
        "random_best_mean",
    )
    evolution_best = _mean_fitness_record(
        [report["evolution"]["best"] for report in seed_reports],
        "evolution_best_mean",
    )
    held_out = _mean_fitness_record(
        [report["held_out"] for report in seed_reports],
        "held_out_evolved_mean",
    )
    report = {
        "target": config.experiment.target,
        "experiment": config.experiment.name,
        "model": config.experiment.model.name,
        "dataset": "openai/gsm8k",
        "started_at": started_at,
        "total_seconds": time.time() - started_at,
        "seeds": [report["seeds"][0] for report in seed_reports],
        "seed_runs": seed_reports,
        "budget": {
            "evolution_generations": config.execution.generations,
            "evolution_population_size": config.execution.population_size,
            "random_trials_per_seed": len(seed_reports[0]["random_baseline"]["trials"]),
        },
        "evolution": {
            "status": "completed",
            "best": evolution_best,
            "best_by_fitness": evolution_best,
        },
        "base_model_baseline": base,
        "fixed_baseline": fixed,
        "random_baseline": {
            "best": random_best,
            "trials_by_seed": [
                report["random_baseline"]["trials"] for report in seed_reports
            ],
        },
        "held_out": held_out,
        "artifacts": {
            "seed_reports": [
                report["artifacts"]["proof_report"] for report in seed_reports
            ]
        },
    }
    report["proof_quality"] = _proof_quality_summary(report)
    return report


def _mean_fitness_record(records: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    """Return a mean-fitness summary for equivalent records across proof seeds."""
    if not records:
        raise ValueError("FAIL-FAST: cannot aggregate empty proof record list")
    fitness_values = [float(record["fitness"]) for record in records]
    return {
        "candidate_id": candidate_id,
        "fitness": sum(fitness_values) / len(fitness_values),
        "seed_records": records,
    }


def _base_model_config() -> LoRAConfig:
    return LoRAConfig(
        r=1,
        alpha=1,
        dropout=0.0,
        target_modules=(),
        adapter_type="none",
    )


def _fixed_lora_config(config: CoralConfig) -> LoRAConfig:
    return LoRAConfig(
        r=config.evo.rank_candidates[0],
        alpha=config.evo.alpha_candidates[0],
        dropout=config.evo.dropout_candidates[0],
        target_modules=tuple(config.evo.target_modules),
        adapter_type="lora",
    )


def _sample_lora_configs(config: CoralConfig, count: int) -> list[LoRAConfig]:
    choices = list(
        itertools.product(
            config.evo.rank_candidates,
            config.evo.alpha_candidates,
            config.evo.dropout_candidates,
        )
    )
    rng = Random(config.seed + 104729)
    rng.shuffle(choices)
    selected = choices[: min(count, len(choices))]
    return [
        LoRAConfig(
            r=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=tuple(config.evo.target_modules),
            adapter_type="lora",
        )
        for rank, alpha, dropout in selected
    ]


def _evaluate_candidate(
    config: CoralConfig,
    model_factory,
    problem: dict[str, Any],
    lora_cfg: LoRAConfig,
    candidate_id: str,
) -> dict[str, Any]:
    genome = Genome(
        seed=CASeed(grid=np.zeros((1, 1), dtype=int), rule=0, steps=1),
        lora_cfg=lora_cfg,
        id=candidate_id,
        run_id=config.cache.run_id,
    )
    runner = model_factory(lora_cfg, genome)
    if not isinstance(runner, GSM8KLoRARunner):
        raise TypeError("GSM8K proof requires GSM8KLoRARunner")

    metrics = runner.train_and_evaluate(problem)
    fitness = score_gsm8k_metrics(metrics, runner.settings.loss_weight)
    return {
        "candidate_id": candidate_id,
        "fitness": fitness,
        "lora": _lora_record(lora_cfg),
        "metrics": metrics.to_report_dict(),
    }


def _held_out_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Return a problem payload that evaluates against held-out rows."""
    held_out_rows = problem.get("held_out")
    if not held_out_rows:
        raise ValueError("FAIL-FAST: proof mode requires plugin-provided held_out rows")
    return {
        **problem,
        "eval": list(held_out_rows),
    }


def _best_genome_record(genome: Genome | None) -> dict[str, Any] | None:
    if genome is None:
        return None
    return {
        "genome_id": genome.id,
        "fitness": genome.fitness,
        "lora": _lora_record(genome.lora_cfg),
        "metadata": genome.metadata or {},
    }


def _lora_record(lora_cfg: LoRAConfig) -> dict[str, Any]:
    return {
        "r": lora_cfg.r,
        "alpha": lora_cfg.alpha,
        "dropout": lora_cfg.dropout,
        "target_modules": list(lora_cfg.target_modules),
        "adapter_type": lora_cfg.adapter_type,
    }


def _lora_config_from_record(record: dict[str, Any]) -> LoRAConfig:
    return LoRAConfig(
        r=int(record["r"]),
        alpha=float(record["alpha"]),
        dropout=float(record["dropout"]),
        target_modules=tuple(record["target_modules"]),
        adapter_type=str(record["adapter_type"]),
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _interpret(
    evolution_best: dict[str, Any] | None,
    base_record: dict[str, Any],
    fixed_record: dict[str, Any],
    random_best: dict[str, Any] | None,
) -> dict[str, Any]:
    evo_fitness = evolution_best["fitness"] if evolution_best else None
    base_fitness = base_record["fitness"]
    random_fitness = random_best["fitness"] if random_best else None
    fixed_fitness = fixed_record["fitness"]
    evo_accuracy = _exact_accuracy(evolution_best)
    base_accuracy = _exact_accuracy(base_record)
    fixed_accuracy = _exact_accuracy(fixed_record)
    random_accuracy = _exact_accuracy(random_best)
    return {
        "fitness_evolution_beats_base_model": evo_fitness is not None
        and evo_fitness > base_fitness,
        "fitness_evolution_beats_fixed": evo_fitness is not None
        and evo_fitness > fixed_fitness,
        "fitness_evolution_beats_random_best": evo_fitness is not None
        and random_fitness is not None
        and evo_fitness > random_fitness,
        "exact_accuracy_evolution_beats_base_model": evo_accuracy is not None
        and base_accuracy is not None
        and evo_accuracy > base_accuracy,
        "exact_accuracy_evolution_beats_fixed": evo_accuracy is not None
        and fixed_accuracy is not None
        and evo_accuracy > fixed_accuracy,
        "exact_accuracy_evolution_beats_random_best": evo_accuracy is not None
        and random_accuracy is not None
        and evo_accuracy > random_accuracy,
        "evolution_minus_base_model": (
            evo_fitness - base_fitness if evo_fitness is not None else None
        ),
        "evolution_minus_fixed": (
            evo_fitness - fixed_fitness if evo_fitness is not None else None
        ),
        "evolution_minus_random_best": (
            evo_fitness - random_fitness
            if evo_fitness is not None and random_fitness is not None
            else None
        ),
        "exact_accuracy": {
            "evolution_best": evo_accuracy,
            "base_model": base_accuracy,
            "fixed": fixed_accuracy,
            "random_best": random_accuracy,
        },
    }


def _exact_accuracy(record: dict[str, Any] | None) -> float | None:
    if record is None:
        return None
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            metrics = metadata.get("evaluation")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("exact_accuracy")
    return float(value) if value is not None else None


def _proof_quality_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-checkable summary of whether a report supports a strong claim."""
    verdict = validate_proof_quality(
        {
            "base": _required_report_value(report, "base_model_baseline"),
            "fixed": _required_report_value(report, "fixed_baseline"),
            "random": _required_report_value(report, "random_baseline", "best"),
            "evolved": _required_report_value(report, "evolution", "best"),
            "held_out": _required_report_value(report, "held_out"),
            "seeds": _required_report_value(report, "seeds"),
        }
    )
    return asdict(verdict)


def _required_report_value(report: dict[str, Any], *path: str) -> Any:
    current: Any = report
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(
                "FAIL-FAST: proof report missing required field "
                + ".".join(path)
            )
        current = current[key]
    return current


def _validate_proof_execution_policy(config: CoralConfig) -> tuple[int, ...]:
    """Validate framework-level proof execution requirements."""
    return validate_proof_execution_policy(config)


def _proof_seed_configs(config: CoralConfig) -> tuple[CoralConfig, ...]:
    """Return isolated configs for each configured proof seed."""
    proof_seeds = _validate_proof_execution_policy(config)
    configs = []
    base_output_dir = config.execution.output_dir
    base_artifacts_dir = config.cache.artifacts_dir
    base_run_id = config.cache.run_id or "proof"
    for seed in proof_seeds:
        payload = config.model_dump(mode="python")
        payload["seed"] = seed
        payload["execution"]["output_dir"] = base_output_dir / f"seed_{seed}"
        payload["cache"]["artifacts_dir"] = base_artifacts_dir / f"seed_{seed}"
        payload["cache"]["run_id"] = f"{base_run_id}_seed_{seed}"
        configs.append(CoralConfig.model_validate(payload))
    return tuple(configs)


def _best_evolution_candidate_by_exact_accuracy(
    candidate_jsonl: Path,
    start_offset: int = 0,
) -> dict[str, Any] | None:
    """Return the strongest exact-accuracy candidate logged for this proof run."""
    if not candidate_jsonl.exists():
        return None

    best: dict[str, Any] | None = None
    best_accuracy = -1.0
    with candidate_jsonl.open() as handle:
        handle.seek(start_offset)
        lines = handle.readlines()

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        evaluation = metadata.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        accuracy = evaluation.get("exact_accuracy")
        if accuracy is None or float(accuracy) < best_accuracy:
            continue
        best_accuracy = float(accuracy)
        best = {
            "genome_id": record.get("genome_id"),
            "generation": record.get("generation"),
            "fitness": record.get("fitness"),
            "lora": record.get("lora"),
            "metrics": evaluation,
        }

    return best


def _file_size(path: Path) -> int:
    """Return the current file size, or zero if the file is absent."""
    return path.stat().st_size if path.exists() else 0
