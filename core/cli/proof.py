"""Proof-report helpers for GSM8K LoRA runs."""

from __future__ import annotations

import asyncio
import itertools
import json
import time
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
from plugins.gsm8k_lora.plugin import GSM8KLoRARunner, score_gsm8k_metrics


def run_gsm8k_lora_proof(
    config: CoralConfig,
    random_trials: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run evolution plus fixed/random controls for the GSM8K LoRA target."""
    if config.experiment.target != "gsm8k_lora":
        raise ValueError(
            "The proof command currently supports experiment.target='gsm8k_lora' only"
        )

    started_at = time.time()
    candidate_jsonl = config.execution.output_dir / "candidate_evaluations.jsonl"
    candidate_jsonl_start = _file_size(candidate_jsonl)
    services = create_evolution_services(config)
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
            "loss_weight": config.experiment.evaluation.get("loss_weight", 1.0)
            if config.experiment.evaluation
            else 1.0,
        },
        "budget": {
            "evolution_generations": config.execution.generations,
            "evolution_population_size": config.execution.population_size,
            "random_trials": len(random_records),
        },
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
        "interpretation": _interpret(
            evolution_best, base_record, fixed_record, random_best
        ),
        "artifacts": {
            "candidate_jsonl": str(candidate_jsonl)
        },
    }

    destination = output_path or (config.execution.output_dir / "proof_report.json")
    report["artifacts"]["proof_report"] = str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


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
        "evolution_minus_base_model": evo_fitness - base_fitness
        if evo_fitness is not None
        else None,
        "evolution_minus_fixed": evo_fitness - fixed_fitness
        if evo_fitness is not None
        else None,
        "evolution_minus_random_best": evo_fitness - random_fitness
        if evo_fitness is not None and random_fitness is not None
        else None,
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
