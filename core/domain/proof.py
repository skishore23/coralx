"""Proof-quality contracts for comparative experiment reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

REQUIRED_COMPARISONS = ("base", "fixed", "random", "evolved", "held_out")


@dataclass(frozen=True)
class ProofQualityVerdict:
    """Machine-checkable proof quality summary."""

    passes: bool
    evolved_beats_base: bool
    evolved_beats_fixed: bool
    evolved_beats_random: bool
    held_out_beats_fixed: bool
    held_out_beats_random: bool
    has_multi_seed_support: bool


def validate_proof_quality(report: Mapping[str, Any]) -> ProofQualityVerdict:
    """Validate that a report supports a serious evolutionary-search claim."""
    missing = tuple(key for key in REQUIRED_COMPARISONS if key not in report)
    if missing:
        raise ValueError(
            "FAIL-FAST: proof report missing comparative records: " + ", ".join(missing)
        )

    base = _fitness(report["base"], "base")
    fixed = _fitness(report["fixed"], "fixed")
    random_best = _fitness(report["random"], "random")
    evolved = _fitness(report["evolved"], "evolved")
    held_out = _fitness(report["held_out"], "held_out")
    seeds = tuple(report.get("seeds", ()))
    has_multi_seed_support = len(set(seeds)) >= 3

    evolved_beats_base = evolved > base
    evolved_beats_fixed = evolved > fixed
    evolved_beats_random = evolved > random_best
    held_out_beats_fixed = held_out > fixed
    held_out_beats_random = held_out > random_best

    return ProofQualityVerdict(
        passes=(
            evolved_beats_base
            and evolved_beats_fixed
            and evolved_beats_random
            and held_out_beats_fixed
            and held_out_beats_random
            and has_multi_seed_support
        ),
        evolved_beats_base=evolved_beats_base,
        evolved_beats_fixed=evolved_beats_fixed,
        evolved_beats_random=evolved_beats_random,
        held_out_beats_fixed=held_out_beats_fixed,
        held_out_beats_random=held_out_beats_random,
        has_multi_seed_support=has_multi_seed_support,
    )


def validate_proof_execution_policy(config: Any) -> tuple[int, ...]:
    """Validate framework-level proof execution requirements."""
    execution = config.execution
    if not execution.run_held_out_benchmark:
        raise ValueError(
            "FAIL-FAST: proof mode requires execution.run_held_out_benchmark=true"
        )
    proof_seeds = tuple(int(seed) for seed in execution.proof_seeds)
    if len(set(proof_seeds)) < 3:
        raise ValueError(
            "FAIL-FAST: proof mode requires at least three distinct execution.proof_seeds"
        )
    return proof_seeds


def _fitness(record: Any, name: str) -> float:
    if not isinstance(record, Mapping):
        raise ValueError(f"FAIL-FAST: proof record '{name}' must be a mapping")
    if "fitness" not in record:
        raise ValueError(f"FAIL-FAST: proof record '{name}' missing fitness")
    value = float(record["fitness"])
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"FAIL-FAST: proof record '{name}' fitness must be between 0.0 and 1.0"
        )
    return value
