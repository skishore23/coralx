"""Target-neutral objective vectors for experiment scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    """One normalized objective value and its scalarization weight."""

    key: str
    value: float
    weight: float
    label: str


@dataclass(frozen=True)
class ObjectiveVector:
    """Immutable target-neutral objective collection."""

    objectives: tuple[Objective, ...]

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float],
        weights: Mapping[str, float],
        labels: Mapping[str, str] | None = None,
    ) -> ObjectiveVector:
        """Build an objective vector from explicit values and weights."""
        if not values:
            raise ValueError("FAIL-FAST: objective values cannot be empty")

        value_keys = tuple(values.keys())
        weight_keys = set(weights.keys())
        missing_weights = tuple(key for key in value_keys if key not in weight_keys)
        if missing_weights:
            raise ValueError(
                "FAIL-FAST: objective vector missing weights for "
                + ", ".join(missing_weights)
            )

        unused_weights = tuple(key for key in weights.keys() if key not in values)
        if unused_weights:
            raise ValueError(
                "FAIL-FAST: objective vector has weights without values for "
                + ", ".join(unused_weights)
            )

        total_weight = sum(float(weights[key]) for key in value_keys)
        if total_weight <= 0.0:
            raise ValueError("FAIL-FAST: objective weights must sum to a positive value")

        objectives = tuple(
            Objective(
                key=key,
                value=_normalized_value(key, float(values[key])),
                weight=_non_negative_weight(key, float(weights[key])),
                label=(labels or {}).get(key, key),
            )
            for key in value_keys
        )
        return cls(objectives)

    def keys(self) -> tuple[str, ...]:
        """Return objective keys in deterministic insertion order."""
        return tuple(objective.key for objective in self.objectives)

    def to_dict(self) -> dict[str, float]:
        """Return objective values keyed by target-neutral objective id."""
        return {objective.key: objective.value for objective in self.objectives}

    def label_for(self, key: str) -> str:
        """Return a human label for an objective key."""
        for objective in self.objectives:
            if objective.key == key:
                return objective.label
        raise ValueError(f"FAIL-FAST: unknown objective key '{key}'")

    def weighted_fitness(self) -> float:
        """Return normalized weighted scalar fitness."""
        total_weight = sum(objective.weight for objective in self.objectives)
        if total_weight <= 0.0:
            raise ValueError("FAIL-FAST: objective weights must sum to a positive value")
        return (
            sum(objective.value * objective.weight for objective in self.objectives)
            / total_weight
        )


def _normalized_value(key: str, value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"FAIL-FAST: objective '{key}' value must be between 0.0 and 1.0"
        )
    return value


def _non_negative_weight(key: str, weight: float) -> float:
    if weight < 0.0:
        raise ValueError(f"FAIL-FAST: objective '{key}' weight must be non-negative")
    return weight
