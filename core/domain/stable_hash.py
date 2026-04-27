"""Stable hashing helpers for deterministic local evolution."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonicalize(value: Any) -> Any:
    """Convert common runtime values to JSON-stable structures."""
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, float):
        return round(value, 12)
    return value


def stable_digest(value: Any, length: int | None = None) -> str:
    """Return a deterministic SHA-256 hex digest for a JSON-like value."""
    payload = json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def stable_int(value: Any, modulo: int | None = None) -> int:
    """Return a deterministic integer hash for a JSON-like value."""
    integer = int(stable_digest(value), 16)
    return integer if modulo is None else integer % modulo
