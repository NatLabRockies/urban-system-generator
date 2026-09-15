"""Scalar confidence signals shared by the native and ensemble methods."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

DEFAULT_BANDS = {"high": 0.85, "medium": 0.60}

# Simulation outputs (produced downstream by URBANopt), never scored as inferred building characteristics.
DEFAULT_EXCLUDED_TARGETS = frozenset({"site_energy.total.energy_consumption.kwh"})


def band(score: float | None, bands: dict[str, float] | None = None) -> str:
    """Label a confidence score ``high`` / ``medium`` / ``low`` (``unknown`` when there is no score)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "unknown"
    bands = bands or DEFAULT_BANDS
    if score >= bands.get("high", DEFAULT_BANDS["high"]):
        return "high"
    if score >= bands.get("medium", DEFAULT_BANDS["medium"]):
        return "medium"
    return "low"


def vote_entropy(votes: Counter, total: int) -> float:
    """Entropy of a vote distribution normalized by the number of distinct voted values (0 = unanimous)."""
    if total <= 1 or len(votes) <= 1:
        return 0.0
    h = -sum((n / total) * math.log(n / total) for n in votes.values() if n > 0)
    return h / math.log(len(votes))


def probability_entropy(probs: np.ndarray) -> float:
    """Entropy of a probability vector normalized by ``log(len(probs))``."""
    k = len(probs)
    if k <= 1:
        return 0.0
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    return float(-(p * np.log(p)).sum() / math.log(k))


def coverage(target: str, known: set[str], sensitivity: dict[str, dict[str, float]]) -> float | None:
    """Importance-weighted fraction of a target's drivers the user supplied."""
    table = sensitivity.get(target)
    if not table:
        return None
    total = sum(max(0.0, float(w)) for w in table.values())
    if total <= 0:
        return None
    return sum(max(0.0, float(w)) for k, w in table.items() if k in known) / total


def blend(primary: float, coverage_score: float | None, calibrated: float | None) -> float:
    """Calibrated accuracy dominates when available; otherwise coverage acts as a light prior."""
    if calibrated is not None:
        return float(np.clip(0.7 * calibrated + 0.3 * primary, 0.0, 1.0))
    if coverage_score is not None:
        return float(np.clip(0.8 * primary + 0.2 * coverage_score, 0.0, 1.0))
    return float(np.clip(primary, 0.0, 1.0))
