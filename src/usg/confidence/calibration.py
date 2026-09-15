"""Loaders for the optional per-bundle artefacts ``confidence_calibration.json`` and ``sensitivity.json``.

Calibration (``usg-confidence-calibration/v1``)::

    {"schema": "usg-confidence-calibration/v1",
     "n_known_convention": "imputable_known" | "all_known",   # optional, legacy files: all_known
     "model_type": "gnn" | "tabddpm",                           # optional
     "bands": {"high": 0.85, "medium": 0.6},                    # optional
     "by_pattern": {"<name>": {"inputs": [col, ...], "by_target": {target: accuracy}}},   # optional
     "by_target": {
        "<target>": {"type": "categorical", "baseline": acc, "n_samples": n,
                     "by_n_known": {"<k>": acc}, "key_drivers": [col], "with_driver": {col: acc}},
        "<numeric target>": {"type": "numerical" | "numeric", "mae": x, "rmse": x, "p90_abs_err": x}}}

Sensitivity: ``{"schema": "usg-sensitivity/v1", "inputs": [...], "by_target": {target: {input: weight}}}`` or the
legacy bare ``{target: {input: weight}}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from usg.constants import CALIBRATION_FORMAT, SENSITIVITY_FORMAT
from usg.errors import InputValidationError

IMPUTABLE_KNOWN = "imputable_known"
ALL_KNOWN = "all_known"


def load_json_if_exists(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise InputValidationError(f"invalid JSON in {p}: {exc}") from None


def _as_dict(source: dict | str | Path | None, what: str) -> dict | None:
    if source is None or isinstance(source, dict):
        return source
    obj = load_json_if_exists(source)
    if obj is None:
        raise InputValidationError(f"{what} file not found: {source}")
    return obj


def load_sensitivity(source: dict | str | Path | None) -> dict[str, dict[str, float]] | None:
    obj = _as_dict(source, "sensitivity")
    if obj is None:
        return None
    if obj.get("schema") not in (None, SENSITIVITY_FORMAT):
        raise InputValidationError(f"unsupported sensitivity schema {obj.get('schema')!r}")
    table = obj.get("by_target", obj)
    return {t: {k: float(w) for k, w in weights.items()} for t, weights in table.items() if isinstance(weights, dict)}


class Calibration:
    def __init__(self, raw: dict):
        if raw.get("schema") not in (None, CALIBRATION_FORMAT):
            raise InputValidationError(f"unsupported calibration schema {raw.get('schema')!r}")
        if not isinstance(raw.get("by_target", {}), dict):
            raise InputValidationError("calibration 'by_target' must be an object")
        self.raw = raw
        self.by_target: dict[str, dict] = raw.get("by_target", {})
        self.n_known_convention = raw.get("n_known_convention", ALL_KNOWN)
        if self.n_known_convention not in (IMPUTABLE_KNOWN, ALL_KNOWN):
            raise InputValidationError(f"unknown calibration n_known_convention {self.n_known_convention!r}")
        self.model_type: str | None = raw.get("model_type")
        self.bands: dict[str, float] | None = raw.get("bands")
        self._patterns = [
            (frozenset(p.get("inputs", [])), p.get("by_target", {})) for p in (raw.get("by_pattern") or {}).values()
        ]

    @classmethod
    def load(cls, source: dict | str | Path | Calibration | None) -> Calibration | None:
        if source is None or isinstance(source, Calibration):
            return source
        obj = _as_dict(source, "calibration")
        return None if obj is None else cls(obj)

    def _counted(self, known: Iterable[str], schema) -> set[str]:
        known = set(known)
        if self.n_known_convention == IMPUTABLE_KNOWN and schema is not None:
            return {k for k in known if k in schema and schema.column(k).role == "imputable"}
        return known

    def count_known(self, known: Iterable[str], schema=None) -> int:
        """Number of known inputs under this table's convention (``imputable_known`` ignores conditioning)."""
        return len(self._counted(known, schema))

    def numeric_errors(self, target: str) -> dict | None:
        entry = self.by_target.get(target)
        if not entry or entry.get("type") not in ("numerical", "numeric"):
            return None
        return {k: entry[k] for k in ("mae", "rmse", "p90_abs_err") if k in entry}

    def expected_accuracy(self, target: str, known: Iterable[str], schema=None) -> float | None:
        """Expected hit rate for ``target`` given the known inputs: exact pattern > n-known curve > baseline."""
        entry = self.by_target.get(target)
        counted = self._counted(known, schema)
        for inputs, table in self._patterns:
            if inputs == counted and target in table:
                return float(table[target])
        if not entry or entry.get("type", "categorical") != "categorical":
            return None
        by_n = entry.get("by_n_known") or {}
        n = len(counted)
        acc = by_n.get(str(n))
        if acc is None and by_n:
            nearest = min((int(k) for k in by_n), key=lambda a: (abs(a - n), a))
            acc = by_n[str(nearest)]
        if acc is None:
            acc = entry.get("baseline")
        if acc is None:
            return None
        with_driver = entry.get("with_driver") or {}
        present = [with_driver[d] for d in entry.get("key_drivers") or [] if d in counted and d in with_driver]
        if present:
            acc = float(np.mean([acc, *present]))
        return float(acc)
