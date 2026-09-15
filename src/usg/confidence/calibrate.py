"""Build a ``usg-confidence-calibration/v1`` table from fully labeled rows (offline, once per bundle).

For every labeled row, ``samples_per_row`` random subsets of its imputable columns are kept known (conditioning
columns always stay known), the rest are imputed, and hits are aggregated per target, per number of known imputable
inputs (``n_known_convention = "imputable_known"``) and per present driver column.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from usg.confidence.calibration import IMPUTABLE_KNOWN
from usg.constants import CALIBRATION_FORMAT
from usg.errors import InputValidationError
from usg.schema import EncodedBatch


def _rate(hits: float, total: float) -> float | None:
    return round(float(hits / total), 4) if total else None


def build_calibration(
    imputer,
    labeled_df: pd.DataFrame,
    samples_per_row: int = 8,
    max_rows: int = 2000,
    min_known: int = 1,
    seed: int = 0,
    *,
    n_samples: int | None = None,
    batch_size: int = 4096,
    min_driver_support: int = 30,
    ignore_columns: Iterable[str] = (),
) -> dict:
    schema = imputer.schema
    df = labeled_df
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    batch, _ = schema.encode(df, ignore_columns=[c for c in df.columns if c not in schema] + list(ignore_columns))

    names = [c.name for c in schema.cat_columns] + [c.name for c in schema.num_columns]
    n_cat = len(schema.cat_columns)
    present = np.concatenate([batch.cat_known, batch.num_known], axis=1)
    imputable = np.concatenate([schema.cat_imputable, schema.num_imputable])

    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(len(df)), samples_per_row)
    known = np.zeros((len(rows), len(names)), dtype=bool)
    valid = np.zeros(len(rows), dtype=bool)
    for r, i in enumerate(rows):
        candidates = np.flatnonzero(present[i] & imputable)
        if len(candidates) <= min_known:
            continue
        k = int(rng.integers(min_known, len(candidates)))
        known[r, rng.choice(candidates, size=k, replace=False)] = True
        valid[r] = True
    if not valid.any():
        raise InputValidationError(f"no labeled rows with more than {min_known} known imputable columns")
    known |= present[rows] & ~imputable[None, :]  # conditioning inputs stay known
    rows, known = rows[valid], known[valid]

    cat_known, num_known = known[:, :n_cat], known[:, n_cat:]
    truth_cat, truth_num = batch.cat[rows], batch.num[rows]
    masked = EncodedBatch(
        np.where(cat_known, truth_cat, -1),
        cat_known,
        np.where(num_known, truth_num, 0.0).astype(np.float32),
        num_known,
    )
    raw = imputer.impute_encoded(masked, n_samples=n_samples, seed=seed, batch_size=batch_size)

    known_imputable = known & imputable[None, :]
    n_known = known_imputable.sum(axis=1)
    known_f = known_imputable.astype(np.float64)
    by_target: dict[str, dict] = {}
    for j, spec in enumerate(schema.cat_columns):
        hidden = ~cat_known[:, j] & batch.cat_known[rows, j]
        if spec.role != "imputable" or not hidden.any():
            continue
        correct = (raw.cat_codes[:, j] == truth_cat[:, j]) & hidden
        base = float(correct.sum() / hidden.sum())
        totals = np.bincount(n_known[hidden], minlength=n_known.max() + 1)
        hits = np.bincount(n_known[hidden], weights=correct[hidden].astype(float), minlength=n_known.max() + 1)
        support = known_f.T @ hidden.astype(np.float64)
        acc_with = np.divide(known_f.T @ correct.astype(np.float64), support, out=np.zeros_like(support), where=support > 0)
        lifts = sorted(
            ((names[d], float(acc_with[d]) - base) for d in np.flatnonzero(support >= min_driver_support) if d != j),
            key=lambda x: x[1],
            reverse=True,
        )
        drivers = [name for name, _ in lifts[:5]]
        by_target[spec.name] = {
            "type": "categorical",
            "baseline": round(base, 4),
            "n_samples": int(hidden.sum()),
            "by_n_known": {str(k): _rate(hits[k], totals[k]) for k in np.flatnonzero(totals)},
            "key_drivers": drivers,
            "with_driver": {name: round(float(acc_with[names.index(name)]), 4) for name in drivers},
        }
    for j, spec in enumerate(schema.num_columns):
        hidden = ~num_known[:, j] & batch.num_known[rows, j]
        if spec.role != "imputable" or not hidden.any():
            continue
        pred = schema.inverse_numeric(spec.name, raw.num_values[hidden, j])
        truth = schema.inverse_numeric(spec.name, truth_num[hidden, j])
        errs = np.abs(pred - truth)
        by_target[spec.name] = {
            "type": "numerical",
            "mae": round(float(errs.mean()), 4),
            "rmse": round(float(np.sqrt((errs**2).mean())), 4),
            "p90_abs_err": round(float(np.percentile(errs, 90)), 4),
            "n_samples": int(errs.size),
        }

    bundle = getattr(imputer, "bundle", None)
    return {
        "schema": CALIBRATION_FORMAT,
        "n_rows": len(df),
        "samples_per_row": int(samples_per_row),
        "n_known_convention": IMPUTABLE_KNOWN,
        "model_type": getattr(bundle, "model_type", None),
        "model_id": getattr(bundle, "model_id", None),
        "by_target": by_target,
    }
