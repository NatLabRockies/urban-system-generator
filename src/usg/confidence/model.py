"""Fast per-attribute confidence from precomputed per-target models (``usg-confidence-calibration/v2``).

Each bundle ships, for every imputable characteristic, two small logistic models fit on held-out buildings:

* **expected accuracy** ``P(correct | input pattern)`` — how often this characteristic is right for users who know
  the same facts;
* **confidence** ``P(correct | model probability of the delivered value, input pattern)`` — the calibrated chance
  that *this* building's value is right.

The input pattern is: one bit per GeoJSON fact (e.g. *Vintage* = Vintage and Vintage ACS), a location bit (State
known), ``log(1 + other known characteristics)`` and all pairwise fact interactions. Evaluating both models is a few
matrix products, so reports add no model passes. Toggling one fact bit yields *drivers* (known facts that matter
most) and *suggestions* (missing facts that would raise expected accuracy most).

The model training pipeline uses this module to fit the models, so both sides share one feature definition.
"""

from __future__ import annotations

import itertools

import numpy as np

from usg.constants import CALIBRATION_V2_FORMAT, STATE_COLUMN

DEFAULT_BANDS = {"high": 0.8, "medium": 0.5}


def pair_index(n_facts: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(n_facts), 2))


def feature_names(fact_names: list[str]) -> list[str]:
    pairs = [f"pair:{fact_names[i]}|{fact_names[j]}" for i, j in pair_index(len(fact_names))]
    return [f"fact:{n}" for n in fact_names] + ["location", "log1p_other_known"] + pairs


def build_features(fact_bits: np.ndarray, location: np.ndarray, n_other: np.ndarray) -> np.ndarray:
    """``(N, F)`` fact bits, ``(N,)`` location bit and other-known count -> ``(N, F + 2 + F(F-1)/2)`` features."""
    bits = np.asarray(fact_bits, dtype=np.float64)
    n, f = bits.shape
    parts = [bits, np.asarray(location, dtype=np.float64).reshape(n, 1),
             np.log1p(np.asarray(n_other, dtype=np.float64)).reshape(n, 1)]
    pairs = pair_index(f)
    if pairs:
        a = np.array([i for i, _ in pairs])
        b = np.array([j for _, j in pairs])
        parts.append(bits[:, a] * bits[:, b])
    return np.concatenate(parts, axis=1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def band(value: float | None, bands: dict[str, float]) -> str:
    if value is None or np.isnan(value):
        return "unknown"
    if value >= bands["high"]:
        return "high"
    return "medium" if value >= bands["medium"] else "low"


class ConfidenceModel:
    """Evaluates a ``usg-confidence-calibration/v2`` table against a bundle schema."""

    def __init__(self, raw: dict, schema):
        if raw.get("schema") != CALIBRATION_V2_FORMAT:
            raise ValueError(f"not a {CALIBRATION_V2_FORMAT} table: {raw.get('schema')!r}")
        self.raw = raw
        self.schema = schema
        self.fact_names = [f["name"] for f in raw["facts"]]
        cat_index = {c.name: j for j, c in enumerate(schema.cat_columns)}
        self.fact_columns = [[cat_index[c] for c in f["columns"] if c in cat_index] for f in raw["facts"]]
        fact_cols = {j for cols in self.fact_columns for j in cols}
        self.location_index = cat_index.get(raw.get("location_column", STATE_COLUMN))
        imputable = schema.cat_imputable
        self.other_mask = np.array([bool(imputable[j]) and j not in fact_cols for j in range(len(schema.cat_columns))])
        self.bands = raw.get("bands") or DEFAULT_BANDS
        self.validation = raw.get("validation") or {}
        n_feat = len(feature_names(self.fact_names))
        self.targets = [t for t in raw.get("by_target", {}) if t in cat_index]
        self.target_index = {t: i for i, t in enumerate(self.targets)}
        T = len(self.targets)
        self.E, self.E0 = np.zeros((T, n_feat)), np.zeros(T)
        self.C, self.Cp, self.C0 = np.zeros((T, n_feat)), np.zeros(T), np.zeros(T)
        self.const_expected = np.full(T, np.nan)
        self.const_confidence = np.full(T, np.nan)
        self.baseline = np.full(T, np.nan)
        for i, t in enumerate(self.targets):
            entry = raw["by_target"][t]
            self.baseline[i] = entry.get("baseline", np.nan)
            ex, cf = entry.get("expected", {}), entry.get("confidence", {})
            if "constant" in ex:
                self.const_expected[i] = ex["constant"]
            else:
                self._check(t, ex["coef"], n_feat)
                self.E[i], self.E0[i] = ex["coef"], ex["intercept"]
            if "constant" in cf:
                self.const_confidence[i] = cf["constant"]
            else:
                self._check(t, cf["coef"], n_feat)
                self.C[i], self.Cp[i], self.C0[i] = cf["coef"], cf["prob_coef"], cf["intercept"]

    @staticmethod
    def _check(target: str, coef: list, n_feat: int) -> None:
        if len(coef) != n_feat:
            raise ValueError(f"confidence model for {target!r} has {len(coef)} coefficients, expected {n_feat}")

    # ---- input pattern -------------------------------------------------------------------------------
    def pattern(self, cat_known: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = cat_known.shape[0]
        bits = np.stack(
            [cat_known[:, cols].any(axis=1) if cols else np.zeros(n, dtype=bool) for cols in self.fact_columns], axis=1
        ) if self.fact_columns else np.zeros((n, 0), dtype=bool)
        location = cat_known[:, self.location_index] if self.location_index is not None else np.zeros(n, dtype=bool)
        n_other = (cat_known & self.other_mask[None, :]).sum(axis=1)
        return bits, location, n_other

    # ---- models ---------------------------------------------------------------------------------------
    def expected_accuracy(self, bits, location, n_other) -> np.ndarray:
        """``(N, T)`` expected accuracy for each row's input pattern."""
        X = build_features(bits, location, n_other)
        out = sigmoid(X @ self.E.T + self.E0)
        const = ~np.isnan(self.const_expected)
        out[:, const] = self.const_expected[const]
        return out

    def confidence(self, bits, location, n_other, p_delivered: np.ndarray) -> np.ndarray:
        """``(N, T)`` calibrated probability that each delivered value is correct."""
        X = build_features(bits, location, n_other)
        out = sigmoid(X @ self.C.T + self.C0 + self.Cp * logit(p_delivered))
        const = ~np.isnan(self.const_confidence)
        out[:, const] = self.const_confidence[const]
        return out

    def fact_effects(self, bits, location, n_other) -> np.ndarray:
        """``(N, F, T)``: expected accuracy with fact f known minus with it unknown (all else as given)."""
        n, f = bits.shape
        effects = np.zeros((n, f, len(self.targets)))
        for k in range(f):
            on, off = bits.copy(), bits.copy()
            on[:, k], off[:, k] = True, False
            effects[:, k, :] = self.expected_accuracy(on, location, n_other) - self.expected_accuracy(off, location, n_other)
        return effects
