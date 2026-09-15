"""Per-building, per-filled-value confidence records.

Methods:

* ``native`` (default) — the model's own distribution (G-NN softmax; TabDDPM smoothed sample frequencies). When the
  bundle ships a ``usg-confidence-calibration/v2`` table (see :mod:`usg.confidence.model`), each value gets a
  calibrated confidence that depends on this building's model probability *and* on which facts the user supplied,
  plus the expected accuracy for that input pattern, the known facts that drove it and the missing facts that would
  help most. No extra model passes are needed.
* ``ensemble`` — evidence-masking stability (see :mod:`usg.confidence.ensemble`): the share of passes that
  reproduce the reported value (``agreement``), vote margin and entropy.
* ``both`` — primary score is the mean of probability and agreement.

Without a v2 table the primary score is blended with evidence coverage (sensitivity table) and calibrated accuracy
(legacy v1 calibration table). Reported values are never changed. Conditioning columns are inputs only and are never
scored.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from usg.confidence.calibration import Calibration, load_json_if_exists, load_sensitivity
from usg.confidence.ensemble import EnsembleChunk, iter_evidence_masking
from usg.confidence.model import ConfidenceModel
from usg.confidence.signals import (
    DEFAULT_BANDS,
    DEFAULT_EXCLUDED_TARGETS,
    band,
    blend,
    coverage,
    probability_entropy,
    vote_entropy,
)
from usg.constants import CALIBRATION_V2_FORMAT
from usg.errors import InputValidationError

logger = logging.getLogger(__name__)

METHODS = ("native", "ensemble", "both")
Z95 = 1.6448536
MIN_NATIVE_SAMPLES = 16
MIN_EFFECT = 0.01  # drivers / suggestions smaller than one accuracy point are not reported


def _r(x: float | None, digits: int = 3) -> float | None:
    return None if x is None else round(float(x), digits)


@dataclass
class _ModelScores:
    """v2 confidence-model outputs for one ``estimate`` call (rows x targets)."""

    bits: np.ndarray  # (N, F) fact known
    confidence: np.ndarray  # (N, T)
    expected: np.ndarray  # (N, T)
    effects: np.ndarray  # (N, F, T)


class ConfidenceEstimator:
    def __init__(
        self,
        imputer,
        method: str = "native",
        n_passes: int = 24,
        drop_prob: float = 0.35,
        seed: int = 0,
        sensitivity: dict | str | Path | None = None,
        calibration: dict | str | Path | Calibration | None = None,
        exclude_targets: Iterable[str] | None = None,
        bands: dict[str, float] | None = None,
    ):
        if method not in METHODS:
            raise InputValidationError(f"unknown confidence method {method!r}; choose one of {METHODS}")
        self.imputer = imputer
        self.schema = imputer.schema
        self.method = method
        self.n_passes = int(n_passes)
        self.drop_prob = float(drop_prob)
        self.seed = int(seed)
        self.sensitivity = load_sensitivity(sensitivity) or {}
        raw = calibration
        if isinstance(calibration, (str, Path)):
            raw = load_json_if_exists(calibration)
            if raw is None:
                raise InputValidationError(f"calibration file not found: {calibration}")
        self.model: ConfidenceModel | None = None
        self.calibration: Calibration | None = None
        if isinstance(raw, dict) and raw.get("schema") == CALIBRATION_V2_FORMAT:
            self.model = ConfidenceModel(raw, self.schema)
        else:
            self.calibration = Calibration.load(raw)
        default_bands = (self.model.bands if self.model else None) or (
            self.calibration.bands if self.calibration and self.calibration.bands else DEFAULT_BANDS
        )
        self.bands = bands or default_bands
        self.exclude_targets = set(DEFAULT_EXCLUDED_TARGETS if exclude_targets is None else exclude_targets)
        self.warnings: list[str] = []
        self._vocab_index = {c.name: {v: i for i, v in enumerate(c.vocab)} for c in self.schema.cat_columns}

    @classmethod
    def for_imputer(cls, imputer, **kwargs) -> ConfidenceEstimator:
        """Like the constructor, but defaults sensitivity/calibration to the bundle's own artefacts."""
        bundle = getattr(imputer, "bundle", None)
        if bundle is not None:
            if kwargs.get("sensitivity") is None:
                kwargs["sensitivity"] = bundle.sensitivity()
            if kwargs.get("calibration") is None:
                kwargs["calibration"] = bundle.calibration()
        return cls(imputer, **kwargs)

    @property
    def uses_native(self) -> bool:
        return self.method in ("native", "both")

    @property
    def uses_ensemble(self) -> bool:
        return self.method in ("ensemble", "both")

    @property
    def uses_model(self) -> bool:
        return self.model is not None and self.method == "native"

    @property
    def validation(self) -> dict:
        """Held-out reliability of the confidence model, for report headers."""
        return dict(self.model.validation) if self.model is not None else {}

    # ---- public API ------------------------------------------------------------------------------
    def estimate(self, df_known: pd.DataFrame, result, *, context_columns: Iterable[str] = ()) -> list[dict]:
        """One confidence record per row of ``df_known`` (the frame ``result`` was imputed from)."""
        schema = self.schema
        df = df_known.reset_index(drop=True)
        batch, _ = schema.encode(df, ignore_columns=[c for c in df.columns if c not in schema])
        model_type = getattr(getattr(self.imputer, "bundle", None), "model_type", None)
        if self.uses_native and model_type == "tabddpm" and result.n_samples < MIN_NATIVE_SAMPLES:
            msg = (
                f"TabDDPM native confidence from only {result.n_samples} samples is coarse; "
                f"use --samples {MIN_NATIVE_SAMPLES} or more"
            )
            if msg not in self.warnings:
                self.warnings.append(msg)
                logger.warning(msg)

        # positional numpy columns: per-cell DataFrame.at lookups dominated report time
        values = {c: result.values[c].to_numpy(dtype=object) for c in result.values.columns}
        imputed = {c: result.imputed[c].to_numpy(dtype=bool) for c in result.imputed.columns}
        context = set(context_columns)
        scores = self._model_scores(batch, values, len(df), result) if self.uses_model else None
        if self.uses_ensemble:
            chunks: Iterable[EnsembleChunk | None] = iter_evidence_masking(
                self.imputer, batch, n_passes=self.n_passes, drop_prob=self.drop_prob, seed=self.seed
            )
        else:
            chunks = [None]
        records: list[dict] = []
        for chunk in chunks:
            rows = range(len(df)) if chunk is None else range(chunk.start, chunk.stop)
            for i in rows:
                offset = None if chunk is None else i - chunk.start
                records.append(self._building_record(i, batch, values, imputed, result, chunk, offset, context, scores))
        return records

    def confidence_for_building(self, known: dict, base_prediction: dict | None = None) -> dict:
        """Legacy single-building API: impute ``known`` and score the filled values."""
        known = {k: v for k, v in known.items() if not (v is None or (not isinstance(v, str) and pd.isna(v)))}
        df = pd.DataFrame([known], dtype=object)
        result = self.imputer.impute(df, seed=self.seed)
        record = self.estimate(df, result)[0]
        if base_prediction:
            for col, rec in record["attributes"].items():
                if col in base_prediction:
                    rec["value"] = base_prediction[col]
        return record

    # ---- v2 model scores ------------------------------------------------------------------------
    def _model_scores(self, batch, values: dict[str, np.ndarray], n: int, result) -> _ModelScores:
        model = self.model
        p = np.full((n, len(model.targets)), 0.5)
        for t, name in enumerate(model.targets):
            probs = result.probabilities.get(name)
            if probs is None:
                continue
            index = self._vocab_index[name]
            codes = np.array([index.get(v, -1) if isinstance(v, str) else -1 for v in values[name]])
            ok = codes >= 0
            p[ok, t] = probs[np.flatnonzero(ok), codes[ok]]
            p[~ok, t] = 0.0
        bits, location, n_other = model.pattern(batch.cat_known)
        return _ModelScores(
            bits=bits,
            confidence=model.confidence(bits, location, n_other, p),
            expected=model.expected_accuracy(bits, location, n_other),
            effects=model.fact_effects(bits, location, n_other),
        )

    # ---- records ---------------------------------------------------------------------------------
    def _building_record(self, i, batch, values, imputed, result, chunk, offset, context, scores) -> dict:
        schema = self.schema
        known = [c.name for j, c in enumerate(schema.cat_columns) if batch.cat_known[i, j]]
        known += [c.name for j, c in enumerate(schema.num_columns) if batch.num_known[i, j]]
        known_set = set(known)
        targets = sorted(c for c in schema.imputable_names if bool(imputed[c][i]) and c not in self.exclude_targets)
        attributes = {}
        for col in targets:
            spec = schema.column(col)
            if spec.type == "categorical":
                codes = None if chunk is None else chunk.cat_codes[offset, :, schema.cat_index(col)]
                attributes[col] = self._categorical(i, col, values[col][i], result, codes, known_set, scores)
            else:
                draws = None if chunk is None else chunk.num_values[offset, :, schema.num_index(col)]
                attributes[col] = self._numeric(i, col, values[col][i], result, draws, known_set)

        scored = [r for r in attributes.values() if r["confidence"] is not None]
        confs = [r["confidence"] for r in scored]
        summary = {
            "n_targets": len(attributes),
            "mean_confidence": _r(np.mean(confs)) if confs else None,
            "min_confidence": _r(np.min(confs)) if confs else None,
            "n_high": sum(r["band"] == "high" for r in attributes.values()),
            "n_medium": sum(r["band"] == "medium" for r in attributes.values()),
            "n_low": sum(r["band"] == "low" for r in attributes.values()),
        }
        record = {
            "known_inputs": sorted(k for k in known if k not in context),
            "context_inputs": sorted(k for k in known if k in context),
            "n_known": len(known),
            "method": self.method,
            "model_id": result.model_id,
            "attributes": attributes,
            "summary": summary,
        }
        if scores is not None:
            record.update(self._fact_summary(i, targets, scores))
            expected = [r["expected_accuracy"] for r in attributes.values() if r.get("expected_accuracy") is not None]
            summary["mean_expected_accuracy"] = _r(np.mean(expected)) if expected else None
        return record

    def _fact_summary(self, i: int, targets: list[str], scores: _ModelScores) -> dict:
        model = self.model
        bits = scores.bits[i]
        t_idx = [model.target_index[t] for t in targets if t in model.target_index]
        suggestions = []
        if t_idx:
            for f, name in enumerate(model.fact_names):
                if bits[f]:
                    continue
                gains = scores.effects[i, f, t_idx]
                mean_gain = float(gains.mean())
                if mean_gain >= MIN_EFFECT / 4:
                    suggestions.append({
                        "fact": name,
                        "mean_gain": _r(mean_gain),
                        "n_attributes_improved": int((gains >= MIN_EFFECT).sum()),
                    })
        suggestions.sort(key=lambda s: -s["mean_gain"])
        return {
            "known_facts": [n for f, n in enumerate(model.fact_names) if bits[f]],
            "missing_facts": [n for f, n in enumerate(model.fact_names) if not bits[f]],
            "suggested_facts": suggestions[:3],
        }

    def _categorical(self, i, col, value, result, codes, known, scores) -> dict:
        vocab = self.schema.column(col).vocab
        probability = margin = entropy = agreement = None
        top = None
        alternatives: list[dict] = []
        if self.uses_native:
            p = result.probabilities[col][i]
            order = np.argsort(-p, kind="stable")
            idx = self._vocab_index[col].get(value)
            probability = float(p[idx]) if idx is not None else 0.0
            margin = float(p[order[0]] - (p[order[1]] if len(order) > 1 else 0.0))
            entropy = probability_entropy(p)
            top = vocab[order[0]]
            alternatives = [{"value": vocab[k], "share": _r(p[k])} for k in order[:4] if vocab[k] != value][:3]
        if self.uses_ensemble:
            votes = Counter(vocab[c] for c in codes if 0 <= c < len(vocab))
            total = sum(votes.values())
            if total:
                ranked = votes.most_common()
                agreement = votes.get(value, 0) / total
                top = ranked[0][0]
                alternatives = [{"value": v, "share": _r(n / total)} for v, n in ranked if v != value][:3]
                if not self.uses_native:
                    margin = (ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)) / total
                    entropy = vote_entropy(votes, total)

        record = {
            "value": value,
            "type": "categorical",
            "method": self.method,
            "probability": _r(probability),
            "agreement": _r(agreement),
            "margin": _r(margin),
            "entropy": _r(entropy),
            "ensemble_top": top,
            "ensemble_alternatives": alternatives,
        }
        t = self.model.target_index.get(col) if (scores is not None and self.model is not None) else None
        if t is not None:
            conf = float(scores.confidence[i, t])
            effects = scores.effects[i, :, t]
            names = self.model.fact_names
            drivers = sorted(
                ({"fact": names[f], "effect": _r(effects[f])} for f in range(len(names)) if scores.bits[i, f] and effects[f] >= MIN_EFFECT),
                key=lambda d: -d["effect"],
            )[:3]
            suggestions = sorted(
                ({"fact": names[f], "gain": _r(effects[f])} for f in range(len(names)) if not scores.bits[i, f] and effects[f] >= MIN_EFFECT),
                key=lambda d: -d["gain"],
            )[:3]
            baseline = self.model.baseline[t]
            record.update({
                "confidence": _r(conf),
                "band": band(conf, self.bands),
                "expected_accuracy": _r(scores.expected[i, t]),
                "baseline_accuracy": None if np.isnan(baseline) else _r(baseline),
                "drivers": drivers,
                "suggestions": suggestions,
                "coverage": None,
                "calibrated_accuracy": _r(scores.expected[i, t]),
            })
            return record

        primaries = [s for s in (probability, agreement) if s is not None]
        cov = coverage(col, known, self.sensitivity)
        cal = self.calibration.expected_accuracy(col, known, self.schema) if self.calibration else None
        conf = blend(float(np.mean(primaries)), cov, cal) if primaries else None
        record.update({
            "confidence": _r(conf),
            "band": band(conf, self.bands),
            "coverage": _r(cov),
            "calibrated_accuracy": _r(cal),
        })
        return record

    def _numeric(self, i, col, value, result, draws, known) -> dict:
        stabilities = []
        mean = std = q05 = q95 = None
        if self.uses_native:
            stats = result.numeric_stats[col]
            mean, q05, q95 = float(stats["mean"][i]), float(stats["q05"][i]), float(stats["q95"][i])
            std = max(0.0, (q95 - q05) / (2 * Z95))
            stabilities.append(max(0.0, 1.0 - std / (abs(mean) if abs(mean) > 1e-9 else 1.0)))
        if self.uses_ensemble and draws is not None and draws.size:
            mean, std = float(np.mean(draws)), float(np.std(draws))
            stabilities.append(max(0.0, 1.0 - std / (abs(mean) if abs(mean) > 1e-9 else 1.0)))
        stability = float(np.mean(stabilities)) if stabilities else None
        cov = coverage(col, known, self.sensitivity)
        conf = blend(stability, cov, None) if stability is not None else None
        record = {
            "value": None if value is None or pd.isna(value) else float(value),
            "type": "numerical",
            "method": self.method,
            "confidence": _r(conf),
            "band": band(conf, self.bands),
            "probability": None,
            "ensemble_mean": _r(mean, 4),
            "ensemble_std": _r(std, 4),
            "stability": _r(stability),
            "coverage": _r(cov),
            "calibrated_accuracy": None,
        }
        if q05 is not None:
            record["q05"], record["q95"] = _r(q05, 4), _r(q95, 4)
        errors = self.calibration.numeric_errors(col) if self.calibration else None
        if errors:
            record["expected_error"] = errors
        return record
