"""Inference-time confidence (ported from urban-system-generator tests/test_confidence.py to fixture bundles)."""

import pandas as pd
import pytest

from usg.confidence import (
    Calibration,
    ConfidenceEstimator,
    band,
    build_calibration,
    load_sensitivity,
)
from usg.errors import InputValidationError
from usg.testing.fixtures import tiny_dataframe

KNOWN = {"Vintage": "1990s", "Heating Fuel": "Natural Gas", "State": "MD"}


@pytest.mark.parametrize("method", ["native", "ensemble", "both"])
def test_record_shape(gnn_imputer, method):
    est = ConfidenceEstimator(gnn_imputer, method=method, n_passes=12, drop_prob=0.35, seed=0)
    rec = est.confidence_for_building(KNOWN)
    assert rec["n_known"] == 3 and rec["method"] == method
    assert rec["summary"]["n_targets"] == len(rec["attributes"]) > 0
    for d in rec["attributes"].values():
        assert "band" in d and "value" in d and d["method"] == method
        assert d["confidence"] is None or 0.0 <= d["confidence"] <= 1.0
        for key in ("confidence", "band", "coverage", "calibrated_accuracy", "probability"):
            assert key in d
        if d["type"] == "categorical":
            for key in ("agreement", "margin", "entropy", "ensemble_top", "ensemble_alternatives"):
                assert key in d
            assert (d["probability"] is not None) == (method != "ensemble")
            assert (d["agreement"] is not None) == (method != "native")


def test_known_inputs_and_conditioning_never_reported(gnn_imputer):
    rec = ConfidenceEstimator(gnn_imputer, n_passes=8).confidence_for_building({"Vintage": "1990s"})
    assert "Vintage" not in rec["attributes"]
    assert "State" not in rec["attributes"]  # conditioning column: input only, never filled
    assert rec["known_inputs"] == ["Vintage"]


def test_context_inputs_are_separated(gnn_imputer):
    df = tiny_dataframe(5, seed=1).drop(columns=["Heating Fuel", "sqft"])
    result = gnn_imputer.impute(df)
    recs = ConfidenceEstimator(gnn_imputer).estimate(df, result, context_columns=["State"])
    assert recs[0]["context_inputs"] == ["State"] and "State" not in recs[0]["known_inputs"]
    assert set(recs[0]["attributes"]) == {"Heating Fuel", "sqft"}
    assert recs[0]["attributes"]["sqft"]["type"] == "numerical"
    value = result.values.at[0, "Heating Fuel"]
    assert recs[0]["attributes"]["Heating Fuel"]["value"] == value


@pytest.mark.parametrize(
    "sensitivity",
    [
        {"Windows": {"Vintage": 1.0}, "Heating Fuel": {"Vintage": 1.0, "Geometry Building Type RECS": 1.0}},
        {"schema": "usg-sensitivity/v1", "inputs": ["Vintage"],
         "by_target": {"Heating Fuel": {"Vintage": 0.5, "Geometry Building Type RECS": 0.5}}},
    ],
)
def test_coverage_uses_sensitivity(gnn_imputer, sensitivity):
    est = ConfidenceEstimator(gnn_imputer, n_passes=8, sensitivity=sensitivity, seed=1)
    rec = est.confidence_for_building({"Vintage": "1990s"})
    assert rec["attributes"]["Heating Fuel"]["coverage"] == pytest.approx(0.5)
    assert rec["attributes"]["Bedrooms"]["coverage"] is None
    assert load_sensitivity(sensitivity)["Heating Fuel"]["Vintage"] > 0


def test_ensemble_is_deterministic(gnn_imputer):
    a = ConfidenceEstimator(gnn_imputer, method="ensemble", n_passes=6, seed=3).confidence_for_building(KNOWN)
    b = ConfidenceEstimator(gnn_imputer, method="ensemble", n_passes=6, seed=3).confidence_for_building(KNOWN)
    assert a == b


def test_ensemble_chunks_cover_all_rows(gnn_imputer):
    df = tiny_dataframe(13, seed=2).drop(columns=["Bedrooms"])
    result = gnn_imputer.impute(df)
    est = ConfidenceEstimator(gnn_imputer, method="ensemble", n_passes=4)
    from usg.confidence.ensemble import iter_evidence_masking

    batch, _ = gnn_imputer.schema.encode(df)
    chunks = list(iter_evidence_masking(gnn_imputer, batch, n_passes=4, chunk_rows=16))
    assert [c.stop - c.start for c in chunks] == [4, 4, 4, 1]
    first = chunks[0].cat_codes[:, 0, :]  # pass 0 keeps full evidence: known codes pass through
    assert (first[batch.cat_known[:4]] == batch.cat[:4][batch.cat_known[:4]]).all()
    assert len(est.estimate(df, result)) == 13


def test_legacy_calibration_counts_all_known(gnn_imputer):
    legacy = {
        "schema": "usg-confidence-calibration/v1",
        "by_target": {
            "Heating Fuel": {"type": "categorical", "baseline": 0.5, "n_samples": 10,
                             "by_n_known": {"1": 0.2, "3": 0.9}, "key_drivers": [], "with_driver": {}},
            "sqft": {"type": "numerical", "mae": 10.0, "rmse": 12.0, "p90_abs_err": 20.0},
        },
    }
    cal = Calibration.load(legacy)
    schema = gnn_imputer.schema
    assert cal.n_known_convention == "all_known"
    assert cal.expected_accuracy("Heating Fuel", {"Vintage", "State", "Bedrooms"}, schema) == 0.9
    assert cal.expected_accuracy("Heating Fuel", {"Vintage", "State"}, schema) == 0.2  # nearest n (tie -> lower)
    assert cal.numeric_errors("sqft") == {"mae": 10.0, "rmse": 12.0, "p90_abs_err": 20.0}
    rec = ConfidenceEstimator(gnn_imputer, calibration=legacy).confidence_for_building(
        {"Vintage": "1990s", "State": "MD", "Bedrooms": "3"}
    )
    assert rec["attributes"]["Heating Fuel"]["calibrated_accuracy"] == 0.9
    assert rec["attributes"]["sqft"]["expected_error"]["mae"] == 10.0


def test_imputable_known_convention_patterns_and_drivers(gnn_imputer):
    schema = gnn_imputer.schema
    cal = Calibration({
        "schema": "usg-confidence-calibration/v1",
        "n_known_convention": "imputable_known",
        "bands": {"high": 0.5, "medium": 0.3},
        "by_pattern": {"geojson": {"inputs": ["Vintage", "Bedrooms"], "by_target": {"Heating Fuel": 0.77}}},
        "by_target": {
            "Heating Fuel": {"type": "categorical", "baseline": 0.4, "by_n_known": {"1": 0.3, "2": 0.6},
                             "key_drivers": ["Vintage"], "with_driver": {"Vintage": 0.5}},
            "sqft": {"type": "numeric", "mae": 1.0},
        },
    })
    assert cal.count_known({"State", "Vintage"}, schema) == 1
    assert cal.expected_accuracy("Heating Fuel", {"State", "Vintage"}, schema) == pytest.approx(0.4)  # mean(0.3, 0.5)
    assert cal.expected_accuracy("Heating Fuel", {"State", "Vintage", "Bedrooms"}, schema) == 0.77
    assert cal.expected_accuracy("Geometry Garage", {"Vintage"}, schema) is None
    assert cal.numeric_errors("sqft") == {"mae": 1.0}
    est = ConfidenceEstimator(gnn_imputer, calibration=cal)
    assert est.bands == {"high": 0.5, "medium": 0.3}
    with pytest.raises(InputValidationError):
        Calibration({"schema": "usg-confidence-calibration/v9"})
    with pytest.raises(InputValidationError):
        ConfidenceEstimator(gnn_imputer, method="mc-dropout")


def test_build_calibration_roundtrip(gnn_imputer):
    table = build_calibration(gnn_imputer, tiny_dataframe(120, seed=11), samples_per_row=4, max_rows=100, seed=0)
    assert table["schema"] == "usg-confidence-calibration/v1"
    assert table["n_known_convention"] == "imputable_known" and table["model_type"] == "gnn"
    assert table["n_rows"] == 100 and "State" not in table["by_target"]
    fuel = table["by_target"]["Heating Fuel"]
    assert fuel["type"] == "categorical" and 0 <= fuel["baseline"] <= 1
    assert all(1 <= int(k) <= 6 for k in fuel["by_n_known"])
    assert table["by_target"]["sqft"]["type"] == "numerical"
    cal = Calibration.load(table)
    assert cal.expected_accuracy("Heating Fuel", {"Vintage"}, gnn_imputer.schema) is not None


def test_tabddpm_native_warns_with_few_samples(tabddpm_imputer):
    df = pd.DataFrame([{"Vintage": "1990s", "State": "VA"}])
    est = ConfidenceEstimator(tabddpm_imputer)
    est.estimate(df, tabddpm_imputer.impute(df, n_samples=4))
    assert est.warnings and "16" in est.warnings[0]
    quiet = ConfidenceEstimator(tabddpm_imputer)
    rec = quiet.estimate(df, tabddpm_imputer.impute(df, n_samples=16))[0]
    assert quiet.warnings == []
    probs = [a["probability"] for a in rec["attributes"].values() if a["type"] == "categorical"]
    assert all(0 < p <= 1 for p in probs)


def test_band_thresholds():
    assert band(0.9) == "high" and band(0.6) == "medium" and band(0.1) == "low" and band(None) == "unknown"
    assert band(0.4, {"high": 0.8, "medium": 0.3}) == "medium"
