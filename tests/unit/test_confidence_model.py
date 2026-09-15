import numpy as np
import pandas as pd
import pytest
import torch

from usg.bundle import Bundle, write_bundle
from usg.confidence.estimator import ConfidenceEstimator
from usg.confidence.model import ConfidenceModel, build_features, feature_names, pair_index
from usg.constants import CALIBRATION_V2_FORMAT
from usg.models import build_model
from usg.testing.fixtures import tiny_config, tiny_dataframe, tiny_schema

FACTS = [
    {"name": "Geometry Building Type RECS", "columns": ["Geometry Building Type RECS"]},
    {"name": "Vintage", "columns": ["Vintage"]},
    {"name": "Bedrooms", "columns": ["Bedrooms"]},
]


def _coef(**weights):
    names = feature_names([f["name"] for f in FACTS])
    coef = [0.0] * len(names)
    for key, value in weights.items():
        coef[names.index(key)] = value
    return coef


def _table():
    return {
        "schema": CALIBRATION_V2_FORMAT,
        "facts": FACTS,
        "location_column": "State",
        "bands": {"high": 0.8, "medium": 0.5},
        "validation": {"ece_model_probability": 0.04, "ece_confidence": 0.01},
        "by_target": {
            "Heating Fuel": {
                "type": "categorical", "baseline": 0.5,
                "expected": {"intercept": -0.5, "coef": _coef(location=1.0, **{"fact:Vintage": 0.8})},
                "confidence": {"intercept": 0.0, "prob_coef": 1.0, "coef": _coef(location=0.5)},
            },
            "Geometry Garage": {
                "type": "categorical", "baseline": 0.3,
                "expected": {"intercept": -1.0, "coef": _coef(**{"fact:Geometry Building Type RECS": 2.0})},
                "confidence": {"constant": 0.9},
            },
        },
    }


def test_features_and_pairs():
    bits = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
    X = build_features(bits, np.array([True, False]), np.array([0, 3]))
    assert X.shape == (2, 3 + 2 + 3)
    assert pair_index(3) == [(0, 1), (0, 2), (1, 2)]
    np.testing.assert_allclose(X[0], [1, 0, 1, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(X[1, 4], np.log1p(3))


def test_pattern_expected_accuracy_and_effects():
    schema = tiny_schema()
    model = ConfidenceModel(_table(), schema)
    df = pd.DataFrame({"State": ["MD", None], "Geometry Building Type RECS": ["Single-Family Detached", None],
                       "Vintage": [None, "1990s"], "Geometry Garage": [None, "None"]})
    batch, _ = schema.encode(df)
    bits, location, n_other = model.pattern(batch.cat_known)
    assert bits.tolist() == [[True, False, False], [False, True, False]]
    assert location.tolist() == [True, False] and n_other.tolist() == [0, 1]  # garage is a non-fact known column
    expected = model.expected_accuracy(bits, location, n_other)
    fuel, garage = model.target_index["Heating Fuel"], model.target_index["Geometry Garage"]
    assert expected[0, garage] > expected[1, garage]  # building type known raises garage accuracy
    effects = model.fact_effects(bits, location, n_other)
    assert effects.shape == (2, 3, 2)
    assert effects[0, 1, fuel] > 0 and effects[0, 2, fuel] == pytest.approx(0)
    conf = model.confidence(bits, location, n_other, np.array([[0.9, 0.2], [0.2, 0.2]]))
    assert conf[0, fuel] > conf[1, fuel] and conf[:, garage].tolist() == [0.9, 0.9]


def test_model_rejects_mismatched_coefficients():
    table = _table()
    table["by_target"]["Heating Fuel"]["expected"]["coef"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="coefficients"):
        ConfidenceModel(table, tiny_schema())


def test_estimator_uses_v2_table(tmp_path):
    torch.manual_seed(0)
    schema, config = tiny_schema(), tiny_config("gnn")
    path = write_bundle(
        tmp_path / "gnn-MD-2024.2-v1", model=build_model(schema, config), schema=schema, config=config,
        manifest={"model_id": "gnn-MD-2024.2-v1"}, metrics={}, model_card="# gnn-MD-2024.2-v1\n",
        selftest_inputs=tiny_dataframe(4, seed=1)[["State", "Vintage"]],
    )
    imputer = Bundle.open(path).load_imputer()
    df = tiny_dataframe(6, seed=2)[["State", "Geometry Building Type RECS"]]
    result = imputer.impute(df)
    estimator = ConfidenceEstimator(imputer, calibration=_table())
    records = estimator.estimate(df, result)
    assert len(records) == 6 and estimator.validation["ece_confidence"] == 0.01
    rec = records[0]
    garage, fuel, beds = (rec["attributes"][c] for c in ("Geometry Garage", "Heating Fuel", "Bedrooms"))
    assert garage["confidence"] == 0.9 and garage["band"] == "high" and garage["expected_accuracy"] > 0.5
    assert {"fact": "Geometry Building Type RECS", "effect": pytest.approx(garage["drivers"][0]["effect"])} == garage["drivers"][0]
    assert 0 < fuel["confidence"] < 1 and fuel["suggestions"][0]["fact"] == "Vintage"
    assert "drivers" not in beds and beds["confidence"] is not None  # no v2 model for Bedrooms -> native fallback
    assert rec["known_facts"] == ["Geometry Building Type RECS"]
    assert rec["suggested_facts"][0]["fact"] == "Vintage"
    assert rec["summary"]["mean_expected_accuracy"] is not None
