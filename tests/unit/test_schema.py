import numpy as np
import pandas as pd
import pytest

from usg.errors import InputValidationError, SchemaError
from usg.schema import ColumnSpec, QuantileNormal, Schema, canonical_label
from usg.testing.fixtures import tiny_dataframe, tiny_schema


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
        ("", None),
        ("  ", None),
        ("None", "None"),
        (3, "3"),
        (3.0, "3"),
        ("3.0", "3"),
        (" 2 Car ", "2 Car"),
        (2.5, "2.5"),
        ("<1940", "<1940"),
    ],
)
def test_canonical_label(raw, expected):
    assert canonical_label(raw) == expected


def test_roundtrip_categorical_and_numeric():
    schema = tiny_schema()
    df = tiny_dataframe(64, seed=3)
    batch, issues = schema.encode(df)
    assert issues == []
    assert batch.cat_known.all() and batch.num_known.all()
    decoded = schema.decode(batch.cat, batch.num)
    cat_names = [c.name for c in schema.cat_columns]
    pd.testing.assert_frame_equal(decoded[cat_names], df[cat_names].reset_index(drop=True), check_dtype=False)
    np.testing.assert_allclose(decoded["sqft"], df["sqft"], rtol=0.02)


def test_none_is_a_value_and_numeric_strings_canonicalize():
    schema = tiny_schema(with_numeric=False)
    df = pd.DataFrame({"Geometry Garage": ["None", None, "2 car"], "Bedrooms": ["3.0", 4, ""]})
    batch, issues = schema.encode(df)
    garage = schema.cat_index("Geometry Garage")
    beds = schema.cat_index("Bedrooms")
    assert batch.cat[:, garage].tolist() == [3, -1, 1]  # "None" known, None missing, case-insensitive match
    assert batch.cat_known[:, garage].tolist() == [True, False, True]
    assert batch.cat[:, beds].tolist() == [2, 3, -1]
    assert issues == []


def test_vocab_spelled_as_float_matches_integer_input():
    schema = Schema([ColumnSpec("Ground Thermal Conductivity", "imputable", "categorical", vocab=("0.5", "2.0", "2.6"))])
    df = pd.DataFrame({"Ground Thermal Conductivity": ["2", "2.0", 2.0, "0.5", "3"]})
    batch, issues = schema.encode(df)
    assert batch.cat[:, 0].tolist() == [1, 1, 1, 0, -1]
    assert [i.kind for i in issues] == ["unknown_label"]
    assert schema.decode(batch.cat)["Ground Thermal Conductivity"].tolist()[:2] == ["2.0", "2.0"]


def test_unseen_integer_option_snaps_to_nearest_and_is_reported():
    schema = tiny_schema(with_numeric=False)  # Bedrooms vocab "1".."5"
    df = pd.DataFrame({"Bedrooms": ["7", 0, "3"], "Geometry Garage": ["4 Car", "None", "2 Car"]})
    batch, issues = schema.encode(df)
    beds, garage = schema.cat_index("Bedrooms"), schema.cat_index("Geometry Garage")
    assert batch.cat[:, beds].tolist() == [4, 0, 2] and batch.cat_known[:, beds].all()
    assert sorted((i.row, i.column, i.value, i.kind) for i in issues) == [
        (0, "Bedrooms", "7", "coerced"),
        (0, "Geometry Garage", "4 Car", "unknown_label"),  # text options never snap
        (1, "Bedrooms", "0", "coerced"),
    ]
    assert not batch.cat_known[0, garage]
    with pytest.raises(InputValidationError):  # only the unknown text label is an error
        schema.encode(df, unknown="error")
    schema.encode(df[["Bedrooms"]], unknown="error")


def test_unknown_labels_become_missing_and_are_reported():
    schema = tiny_schema(with_numeric=False)
    df = pd.DataFrame({"Geometry Building Type RECS": ["Other Category", "Single-Family Detached"], "Feature ID": [1, 2]})
    batch, issues = schema.encode(df, ignore_columns=["Building"])
    j = schema.cat_index("Geometry Building Type RECS")
    assert batch.cat_known[:, j].tolist() == [False, True]
    kinds = sorted((i.kind, i.column) for i in issues)
    assert kinds == [("unknown_column", "Feature ID"), ("unknown_label", "Geometry Building Type RECS")]
    with pytest.raises(InputValidationError):
        schema.encode(df, unknown="error")


def test_hashes_ignore_metadata_but_track_vocab():
    schema = tiny_schema(with_numeric=False)
    h, v = schema.schema_hash(), schema.vocab_hash()
    assert schema.with_temperatures({"Vintage": 1.3}).schema_hash() == h
    reloaded = Schema.from_json(schema.to_json())
    assert reloaded.schema_hash() == h and reloaded.vocab_hash() == v
    changed = Schema([*schema.columns[:-1], ColumnSpec("Bedrooms", "imputable", "categorical", vocab=("1", "2"))])
    assert changed.schema_hash() != h and changed.vocab_hash() != v


def test_role_change_changes_schema_hash_not_vocab_hash():
    schema = tiny_schema(with_numeric=False)
    cols = list(schema.columns)
    cols[0] = ColumnSpec("State", "imputable", "categorical", vocab=cols[0].vocab)
    other = Schema(cols)
    assert other.schema_hash() != schema.schema_hash()
    assert other.vocab_hash() == schema.vocab_hash()


def test_invalid_schema_rejected():
    with pytest.raises(SchemaError):
        ColumnSpec("x", "imputable", "categorical", vocab=("a", "a"))
    with pytest.raises(SchemaError):
        ColumnSpec("x", "target", "categorical", vocab=("a",))
    with pytest.raises(SchemaError):
        Schema([ColumnSpec("x", "imputable", "categorical", vocab=("a",))], groups={"g": ["y"]})


def test_quantile_normal_monotone_with_ties():
    qn = QuantileNormal.fit(np.array([1, 1, 1, 2, 3, 3, 10], dtype=float), 50)
    x = np.linspace(0, 12, 100)
    z = qn.forward(x)
    assert np.all(np.diff(z) >= -1e-12)
    np.testing.assert_allclose(qn.inverse(qn.forward(np.array([2.0, 3.0, 10.0]))), [2.0, 3.0, 10.0], atol=0.2)
