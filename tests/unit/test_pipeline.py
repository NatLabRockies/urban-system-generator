import json

import pandas as pd
import pytest

from usg.download import Cache
from usg.errors import InputValidationError
from usg.pipeline import ConfidenceOptions, complete_dataframe, run_complete
from usg.registry import Registry
from usg.selection import ModelRequest, ResolvedModel
from usg.testing.fixtures import tiny_dataframe


def _input(n=12, seed=4):
    df = tiny_dataframe(n, seed=seed).drop(columns=["Heating Fuel", "Geometry Garage"]).astype(object)
    df.insert(0, "Building", [f"b{i}" for i in range(n)])
    return df


def test_complete_with_imputer(gnn_imputer):
    df = _input()
    df["Vintage ACS"] = "1980-99"  # required ResStock parameter outside the schema: passed through
    df["Comment"] = "x"  # not a parameter: reported, dropped
    df.loc[3, "Vintage"] = "1890s"
    out = complete_dataframe(df, gnn_imputer, confidence=ConfidenceOptions(method="both", n_passes=4))
    schema = gnn_imputer.schema
    assert list(out.data.columns) == ["Building", *schema.names, "Vintage ACS"]
    assert out.data["Building"].tolist() == df["Building"].tolist()
    assert out.data["Heating Fuel"].notna().all() and out.data["Vintage"].notna().all()
    assert out.imputed["Heating Fuel"].all() and not out.imputed["State"].any() and bool(out.imputed.at[3, "Vintage"])
    kinds = {(i.row, i.column, i.kind) for i in out.issues}
    assert kinds == {(3, "Vintage", "unknown_label"), (-1, "Comment", "unknown_column")}
    assert len(out.records) == 12 and out.records[0]["building"] == "b0"
    assert out.models[0]["model_id"] == "gnn-MD-2024.2-v1" and out.models[0]["n_rows"] == 12


def test_conditioning_column_only_written_when_known(gnn_imputer):
    df = _input().drop(columns=["State"])
    assert "State" not in complete_dataframe(df, gnn_imputer).data.columns


def test_partitioned_completion_keeps_row_order(bundle_factory):
    md, va = bundle_factory("gnn", "MD"), bundle_factory("gnn", "VA")
    df = _input(20)
    df["State"] = ["MD", "VA"] * 10
    df.loc[0, "State"] = None
    mapping = {
        "MD": ResolvedModel(md, md.name, "MD", "test"),
        "VA": ResolvedModel(va, va.name, "VA", "test"),
        None: ResolvedModel(md, md.name, "MD", "test"),
    }
    out = complete_dataframe(df, mapping, confidence=ConfidenceOptions())
    assert out.data["Building"].tolist() == df["Building"].tolist()
    assert {m["model_id"]: m["n_rows"] for m in out.models} == {md.name: 10, va.name: 10}
    assert [r["confidence"]["model_id"] for r in out.records[:2]] == [md.name, va.name]

    with pytest.raises(InputValidationError, match="no resolved model"):
        complete_dataframe(df, {"MD": mapping["MD"]})
    with pytest.raises(InputValidationError, match="ID column"):
        complete_dataframe(df.drop(columns=["Building"]), mapping)


def test_run_complete_partitions_by_state(bundle_factory, helpers, tmp_path):
    cache = Cache(tmp_path / "cache")
    bundles = [helpers.copy(bundle_factory("gnn", r), cache.dir) for r in ("MD", "VA", "national")]
    registry = Registry(helpers.registry(
        [helpers.entry(b) for b in bundles],
        {"MD": {"gnn": bundles[0].name}, "VA": {"gnn": bundles[1].name}, "national": {"gnn": bundles[2].name}},
    ))
    src = tmp_path / "in.csv"
    df = _input(9)
    df["State"] = ["MD", "VA", "DC"] * 3
    df.to_csv(src, index=False)
    result = run_complete(src, tmp_path / "out" / "done.csv", ModelRequest(), registry=registry, cache=cache,
                          embedded_dir=tmp_path / "none")
    assert set(result.resolved) == {"MD", "VA", "DC"}
    manifest = json.loads(result.manifest_path.read_text())
    assert sorted(m["model_id"] for m in manifest["models"]) == sorted(b.name for b in bundles)
    assert any("DC" in w for w in manifest["warnings"])
    done = pd.read_csv(result.output_csv, dtype=str, keep_default_na=False, na_values=[""])
    assert done["State"].tolist() == df["State"].tolist() and done["Heating Fuel"].notna().all()
