"""GeoJSON -> `usg workflow` with a tiny fixture bundle -> URBANopt-BuildStock mapping CSV."""

import json
import shutil

import pandas as pd
import pytest
from click.testing import CliRunner

from usg.cli import cli
from usg.geojson import geojson_to_dataframe
from usg.postprocessor import load_allowed_options, load_required_columns, validate_and_fix_schema
from usg.resources import options_lookup_path

ENV = {"USG_REGISTRY": "", "USG_OFFLINE": "1"}


@pytest.fixture
def project(tmp_path, helpers):
    geojson = tmp_path / "project" / "buildings.json"
    geojson.parent.mkdir()
    shutil.copy(helpers.example_geojson, geojson)
    return geojson


def test_workflow_produces_valid_buildstock_mapping(project, gnn_bundle, tmp_path):
    out = tmp_path / "run" / "uo_buildstock_mapping.csv"
    result = CliRunner().invoke(
        cli,
        ["workflow", "-i", str(project), "-o", str(out), "--model-dir", str(gnn_bundle), "--confidence-report"],
        env=ENV,
    )
    assert result.exit_code == 0, result.output
    assert "URBANopt-BuildStock compatible: Yes" in result.output

    final = pd.read_csv(out, dtype=str, keep_default_na=False, na_values=[""])
    assert len(final) == 77
    required = load_required_columns()
    assert set(required) <= set(final.columns)
    assert list(final.columns[:2]) == ["Feature ID", "Building"]
    _, issues = validate_and_fix_schema(final, load_allowed_options(options_lookup_path()), fix_invalid_values=False)
    assert issues == []
    assert (final["State"] == "MD").all() and (final["ASHRAE IECC Climate Zone 2004"] == "4A").all()
    assert final["City"].str.startswith("MD,").all()
    # GeoJSON inputs are preserved unless a consistency rule changed them, which the audit report must record
    mapped = geojson_to_dataframe(project)
    audit = pd.read_csv(out.parent / "consistency_audit_report.csv", dtype=str, keep_default_na=False)
    fixed = set(zip(audit.loc[audit["status"] == "FIXED", "row_index"].astype(int),
                    audit.loc[audit["status"] == "FIXED", "field"]))
    for column in ("Vintage", "Geometry Foundation Type", "Heating Fuel", "Geometry Building Type RECS"):
        changed = [i for i, (a, b) in enumerate(zip(final[column], mapped[column])) if a != b]
        assert all((i, column) in fixed for i in changed), (column, changed)
    assert final["Vintage"].tolist() == mapped["Vintage"].tolist()
    assert not list(out.parent.glob("*_intermediate.csv")) and not list(out.parent.glob("*_inference_output.csv"))

    manifest = json.loads((out.parent / "run_manifest.json").read_text())
    assert manifest["command"] == "workflow" and manifest["models"][0]["model_id"] == "gnn-MD-2024.2-v1"
    assert manifest["project_context"]["state"] == "MD" and manifest["n_buildings"] == 77
    assert manifest["selection"]["source"] == "bundle_dir"
    post = json.loads((out.parent / "postprocess_report.json").read_text())
    assert post["provenance"] and post["cells"]["model"]["total"] > 0
    assert manifest["postprocess"]["headline"] == post["headline"] and "Post-processing changed" in result.output
    assert (out.parent / "postprocess_changes.csv").exists() and (out.parent / "postprocess_report.html").exists()

    report = json.loads((out.parent / "confidence_report.json").read_text())
    attrs = report["buildings"][0]["confidence"]["attributes"]
    assert attrs["State"]["source"] == "project_context"
    assert attrs["Vintage"]["source"] == "user_provided"
    assert attrs["Heating Fuel"]["source"] == "user_provided"
    assert attrs["Geometry Garage"]["source"].startswith("model_inferred")
    assert attrs["Windows"]["source"] == "postprocess_default"
    assert set(attrs) == set(final.columns) - {"Building"}


def test_workflow_skip_postprocess_with_tabddpm(project, tabddpm_bundle, tmp_path):
    out = tmp_path / "inference.csv"
    result = CliRunner().invoke(
        cli,
        ["workflow", "-i", str(project), "-o", str(out), "-d", str(tabddpm_bundle), "--skip-postprocess",
         "--keep-intermediate", "--samples", "4", "--seed", "7"],
        env=ENV,
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out, dtype=str, keep_default_na=False, na_values=[""])
    assert "Feature ID" not in df.columns and df["Geometry Garage"].notna().all()
    assert (tmp_path / "buildings_intermediate.csv").exists()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["n_samples"] == 4 and manifest["seed"] == 7
