import json

import pandas as pd
import pytest
from click.testing import CliRunner

from usg.__about__ import __version__
from usg.cli import cli
from usg.testing.fixtures import tiny_dataframe


@pytest.fixture
def run(tmp_path):
    env = {"USG_CACHE_DIR": str(tmp_path / "cache"), "USG_REGISTRY": "", "USG_OFFLINE": ""}

    def invoke(*args, **kwargs):
        return CliRunner().invoke(cli, [str(a) for a in args], env={**env, **kwargs.pop("env", {})}, **kwargs)

    return invoke


def _csv(tmp_path, n=6, replace=None):
    df = tiny_dataframe(n, seed=8).drop(columns=["Heating Fuel", "sqft"])
    df.insert(0, "Building", [f"b{i}" for i in range(n)])
    for (row, col), value in (replace or {}).items():
        df.loc[row, col] = value
    path = tmp_path / "in.csv"
    df.to_csv(path, index=False)
    return path


def test_help_and_version(run):
    result = run("--help")
    assert result.exit_code == 0
    for command in ("geojson2csv", "predict-single", "complete", "confidence", "process", "workflow", "info", "models"):
        assert command in result.output
    assert __version__ in run("--version").output
    models_help = run("models", "--help").output
    for sub in ("list", "download", "info", "verify", "clear-cache", "path"):
        assert sub in models_help
    assert "--unknown-labels" in run("complete", "--help").output


def test_geojson2csv(run, helpers, gnn_bundle, tmp_path):
    out = tmp_path / "b.csv"
    result = run("geojson2csv", "-i", helpers.example_geojson, "-o", out, "-v")
    assert result.exit_code == 0, result.output
    assert "77 buildings" in result.output and "MD" in result.output
    df = pd.read_csv(out, dtype=str)
    assert {"Building", "Vintage", "State", "ASHRAE IECC Climate Zone 2004"} <= set(df.columns)
    with_schema = tmp_path / "s.csv"
    assert run("geojson2csv", "-i", helpers.example_geojson, "-o", with_schema, "-d", gnn_bundle).exit_code == 0
    assert "Geometry Garage" in pd.read_csv(with_schema).columns


def test_predict_single(run, gnn_bundle, tmp_path):
    result = run("predict-single", "-a", "Vintage", "1990s", "-a", "State", "MD", "--model-dir", gnn_bundle)
    assert result.exit_code == 0, result.output
    assert "gnn-MD-2024.2-v1" in result.output and "[✓] Vintage: 1990s" in result.output
    assert "conf " in result.output and "Mean confidence" in result.output

    known = tmp_path / "known.json"
    known.write_text(json.dumps({"Bedrooms": "3"}))
    out = tmp_path / "pred.json"
    result = run("predict-single", "-j", known, "-a", "Vintage", "2000s", "-o", out, "--top-k", "2", "-d", gnn_bundle)
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["Vintage"] == "2000s" and payload["Bedrooms"] == "3" and "Heating Fuel" in payload
    assert len(payload["_meta"]["top_k"]["Heating Fuel"]) == 2
    assert "Vintage" not in payload["_meta"]["imputed"]
    assert payload["_meta"]["confidence"]["attributes"]["Heating Fuel"]["confidence"] is not None
    assert (tmp_path / "pred_confidence_report.json").exists() and (tmp_path / "pred_confidence_report.html").exists()


def test_complete_with_confidence_and_issues(run, gnn_bundle, tmp_path):
    src = _csv(tmp_path, replace={(2, "Vintage"): "1890s"})
    out = tmp_path / "out" / "completed.csv"
    result = run("complete", "-i", src, "-o", out, "--model-dir", gnn_bundle, "--confidence-report",
                 "--confidence-method", "ensemble", "--passes", "4", "--report-format", "json,md")
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out, dtype=str, keep_default_na=False, na_values=[""])
    assert df["Heating Fuel"].notna().all() and df["Vintage"].notna().all()
    assert (out.parent / "confidence_report.json").exists() and (out.parent / "confidence_report.md").exists()
    issues = pd.read_csv(out.parent / "completed_input_issues.csv")
    assert issues[["row", "Building", "column", "value"]].values.tolist() == [[2, "b2", "Vintage", "1890s"]]
    manifest = json.loads((out.parent / "run_manifest.json").read_text())
    assert manifest["models"][0]["model_id"] == "gnn-MD-2024.2-v1" and manifest["seed"] == 0
    assert manifest["versions"]["torch"]

    strict = run("complete", "-i", src, "-o", out, "-d", gnn_bundle, "--unknown-labels", "error")
    assert strict.exit_code == 5 and "not valid options" in strict.output


def test_confidence_reports_are_written_by_default(run, gnn_bundle, tmp_path):
    out = tmp_path / "default" / "completed.csv"
    result = run("complete", "-i", _csv(tmp_path), "-o", out, "-d", gnn_bundle)
    assert result.exit_code == 0, result.output
    assert (out.parent / "confidence_report.json").exists() and (out.parent / "confidence_report.html").exists()
    assert "Confidence: mean" in result.output
    manifest = json.loads((out.parent / "run_manifest.json").read_text())
    assert manifest["confidence"]["summary"]["n_buildings"] == 6

    quiet = tmp_path / "quiet" / "completed.csv"
    result = run("complete", "-i", _csv(tmp_path), "-o", quiet, "-d", gnn_bundle, "--no-confidence-report")
    assert result.exit_code == 0, result.output
    assert not (quiet.parent / "confidence_report.json").exists()
    assert json.loads((quiet.parent / "run_manifest.json").read_text())["confidence"] is None


def test_confidence_command(run, gnn_bundle, tmp_path):
    sens = tmp_path / "sens.json"
    sens.write_text(json.dumps({"Heating Fuel": {"Vintage": 1.0}}))
    out = tmp_path / "c.csv"
    result = run("confidence", "-i", _csv(tmp_path), "-o", out, "-d", gnn_bundle, "--sensitivity", sens,
                 "--exclude", "sqft", "--report-dir", tmp_path / "rep")
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "rep" / "confidence_report.json").read_text())
    attrs = report["buildings"][0]["confidence"]["attributes"]
    assert attrs["Heating Fuel"]["coverage"] == 1.0 and "sqft" not in attrs


def test_process(run, helpers, tmp_path):
    raw = tmp_path / "raw.csv"
    assert run("geojson2csv", "-i", helpers.example_geojson, "-o", raw).exit_code == 0
    out = tmp_path / "final.csv"
    result = run("process", "-i", raw, "-o", out, "-g", helpers.example_geojson, "--reports-dir", tmp_path / "r", "-v")
    assert result.exit_code == 0, result.output
    assert "Total buildings: 77" in result.output
    assert pd.read_csv(out, dtype=str)["City"].str.startswith("MD,").all()


@pytest.mark.parametrize(
    "args, exit_code, message",
    [
        (["workflow", "-i", "{geojson}", "-o", "{tmp}/x.csv", "--registry", "{empty}"], 3, "no gnn model"),
        (["workflow", "-i", "{geojson}", "-o", "{tmp}/x.csv", "--state", "ZZ"], 5, "invalid state"),
        (["complete", "-i", "{csv}", "-o", "{tmp}/x.csv", "-d", "{broken}"], 4, "missing bundle file"),
        (["complete", "-i", "{csv}", "-o", "{tmp}/x.csv", "-d", "{bundle}", "--id-col", "Nope"], 5, "ID column"),
        (["complete", "-i", "{csv}", "-o", "{tmp}/x.csv", "--registry", "{tmp}/missing.json"], 3, "registry"),
        (["complete", "-i", "{csv}", "-o", "{tmp}/x.csv", "-d", "{bundle}", "--confidence-report",
          "--report-format", "pdf"], 2, "report-format"),
    ],
)
def test_error_exit_codes(run, helpers, gnn_bundle, tmp_path, args, exit_code, message):
    broken = tmp_path / "broken"
    broken.mkdir()
    empty = tmp_path / "empty_registry.json"
    empty.write_text(json.dumps({"registry_format": "usg-registry/v1", "models": [], "champions": {},
                                 "default_model_type": {"__default__": "gnn"}}))
    values = {"geojson": helpers.example_geojson, "tmp": tmp_path, "csv": _csv(tmp_path), "broken": broken,
              "bundle": gnn_bundle, "empty": empty}
    result = run(*[a.format(**values) for a in args])
    assert result.exit_code == exit_code, result.output
    assert message in result.output


def test_info(run, gnn_bundle, tmp_path):
    export = tmp_path / "attrs.json"
    result = run("info", "-d", gnn_bundle, "--export", export)
    assert result.exit_code == 0, result.output
    assert "gnn-MD-2024.2-v1" in result.output and "required_columns.json (161 columns)" in result.output
    assert "Heating Fuel" in json.loads(export.read_text())["imputable_attributes"]
