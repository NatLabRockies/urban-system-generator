"""Location options and location-aware inputs across `usg` commands."""

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from usg.cli import cli
from usg.testing.fixtures import tiny_dataframe

ZONE = "ASHRAE IECC Climate Zone 2004"
BA = "Building America Climate Zone"
BALTIMORE = ("39.29", "-76.61")
RICHMOND = ("37.54", "-77.43")
EXAMPLE_LOCATION = "Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state)"


@pytest.fixture
def run(tmp_path):
    env = {"USG_CACHE_DIR": str(tmp_path / "cache"), "USG_REGISTRY": "", "USG_OFFLINE": "1"}

    def invoke(*args):
        return CliRunner().invoke(cli, [str(a) for a in args], env=env)

    return invoke


@pytest.fixture
def origin_only_geojson(helpers, tmp_path):
    """The example project without any weather filename: only its Site Origin point locates it."""
    data = json.loads(helpers.example_geojson.read_text(encoding="utf-8"))
    data["project"] = {}
    for feature in data["features"]:
        if feature["properties"].get("type") == "Site Origin":
            feature["properties"].pop("weather_filename", None)
    path = tmp_path / "project" / "origin.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_location_options_are_listed(run):
    for command in ("workflow", "complete", "confidence", "predict-single", "geojson2csv", "process"):
        text = run(command, "--help").output
        assert all(o in text for o in ("--climate-zone", "--latitude", "--longitude", "--state")), command


def test_geojson2csv_options_override_the_project(run, helpers, tmp_path):
    out = tmp_path / "b.csv"
    plain = run("geojson2csv", "-i", helpers.example_geojson, "-o", out)
    assert plain.exit_code == 0, plain.output
    assert EXAMPLE_LOCATION in plain.output

    result = run("geojson2csv", "-i", helpers.example_geojson, "-o", out, "--state", "va", "--climate-zone", "5a")
    assert result.exit_code == 0, result.output
    assert "Location: VA (explicit) · 5A (explicit) · Cold (from 5A) · South Atlantic (from state)" in result.output
    df = pd.read_csv(out, dtype=str)
    assert (df["State"] == "VA").all() and (df[ZONE] == "5A").all() and (df[BA] == "Cold").all()


@pytest.mark.parametrize(
    "args, message",
    [
        (["--climate-zone", "9Z"], "not an ASHRAE IECC 2004 climate zone"),
        (["--latitude", "39"], "--latitude and --longitude must be given together"),
        (["--state", "ZZ"], "not a two-letter US state code"),
        (["--latitude", "95", "--longitude", "0"], "--latitude"),
    ],
)
def test_invalid_location_options(run, helpers, tmp_path, args, message):
    result = run("geojson2csv", "-i", helpers.example_geojson, "-o", tmp_path / "x.csv", *args)
    assert result.exit_code == 2 and message in result.output


def test_workflow_locates_a_project_from_its_site_origin(run, origin_only_geojson, gnn_bundle, tmp_path):
    out = tmp_path / "run" / "mapping.csv"
    result = run("workflow", "-i", origin_only_geojson, "-o", out, "-d", gnn_bundle)
    assert result.exit_code == 0, result.output
    assert ("Location: MD (nearest station) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state)"
            in result.output)
    final = pd.read_csv(out, dtype=str, keep_default_na=False, na_values=[""])
    assert (final["State"] == "MD").all() and (final[ZONE] == "4A").all() and (final[BA] == "Mixed-Humid").all()
    manifest = json.loads((out.parent / "run_manifest.json").read_text())
    location = manifest["location"]
    assert location["coordinates_source"] == "Site Origin" and location["values"]["State"] == "MD"
    assert location["summary"].startswith("Location: MD (nearest station)") and location["nearest_station"]
    assert manifest["project_context"]["state"] == "MD"


def test_workflow_explicit_coordinates_override_the_site_origin(run, origin_only_geojson, gnn_bundle, tmp_path):
    out = tmp_path / "run" / "inference.csv"
    result = run("workflow", "-i", origin_only_geojson, "-o", out, "-d", gnn_bundle, "--skip-postprocess",
                 "--keep-intermediate", "--latitude", RICHMOND[0], "--longitude", RICHMOND[1])
    assert result.exit_code == 0, result.output
    assert "Location: VA (nearest station)" in result.output
    model_inputs = pd.read_csv(out.parent / "origin_intermediate.csv", dtype=str)
    assert (model_inputs["State"] == "VA").all()  # State is the tiny test schema's only location column
    assert json.loads((out.parent / "run_manifest.json").read_text())["location"]["coordinates_source"] == "explicit"


def test_complete_locates_rows_from_latitude_longitude_columns(run, gnn_bundle, tmp_path):
    df = tiny_dataframe(4, seed=8).drop(columns=["State", "Heating Fuel", "sqft"])
    df.insert(0, "Building", ["balt", "richmond", "unknown", "kept"])
    df["Latitude"] = [BALTIMORE[0], RICHMOND[0], "", BALTIMORE[0]]
    df["Longitude"] = [BALTIMORE[1], RICHMOND[1], "", BALTIMORE[1]]
    df[ZONE] = ["", "", "", "5A"]
    src = tmp_path / "in.csv"
    df.to_csv(src, index=False)
    out = tmp_path / "out" / "done.csv"

    result = run("complete", "-i", src, "-o", out, "-d", gnn_bundle)
    assert result.exit_code == 0, result.output
    assert "Location: derived per row from Latitude/Longitude" in result.output
    assert "keep their input value" in result.output  # row "kept": its own 5A differs from Baltimore's 4A
    done = pd.read_csv(out, dtype=str, keep_default_na=False, na_values=[""]).fillna("")
    assert done["State"].tolist() == ["MD", "VA", "", "MD"]
    assert done[ZONE].tolist() == ["4A", "4A", "", "5A"]
    assert done[BA].tolist() == ["Mixed-Humid", "Mixed-Humid", "", "Cold"]
    assert "Latitude" not in done.columns
    manifest = json.loads((out.parent / "run_manifest.json").read_text())
    assert manifest["location"]["mode"] == "coordinates" and manifest["location"]["filled"]["State"] == 3
    assert not (out.parent / "done_input_issues.csv").exists()


def test_complete_explicit_options_apply_to_every_row_without_overwriting(run, gnn_bundle, tmp_path):
    df = tiny_dataframe(5, seed=8).drop(columns=["Heating Fuel", "sqft"])
    df.insert(0, "Building", [f"b{i}" for i in range(5)])
    df["State"] = [None, "MD", "VA", "DC", "MD"]
    src = tmp_path / "in.csv"
    df.to_csv(src, index=False)
    out = tmp_path / "done.csv"

    result = run("complete", "-i", src, "-o", out, "-d", gnn_bundle, "--state", "VA", "--climate-zone", "5A")
    assert result.exit_code == 0, result.output
    done = pd.read_csv(out, dtype=str, keep_default_na=False, na_values=[""])
    assert done["State"].tolist() == ["VA", "MD", "VA", "DC", "MD"]
    assert (done[ZONE] == "5A").all() and (done[BA] == "Cold").all()
    assert "State: 3 row(s) keep their input value" in result.output


def test_predict_single_location_options(run, gnn_bundle, tmp_path):
    out = tmp_path / "pred.json"
    result = run("predict-single", "-a", "Vintage", "1990s", "--latitude", BALTIMORE[0], "--longitude", BALTIMORE[1],
                 "-d", gnn_bundle, "-o", out)
    assert result.exit_code == 0, result.output
    assert "Location: MD (nearest station)" in result.output
    payload = json.loads(out.read_text())
    assert payload["State"] == "MD" and "State" not in payload["_meta"]["imputed"]

    kept = run("predict-single", "-a", "State", "VA", "--latitude", BALTIMORE[0], "--longitude", BALTIMORE[1],
               "-d", gnn_bundle)
    assert kept.exit_code == 0, kept.output
    assert "[✓] State: VA" in kept.output and "keep their input value" in kept.output


def test_process_location_options(run, helpers, tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"Building": ["1"], "Vintage": ["1990s"]}).to_csv(raw, index=False)
    out = tmp_path / "final.csv"

    result = run("process", "-i", raw, "-o", out, "--state", "CO", "--climate-zone", "5B", "--no-reports")
    assert result.exit_code == 0, result.output
    assert "Location: CO (explicit) · 5B (explicit) · Cold (from 5B) · Mountain (from state)" in result.output
    final = pd.read_csv(out, dtype=str)
    assert (final.loc[0, "State"], final.loc[0, ZONE], final.loc[0, BA]) == ("CO", "5B", "Cold")
    assert final.loc[0, "City"].startswith("CO,")

    result = run("process", "-i", raw, "-o", out, "-g", helpers.example_geojson, "--climate-zone", "5A", "--no-reports")
    assert result.exit_code == 0, result.output
    final = pd.read_csv(out, dtype=str)
    assert (final.loc[0, "State"], final.loc[0, ZONE], final.loc[0, BA]) == ("MD", "5A", "Cold")


def test_process_defaults_follow_the_resolved_location_not_the_project_climate_zone(run, helpers, tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"Building": ["1"], "Vintage": ["1990s"]}).to_csv(raw, index=False)
    out = tmp_path / "final.csv"
    # the Buffalo project says climate_zone 6A; its Site Origin's weather station homes are 5A
    result = run("process", "-i", raw, "-o", out, "-g", helpers.combined_geojson, "--no-reports")
    assert result.exit_code == 0, result.output
    assert "Location: NY (weather file) · 5A (nearest station) · Cold (from 5A)" in result.output
    final = pd.read_csv(out, dtype=str)
    assert (final.loc[0, "State"], final.loc[0, ZONE], final.loc[0, BA]) == ("NY", "5A", "Cold")
