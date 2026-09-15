import json
from pathlib import Path

import pandas as pd
import pytest

from usg.geojson import ProjectContext
from usg.postprocess_report import build_postprocess_report, write_postprocess_report
from usg.postprocessor import (
    USGPostProcessor,
    eval_condition,
    fill_missing_columns,
    load_allowed_options,
    load_required_columns,
    read_csv_strings,
    run_consistency_engine,
    validate_and_fix_schema,
)
from usg.resources import consistency_rules_path, options_lookup_path


@pytest.fixture(scope="module")
def allowed():
    return load_allowed_options(options_lookup_path())


@pytest.fixture(scope="module")
def processor():
    return USGPostProcessor(verbose=False)


def test_required_columns_file(allowed):
    required = load_required_columns()
    assert len(required) == len(set(required)) == 161
    assert all(c in allowed for c in required)
    for legacy_only in ("Electric Vehicle", "Solar Hot Water", "site_energy.total.energy_consumption.kwh"):
        assert legacy_only not in required
    assert required[0] == "Bathroom Spot Vent Hour" and required[-1] == "Vacancy Status"
    assert {"State", "ASHRAE IECC Climate Zone 2004", "Windows"} <= set(required)


def test_fill_and_drop(allowed):
    df = pd.DataFrame({"Building": ["a", "b"], "Vintage": ["1990s", None], "sqft": [1.0, 2.0],
                       "Electric Vehicle": ["x", "y"], "Feature ID": ["?", "?"]})
    out = fill_missing_columns(df, allowed)
    assert list(out.columns[:3]) == ["Feature ID", "Building", "Vintage"]
    assert out["Feature ID"].tolist() == ["a", "b"]
    assert "sqft" not in out.columns and "Electric Vehicle" not in out.columns
    assert set(load_required_columns()) <= set(out.columns)
    assert out.at[1, "Vintage"] == "None"  # missing cells become literal None, validation fixes them later
    assert out.at[0, "Windows"] == "Double, Clear, Non-metal, Air"  # model column default
    assert out.at[0, "Has PV"] == "No"


def test_explicit_empty_lists_are_respected(allowed):
    df = pd.DataFrame({"Building": ["a"], "sqft": ["1"]})
    out = fill_missing_columns(df, allowed, missing_parameters=[], extra_cols_to_drop=[])
    assert list(out.columns) == ["Feature ID", "Building", "sqft"]


def test_state_and_climate_aware_defaults(allowed):
    out = fill_missing_columns(pd.DataFrame({"Building": ["a"]}), allowed, state="NY", climate_zone="6a")
    row = out.iloc[0]
    assert row["State"] == "NY" and row["City"].startswith("NY,") and row["County"].startswith("NY,")
    assert row["Census Division"] == "Middle Atlantic" and row["ISO RTO Region"] == "NYISO"
    assert row["ASHRAE IECC Climate Zone 2004"] == "6A"
    assert row["Custom State"] == "Others"

    per_row = fill_missing_columns(pd.DataFrame({"Building": ["a", "b"], "State": ["MD", "TX"]}), allowed)
    assert per_row["City"].str[:2].tolist() == ["MD", "TX"]
    assert per_row["Census Region"].tolist() == ["South", "South"]
    assert per_row["ISO RTO Region"].tolist() == ["PJM", "ERCOT"]


def test_none_option_survives_csv_roundtrip(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("Building,Geometry Garage,Vintage\na,None,\n")
    df = read_csv_strings(path)
    assert df.at[0, "Geometry Garage"] == "None" and pd.isna(df.at[0, "Vintage"])


def test_schema_validation_fixes(allowed):
    df = pd.DataFrame({"Feature ID": ["a", "b"], "Vintage": ["1890s", "None"], "Geometry Garage": ["none", "2 Car"]})
    fixed, issues = validate_and_fix_schema(df, allowed)
    kinds = {(i.column, i.issue_type) for i in issues}
    assert kinds == {("Vintage", "NOT_IN_ALLOWED_OPTIONS"), ("Vintage", "VOID_NOT_ALLOWED")}
    assert fixed["Vintage"].tolist() == ["1980s", "1980s"]
    _, remaining = validate_and_fix_schema(fixed, allowed, fix_invalid_values=False)
    assert remaining == []


def test_consistency_rules(allowed):
    rules = json.loads(consistency_rules_path().read_text())["rules"]
    df = pd.DataFrame({"Feature ID": ["a", "b"], "HVAC Heating Type": ["Ducted Heating", "Non-Ducted Heating"],
                       "HVAC Has Ducts": ["No", "No"]})
    out, audit = run_consistency_engine(df, allowed, rules, verbose=False)
    assert out["HVAC Has Ducts"].tolist()[0] == "Yes"
    assert any(e.rule_id == "ducted_heating_requires_hvac_has_ducts_yes" and e.status == "FIXED" for e in audit)
    row = pd.Series({"A": "x", "B": "y"})
    assert eval_condition({"all": [{"field": "A", "op": "equals", "value": "x"},
                                   {"any": [{"field": "B", "op": "in", "value": ["z", "y"]}]}]}, row)
    assert not eval_condition({"field": "B", "op": "starts_with_any", "value": ["q"]}, row)


def test_process_csv_with_context(processor, tmp_path):
    src = tmp_path / "completed.csv"
    pd.DataFrame({"Building": ["1", "2"], "Vintage": ["1990s", "bogus"], "HVAC Heating Type": ["Ducted Heating"] * 2,
                  "HVAC Has Ducts": ["No", "No"]}).to_csv(src, index=False)
    out = tmp_path / "out" / "final.csv"
    status = processor.process(src, out, context=ProjectContext(state="CO", climate_zone="5B"))
    assert "2 buildings" in status
    df = read_csv_strings(out)
    assert df["State"].tolist() == ["CO", "CO"] and df["ASHRAE IECC Climate Zone 2004"].tolist() == ["5B", "5B"]
    assert (out.parent / "schema_validation_report.csv").exists()
    assert (out.parent / "consistency_audit_report.csv").exists()
    _, issues = validate_and_fix_schema(df, processor.allowed_options, fix_invalid_values=False)
    assert issues == []
    summary = json.loads((out.parent / "postprocess_report.json").read_text())
    assert not summary["provenance"] and summary["cells"]["input"]["changed"] >= 3
    changes = pd.read_csv(out.parent / "postprocess_changes.csv", dtype=str, keep_default_na=False)
    assert {("2", "Vintage"), ("1", "HVAC Has Ducts"), ("2", "HVAC Has Ducts")} <= set(
        zip(changes["Building"], changes["column"])
    )
    assert (out.parent / "postprocess_report.html").exists()


def test_change_report_labels_sources_and_rules(processor, tmp_path):
    before = pd.DataFrame({"Building": ["1", "2"], "State": ["CO", "CO"], "Vintage": ["1990s", "bogus"],
                           "HVAC Heating Type": ["Ducted Heating", "Non-Ducted Heating"], "HVAC Has Ducts": ["No", "No"],
                           "Geometry Garage": [None, "None"]})
    imputed = pd.DataFrame({"State": [False] * 2, "Vintage": [False] * 2, "HVAC Heating Type": [True] * 2,
                            "HVAC Has Ducts": [True] * 2, "Geometry Garage": [False, True]})
    after, issues, events = processor.process_dataframe(before.copy(), ProjectContext(state="CO", climate_zone="5B"))
    report = build_postprocess_report(before, after, issues, events, processor.rules, imputed=imputed,
                                      context_columns=["State"], fill_log=processor.last_fill_log)
    added = {c["column"]: c for c in report.summary["columns_added"]}
    assert all(c["methods"] for c in added.values()) and report.summary["fill_methods"]
    assert added["County"]["methods"] == [{"method": "state default", "n": 2}]  # no derivation table: static only
    changes = {(r["row"], r["column"]): r for r in report.changes.to_dict("records")}
    ducts = changes[(0, "HVAC Has Ducts")]
    assert ducts["source"] == "model" and ducts["after"] == "Yes" and "consistency_rule" in ducts["step"]
    assert "ducted_heating_requires_hvac_has_ducts_yes" in ducts["rules"]
    vintage = changes[(1, "Vintage")]
    assert vintage["source"] == "input" and vintage["step"].startswith("schema_validation")
    assert (0, "Geometry Garage") not in changes

    s = report.summary
    assert s["provenance"] and s["verdict"] == "review"  # one of the user's values was replaced
    assert s["cells"]["project"] == {"total": 2, "changed": 0, "share": 0.0}
    assert s["n_filled_empty"] >= 1 and s["columns_added"]
    assert "Feature ID" not in {c["column"] for c in s["columns_added"]}
    assert any(r["rule_id"] == "ducted_heating_requires_hvac_has_ducts_yes" and r["changes"] >= 1 for r in s["by_rule"])
    assert "ducted_heating_requires_hvac_has_ducts_yes" not in {v["rule_id"] for v in s["remaining_violations"]}

    paths = write_postprocess_report(report, tmp_path, formats=["json", "html", "md"])
    assert sorted(Path(p).name for p in paths) == [
        "postprocess_changes.csv", "postprocess_report.html", "postprocess_report.json"
    ]
    assert "HVAC Has Ducts" in (tmp_path / "postprocess_report.html").read_text()


def test_change_report_lists_remaining_violations():
    rules = [{"id": "garage_needs_detached", "severity": "warning", "note": "garages only on detached homes",
              "when": {"field": "Geometry Garage", "op": "equals", "value": "1 Car"},
              "then": [{"field": "Geometry Building Type RECS", "op": "equals", "value": "Single-Family Detached"}]}]
    frame = pd.DataFrame({"Building": ["a", "b"], "Geometry Garage": ["1 Car", "None"],
                          "Geometry Building Type RECS": ["Mobile Home", "Mobile Home"]})
    report = build_postprocess_report(frame, frame.copy(), rules=rules)
    s = report.summary
    assert s["remaining_violations"] == [{"rule_id": "garage_needs_detached", "severity": "warning",
                                          "note": "garages only on detached homes", "buildings": 1, "examples": ["a"]}]
    assert s["n_changed"] == 0 and s["verdict"] == "ok" and not s["provenance"]
    assert "1 rule violation(s) remain" in report.headline
