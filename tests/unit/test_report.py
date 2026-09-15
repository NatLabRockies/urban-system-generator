import json

import pandas as pd
import pytest

from usg.report import reconcile_with_final, write_reports


def _records():
    return [{
        "building": "b1",
        "confidence": {
            "known_inputs": ["Vintage"],
            "context_inputs": ["State"],
            "n_known": 2,
            "method": "native",
            "model_id": "gnn-MD-2024.2-v1",
            "attributes": {
                "Heating Fuel": {"value": "Natural Gas", "type": "categorical", "confidence": 0.9, "band": "high"},
                "HVAC Has Ducts": {"value": "No", "type": "categorical", "confidence": 0.4, "band": "low"},
                "Geometry Garage": {"value": "None", "type": "categorical", "confidence": 0.7, "band": "medium"},
            },
            "summary": {"n_targets": 3, "mean_confidence": 0.667},
        },
    }]


def test_reconcile_labels_every_final_column():
    final = pd.DataFrame([{
        "Building": "b1", "Feature ID": "b1", "Vintage": "1990s", "State": "MD", "Heating Fuel": "Natural Gas",
        "HVAC Has Ducts": "Yes", "Geometry Garage": "None", "Windows": "Double, Clear, Non-metal, Air",
        "sqft_passthrough": "",
    }])
    out = reconcile_with_final(_records(), final, "Building", ["Building", "State", "Vintage", "Heating Fuel",
                                                                "HVAC Has Ducts", "Geometry Garage", "sqft_passthrough"])
    attrs = out[0]["confidence"]["attributes"]
    assert attrs["Vintage"]["source"] == "user_provided"
    assert attrs["State"]["source"] == "project_context" and attrs["State"]["band"] == "context"
    assert attrs["Heating Fuel"]["source"] == "model_inferred"
    assert attrs["HVAC Has Ducts"]["source"] == "model_inferred_adjusted"
    assert attrs["HVAC Has Ducts"]["model_value"] == "No" and attrs["HVAC Has Ducts"]["value"] == "Yes"
    assert attrs["Geometry Garage"]["source"] == "model_inferred" and attrs["Geometry Garage"]["value"] == "None"
    assert attrs["Windows"]["source"] == "postprocess_default"
    assert attrs["sqft_passthrough"]["band"] == "unknown"
    summary = out[0]["confidence"]["summary"]
    assert summary["n_project_context"] == 1 and summary["n_adjusted"] == 1 and summary["n_low"] == 1
    assert out[0]["confidence"]["model_id"] == "gnn-MD-2024.2-v1"


def test_write_reports_all_formats(tmp_path):
    records = _records()
    records[0]["building"] = "<b1>"
    written = write_reports(records, tmp_path / "r", formats=["json", "md", "html"], meta={"model": "<x>"})
    assert sorted(p.rsplit(".", 1)[1] for p in written) == ["html", "json", "md"]
    payload = json.loads((tmp_path / "r" / "confidence_report.json").read_text())
    assert payload["run_summary"]["cell_band_counts"]["high"] == 1
    assert "project_context" in payload["source_labels"]
    html = (tmp_path / "r" / "confidence_report.html").read_text()
    assert "&lt;b1&gt;" in html and "<b1>" not in html


def _v2_records():
    def attr(value, conf, band, expected, baseline, drivers=(), suggestions=()):
        return {"value": value, "type": "categorical", "confidence": conf, "band": band, "expected_accuracy": expected,
                "baseline_accuracy": baseline, "probability": conf, "drivers": list(drivers),
                "suggestions": list(suggestions)}

    return [{
        "building": "b1",
        "confidence": {
            "known_inputs": ["Vintage"], "context_inputs": ["State"], "n_known": 2, "method": "native",
            "model_id": "gnn-national-2024.2-v1",
            "attributes": {
                "Heating Fuel": attr("Natural Gas", 0.84, "high", 0.78, 0.6, [{"fact": "Vintage", "effect": 0.05}],
                                     [{"fact": "Geometry Floor Area", "gain": 0.02}]),
                "Doors": attr("Fiberglass", 1.0, "high", 1.0, 1.0),
                "Orientation": attr("North", 0.2, "low", 0.19, 0.15),
            },
            "summary": {"n_targets": 3, "mean_confidence": 0.68},
            "known_facts": ["Vintage"], "missing_facts": ["Geometry Floor Area", "Bedrooms"],
            "suggested_facts": [{"fact": "Geometry Floor Area", "mean_gain": 0.02, "n_attributes_improved": 1}],
        },
    }]


def test_v2_records_reconciled_and_rendered(tmp_path):
    final = pd.DataFrame([{"Building": "b1", "Vintage": "1990s", "State": "MD", "Heating Fuel": "Natural Gas",
                           "Doors": "Fiberglass", "Orientation": "North"}])
    records = reconcile_with_final(_v2_records(), final, "Building", list(final.columns))
    conf = records[0]["confidence"]
    assert conf["known_facts"] == ["Vintage"] and conf["suggested_facts"][0]["fact"] == "Geometry Floor Area"
    assert conf["summary"]["mean_expected_accuracy"] == pytest.approx(0.657)

    validation = {"n_validation_cells": 1000, "ece_confidence": 0.004, "ece_model_probability": 0.03}
    write_reports(records, tmp_path, formats=["json", "md", "html"], validation=validation)
    payload = json.loads((tmp_path / "confidence_report.json").read_text())
    summary = payload["run_summary"]
    assert payload["validation"] == validation
    assert summary["near_constant_attributes"] == ["Doors"] and "Doors" not in dict(summary["strongest_attributes"])
    assert summary["suggested_facts"] == ["Geometry Floor Area"]
    floor = next(f for f in summary["facts"] if f["fact"] == "Geometry Floor Area")
    assert floor["geojson_field"] == "floor_area"
    assert floor["n_buildings_known"] == 0 and floor["n_buildings_suggested"] == 1
    html = (tmp_path / "confidence_report.html").read_text()
    assert "floor_area" in html and "driven by Vintage" in html
    assert "Reliability" in html and "3.0 points for the raw model probability" in html
    md = (tmp_path / "confidence_report.md").read_text()
    assert "Most helpful facts to add: Geometry Floor Area (floor_area)" in md


def test_geojson_fields_match_the_geojson_mapper():
    import inspect

    from usg.geojson import GeoJSONProcessor
    from usg.report import GEOJSON_FIELDS

    source = inspect.getsource(GeoJSONProcessor.map_properties)
    for fact, field in GEOJSON_FIELDS.items():
        assert f'"{fact}"' in source and f'props.get("{field}")' in source, fact
