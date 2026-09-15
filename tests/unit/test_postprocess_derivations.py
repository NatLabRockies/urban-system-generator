import pandas as pd
import pytest

from usg.postprocess_derivations import (
    WEATHER_STATION,
    apply_derivations,
    describe_level,
    load_derivations,
)
from usg.postprocessor import fill_missing_columns, load_allowed_options
from usg.resources import options_lookup_path


def _table():
    return {
        "format": "usg-postprocess-derivations/v1",
        "derive_purity": 0.99,
        "key_separator": "\t",
        "order": ["County and PUMA", "County", "Geometry Story Bin"],
        "columns": {
            "County and PUMA": {"kind": "anchor", "mode": "G2400050, G24000100", "levels": [
                {"from": ["State", WEATHER_STATION], "purity": 0.3,
                 "values": {"NY\tBuffalo Niagara Intl Ap": "G3600290, G36001204"}},
                {"from": ["State"], "purity": 0.05,
                 "values": {"NY": "G3600610, G36003805", "MD": "G2400050, G24000100"}},
            ]},
            "County": {"kind": "derived", "mode": "MD, Baltimore County", "levels": [
                {"from": ["County and PUMA"], "purity": 1.0,
                 "values": {"G3600290, G36001204": "NY, Erie County", "G3600610, G36003805": "NY, New York County",
                            "G2400050, G24000100": "MD, Baltimore County"}},
            ]},
            "Geometry Story Bin": {"kind": "derived", "mode": "<8", "levels": [
                {"from": ["Geometry Stories"], "purity": 1.0, "values": {"2": "<8", "9": "8+"}},
            ]},
        },
    }


def test_location_columns_follow_one_anchor_and_fall_back_to_the_state():
    df = pd.DataFrame({"Building": ["a", "b"], "State": ["NY", "NY"], "Geometry Stories": ["2", "9"]})
    log: dict = {}
    out = apply_derivations(df, _table(), context={WEATHER_STATION: "Buffalo Niagara Intl Ap"}, log=log)
    assert out["County and PUMA"].tolist() == ["G3600290, G36001204"] * 2
    assert out["County"].tolist() == ["NY, Erie County"] * 2
    assert out["Geometry Story Bin"].tolist() == ["<8", "8+"]
    assert log["County and PUMA"] == {"most common for State + Weather Station": 2}
    assert log["County"] == {"derived from County and PUMA": 2}
    assert "County" not in df.columns  # the input frame is not modified

    no_station: dict = {}
    out = apply_derivations(df, _table(), log=no_station)
    assert out["County"].tolist() == ["NY, New York County"] * 2
    assert no_station["County and PUMA"] == {"most common for State": 2}


def test_existing_values_are_kept_and_invalid_options_skipped():
    df = pd.DataFrame({"Building": ["a", "b"], "State": ["MD", None], "County": ["MD, Howard County", None],
                       "Geometry Stories": ["2", "2"]})
    allowed = {"County": ["MD, Howard County", "MD, Baltimore County"], "County and PUMA": ["G2400050, G24000100"],
               "Geometry Story Bin": ["8+"]}
    log: dict = {}
    out = apply_derivations(df, _table(), allowed=allowed, log=log)
    assert out.at[0, "County"] == "MD, Howard County"
    assert out["County and PUMA"].tolist() == ["G2400050, G24000100"] * 2
    assert out.at[1, "County"] == "MD, Baltimore County"
    assert "Geometry Story Bin" not in out.columns  # nothing valid to fill ("<8" rejected): left to static defaults
    assert log["County and PUMA"] == {"most common for State": 1, "most common nationally": 1}


def test_project_context_fills_its_own_columns():
    df = pd.DataFrame({"Building": ["a"], "County": [None]})
    table = _table()
    table["order"] = ["County"]
    log: dict = {}
    out = apply_derivations(df, table, context={"County": "NY, Erie County"}, log=log)
    assert out.at[0, "County"] == "NY, Erie County" and log["County"] == {"project location": 1}


def test_format_and_labels():
    with pytest.raises(ValueError, match="usg-postprocess-derivations/v1"):
        load_derivations({"format": "other"})
    assert describe_level({"from": ["A", "B"], "purity": 0.5}, 0.99) == "most common for A + B"
    assert describe_level({"from": ["A"], "purity": 1.0}, 0.99) == "derived from A"


def test_fill_missing_columns_derives_first_then_prefers_neutral_static_defaults():
    allowed = load_allowed_options(options_lookup_path())
    log: dict = {}
    out = fill_missing_columns(pd.DataFrame({"Building": ["a"], "Geometry Stories": ["2"]}), allowed, state="MD",
                               derivations=_table(), fill_log=log)
    assert out.at[0, "Geometry Story Bin"] == "<8" and log["Geometry Story Bin"] == {"derived from Geometry Stories": 1}
    assert out.at[0, "HVAC Cooling Autosizing Factor"] == "None"  # not "40% Oversized"
    assert log["HVAC Cooling Autosizing Factor"] == {"static default": 1}
