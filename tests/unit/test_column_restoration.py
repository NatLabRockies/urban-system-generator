"""Restoring the columns the reduced model is not trained to predict.

The reduced target set is only safe if every column it drops can be put back exactly. These tests pin the
properties that make it safe: the caller's own data wins, an exact lookup is never a guess, and anything
the table cannot resolve is reported rather than left silently blank.
"""

from __future__ import annotations

import pandas as pd
import pytest

from usg.column_restoration import load_table, restorable_columns, restore

# a row using the stock's own vocabulary, carrying every source the rules need
ROW = {
    "Vintage": "1940s",
    "Geometry Building Type RECS": "Single-Family Detached",
    "Geometry Building Type ACS": "Single-Family Detached",
    "Geometry Foundation Type": "Slab",
    "Geometry Attic Type": "Vented Attic",
    "Geometry Garage": "1 Car",
    "Geometry Building Level MF": "None",
    "Geometry Stories": "2",
    "Geometry Floor Area": "1500-1999",
    "Usage Level": "Medium",
    "Clothes Washer": "Standard",
    "Misc Pool": "None",
    "Insulation Foundation Wall": "Uninsulated",
    "Duct Location": "Living Space",
    "Water Heater Location": "Living Space",
    "HVAC Shared Efficiencies": "None",
    "HVAC Heating Efficiency": "Fuel Furnace, 80% AFUE",
    "Heating Setpoint Offset Magnitude": "0F",
    "Cooling Setpoint Offset Magnitude": "0F",
}


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame([ROW])


def test_the_table_ships_with_the_package():
    table = load_table()
    assert table["constants"] and table["rules"], "the restoration table is missing from the wheel"
    assert len(restorable_columns()) == len(table["constants"]) + len(table["rules"])


def test_every_rule_resolves_when_its_sources_are_present(frame):
    out, report = restore(frame)
    assert report.unmapped == [], [u.column for u in report.unmapped]
    assert report.missing_sources == []
    assert report.complete
    assert set(restorable_columns()) <= set(out.columns)


def test_a_composite_is_assembled_from_its_parts(frame):
    """Geometry Space Combination is the clearest case: five predictions, one required output."""
    out, _ = restore(frame)
    assert out.at[0, "Geometry Space Combination"] == "Single-Family Detached, Slab, Vented Attic, 1 Car Garage"


def test_a_coarsening_follows_its_finer_column(frame):
    out, _ = restore(frame)
    assert out.at[0, "Vintage ACS"] == "1940-59"


def test_constants_are_filled(frame):
    out, _ = restore(frame)
    assert out.at[0, "Dehumidifier"] == "None"


def test_the_callers_own_value_is_never_overwritten(frame):
    """A supplied value wins and the disagreement is reported, as everywhere else in usg."""
    frame["Vintage ACS"] = "1980-99"          # contradicts Vintage = 1940s
    out, report = restore(frame)
    assert out.at[0, "Vintage ACS"] == "1980-99"
    conflict = next(c for c in report.conflicts if c.column == "Vintage ACS")
    assert conflict.given == "1980-99" and conflict.stock == "1940-59"


def test_a_combination_absent_from_the_stock_is_reported_not_guessed(frame):
    """A wrong value here reaches the energy model as if it were real, so silence is not acceptable."""
    # the stock only ever gives a garage to Single-Family Detached, so this pair cannot occur
    frame["Geometry Building Type RECS"] = "Multi-Family with 5+ Units"
    frame["Geometry Garage"] = "1 Car"
    out, report = restore(frame)
    unmapped = [u for u in report.unmapped if u.column == "Geometry Space Combination"]
    assert unmapped, "an unmapped combination must be reported"
    assert pd.isna(out.at[0, "Geometry Space Combination"])
    assert not report.complete


def test_missing_source_columns_are_reported(frame):
    _out, report = restore(frame.drop(columns=["Duct Location"]))
    assert "Duct Location" in report.missing_sources
    assert not report.complete


def test_restoration_runs_before_the_generic_fill(frame):
    """Step order matters: an exact lookup must not be pre-empted by a default."""
    from usg.postprocessor import USGPostProcessor

    processor = USGPostProcessor(verbose=False)
    frame.insert(0, "Building", ["b0"])       # the post-processor keys rows by this column
    out, _issues, _events = processor.process_dataframe(frame.copy())
    assert out.at[0, "Vintage ACS"] == "1940-59"
    methods = processor.last_fill_log.get("Vintage ACS", {})
    assert any("restored from Vintage" in m for m in methods), methods
