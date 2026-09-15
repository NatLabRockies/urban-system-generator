"""Post-processing of inference output into a validated URBANopt-BuildStock mapping CSV.

Three sequential steps (ported from urban-system-generator):

1. Missing column filler — adds every column in ``required_columns.json`` that is absent (state/climate-aware
   defaults) and drops columns that URBANopt-BuildStock does not use.
2. Schema validator — validates and fixes values against ``options_lookup.tsv``.
3. Consistency processor — enforces cross-field rules from ``consistency_rules.json``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from usg.geojson import (  # noqa: F401 - re-exported for backwards compatibility
    ProjectContext,
    read_project_climate_zone,
    read_project_context,
    read_project_state,
)
from usg.resources import consistency_rules_path, options_lookup_path, required_columns_path

ID_COLUMN = "Building"
FEATURE_ID_COLUMN = "Feature ID"
CLIMATE_ZONE_PARAMETERS = {"ASHRAE IECC Climate Zone 2004", "ASHRAE IECC Climate Zone 2004 - Sub-CZ Split"}

# ============================================================================
# LOGGING HELPERS
# ============================================================================


def _now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{_now()}] {msg}", flush=True)


class Timer:
    def __init__(self, label: str, verbose: bool = True):
        self.label = label
        self.verbose = verbose
        self.t0 = time.perf_counter()

    def done(self, extra: str = "") -> None:
        dt = time.perf_counter() - self.t0
        suffix = f" {extra}" if extra else ""
        log(f"{self.label} ✓ ({dt:.3f}s){suffix}", self.verbose)


# ============================================================================
# OPTIONS LOOKUP HELPERS (shared across all steps)
# ============================================================================


def load_allowed_options(options_lookup_tsv: str | Path) -> dict[str, list[str]]:
    """Read options_lookup.tsv into ``{Parameter Name: [Option Name, ...]}`` (only the first two columns)."""
    allowed: dict[str, list[str]] = {}
    with Path(options_lookup_tsv).open("r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            param = parts[0].strip()
            opt = parts[1].strip().strip('"')
            if not param or not opt:
                continue
            allowed.setdefault(param, []).append(opt)
    return {k: list(dict.fromkeys(v)) for k, v in allowed.items()}


def coerce_to_allowed(value: str | None, allowed: list[str]) -> str | None:
    """Return a value guaranteed to be in allowed, else None (case-insensitive match supported)."""
    if value is None:
        return None
    s = str(value).strip().strip('"')
    if not s:
        return None
    if s in allowed:
        return s
    low = s.lower()
    for a in allowed:
        if a.lower() == low:
            return a
    return None


def build_allowed_sets(allowed_options: dict[str, list[str]]) -> dict[str, tuple[set, dict[str, str]]]:
    """For each param: the exact allowed set and a case-insensitive ``lower(opt) -> opt`` map."""
    out: dict[str, tuple[set, dict[str, str]]] = {}
    for p, opts in allowed_options.items():
        ci_map: dict[str, str] = {}
        for o in opts:
            ci_map.setdefault(o.lower(), o)
        out[p] = (set(opts), ci_map)
    return out


def load_required_columns(path: str | Path | None = None) -> list[str]:
    """Ordered parameter columns an URBANopt-BuildStock mapping CSV must contain (``required_columns.json``)."""
    if path is None:
        return list(_default_required_columns())
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(obj["columns"] if isinstance(obj, dict) else obj)


@lru_cache(maxsize=1)
def _default_required_columns() -> tuple[str, ...]:
    return tuple(load_required_columns(required_columns_path()))


def read_csv_strings(path: str | Path) -> pd.DataFrame:
    """Read a CSV as strings; only empty cells are missing (the option ``"None"`` is kept as text)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


# ============================================================================
# STEP 1: MISSING COLUMN FILLER
# ============================================================================

# Default values for missing parameters
DEFAULT_FILL_VALUES: dict[str, str] = {
    # Location-ish / admin:
    "AHS Region": "",
    "AIANNH Area": "",
    "Area Median Income": "",
    "Census Division": "",
    "Census Division RECS": "",
    "Census Region": "",
    "City": "",
    "County": "",
    "County Metro Status": "",
    "County and PUMA": "",
    "Custom State": "",
    "Federal Poverty Level": "",
    "Generation And Emissions Assessment Region": "",
    "ISO RTO Region": "",
    "Income": "",
    "Income RECS2015": "",
    "Income RECS2020": "",
    "Location Region": "",
    "Metropolitan and Micropolitan Statistical Area": "",
    "PUMA": "",
    "PUMA Metro Status": "",
    "REEDS Balancing Area": "",
    "State": "",
    "State Metro Median Income": "",
    "Tenure": "",
    "Vacancy Status": "",
    # Climate zones:
    "ASHRAE IECC Climate Zone 2004": "",
    "ASHRAE IECC Climate Zone 2004 - Sub-CZ Split": "",
    "Building America Climate Zone": "",
    "CEC Climate Zone": "",
    "Energystar Climate Zone 2023": "",
    # HVAC / energy / equipment:
    "Battery": "None",
    "Has PV": "No",
    "PV Orientation": "None",
    "PV System Size": "None",
    "HVAC Cooling Autosizing Factor": "",
    "HVAC Heating Autosizing Factor": "",
    "HVAC Secondary Heating Type": "None",
    "HVAC System Is Scaled": "",
    # Availability days:
    "Cooling Unavailable Days": "Never",
    "Heating Unavailable Days": "Never",
    # EV-related:
    "Electric Vehicle Battery": "None",
    "Electric Vehicle Charge At Home": "None",
    "Electric Vehicle Charger": "None",
    "Electric Vehicle Miles Traveled": "None",
    "Electric Vehicle Outlet Access": "None",
    "Electric Vehicle Ownership": "None",
    # Misc:
    "Geometry Story Bin": "",
    "Radiant Barrier": "None",
}


# State-to-region mappings for location-based fields
STATE_TO_CENSUS_DIVISION = {
    # New England
    "CT": "New England", "ME": "New England", "MA": "New England",
    "NH": "New England", "RI": "New England", "VT": "New England",
    # Middle Atlantic
    "NJ": "Middle Atlantic", "NY": "Middle Atlantic", "PA": "Middle Atlantic",
    # East North Central
    "IL": "East North Central", "IN": "East North Central", "MI": "East North Central",
    "OH": "East North Central", "WI": "East North Central",
    # West North Central
    "IA": "West North Central", "KS": "West North Central", "MN": "West North Central",
    "MO": "West North Central", "NE": "West North Central", "ND": "West North Central",
    "SD": "West North Central",
    # South Atlantic
    "DE": "South Atlantic", "FL": "South Atlantic", "GA": "South Atlantic",
    "MD": "South Atlantic", "NC": "South Atlantic", "SC": "South Atlantic",
    "VA": "South Atlantic", "WV": "South Atlantic", "DC": "South Atlantic",
    # East South Central
    "AL": "East South Central", "KY": "East South Central", "MS": "East South Central",
    "TN": "East South Central",
    # West South Central
    "AR": "West South Central", "LA": "West South Central", "OK": "West South Central",
    "TX": "West South Central",
    # Mountain
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain",
    "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",
    # Pacific
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific", "OR": "Pacific", "WA": "Pacific",
}  # fmt: skip

STATE_TO_CENSUS_REGION = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast", "RI": "Northeast",
    "VT": "Northeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest", "KS": "Midwest",
    "MI": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",
    # South
    "AL": "South", "AR": "South", "DC": "South", "DE": "South", "FL": "South",
    "GA": "South", "KY": "South", "LA": "South", "MD": "South", "MS": "South",
    "NC": "South", "OK": "South", "SC": "South", "TN": "South", "TX": "South",
    "VA": "South", "WV": "South",
    # West
    "AK": "West", "AZ": "West", "CA": "West", "CO": "West", "HI": "West",
    "ID": "West", "MT": "West", "NM": "West", "NV": "West", "OR": "West",
    "UT": "West", "WA": "West", "WY": "West",
}  # fmt: skip

STATE_TO_ISO_RTO = {
    # CAISO - California
    "CA": "CAISO",
    # ERCOT - Texas
    "TX": "ERCOT",
    # NYISO - New York
    "NY": "NYISO",
    # NEISO - New England
    "CT": "NEISO", "ME": "NEISO", "MA": "NEISO", "NH": "NEISO", "RI": "NEISO", "VT": "NEISO",
    # PJM - Mid-Atlantic and some Midwest
    "DC": "PJM", "DE": "PJM", "IL": "PJM", "IN": "PJM", "KY": "PJM", "MD": "PJM",
    "MI": "PJM", "NC": "PJM", "NJ": "PJM", "OH": "PJM", "PA": "PJM", "TN": "PJM",
    "VA": "PJM", "WV": "PJM",
    # MISO - Midwest
    "AR": "MISO", "IA": "MISO", "LA": "MISO", "MN": "MISO", "MO": "MISO",
    "MS": "MISO", "MT": "MISO", "ND": "MISO", "SD": "MISO", "WI": "MISO",
    # SPP - South Central
    "KS": "SPP", "NE": "SPP", "NM": "SPP", "OK": "SPP",
    # SERC - Southeast (not in RTO)
    "AL": "SERC", "FL": "SERC", "GA": "SERC", "SC": "SERC",
    # WECC - Western (not in RTO)
    "AK": "None", "AZ": "WECC", "CO": "WECC", "HI": "None", "ID": "WECC",
    "NV": "WECC", "OR": "WECC", "UT": "WECC", "WA": "WECC", "WY": "WECC",
}  # fmt: skip

# Location-related parameters that should be state-aware
LOCATION_PARAMETERS = [
    "State", "Custom State", "City", "County", "PUMA",
    "Census Division", "Census Division RECS", "Census Region",
    "ISO RTO Region",
]  # fmt: skip


def get_state_aware_location_defaults(state: str, allowed_options: dict[str, list[str]]) -> dict[str, str]:
    """Location-related default values that match ``state`` (e.g. ``'NY'``)."""
    defaults: dict[str, str] = {}
    if not state:
        return defaults
    state = state.upper()

    if "State" in allowed_options and state in allowed_options["State"]:
        defaults["State"] = state

    if "Custom State" in allowed_options:
        if state in allowed_options["Custom State"]:
            defaults["Custom State"] = state
        elif "Others" in allowed_options["Custom State"]:
            defaults["Custom State"] = "Others"

    # City / County / PUMA options are prefixed by the state ("NY, Buffalo", "NY, Erie County", "NY, 00100")
    for param in ("City", "County", "PUMA"):
        if param in allowed_options:
            matches = [c for c in allowed_options[param] if c.startswith((f'"{state},', f"{state},"))]
            if matches:
                defaults[param] = matches[0].strip('"')

    # Metropolitan and Micropolitan Statistical Area ("Albany-Schenectady-Troy, NY MSA")
    msa = "Metropolitan and Micropolitan Statistical Area"
    if msa in allowed_options:
        state_metros = [m for m in allowed_options[msa] if f", {state} " in m or f", {state}-" in m]
        if state_metros:
            defaults[msa] = state_metros[0].strip('"')

    census_div = STATE_TO_CENSUS_DIVISION.get(state)
    if census_div:
        if "Census Division" in allowed_options and census_div in allowed_options["Census Division"]:
            defaults["Census Division"] = census_div
        if "Census Division RECS" in allowed_options and census_div in allowed_options["Census Division RECS"]:
            defaults["Census Division RECS"] = census_div

    census_reg = STATE_TO_CENSUS_REGION.get(state)
    if census_reg and "Census Region" in allowed_options and census_reg in allowed_options["Census Region"]:
        defaults["Census Region"] = census_reg

    iso_rto = STATE_TO_ISO_RTO.get(state)
    if iso_rto and "ISO RTO Region" in allowed_options and iso_rto in allowed_options["ISO RTO Region"]:
        defaults["ISO RTO Region"] = iso_rto

    return defaults


def force_none_string_everywhere(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OUTPUT CSV contains literal text 'None' for any missing value."""
    df = df.astype(object)
    df = df.where(pd.notna(df), None)
    df = df.replace(to_replace=r"^\s*(nan|NaN|NAN|<NA>|None|null|NULL)\s*$", value=None, regex=True)
    return df.where(pd.notna(df), "None")


# When a static default is empty or not a valid option, prefer an option meaning "absent" over the alphabetically
# first one (which gave e.g. "40% Oversized" HVAC autosizing).
NEUTRAL_OPTIONS = ("None", "Never", "No", "Not Available", "Void")


def _choose_fill_value(
    param: str, allowed: list[str], location_defaults: dict[str, str], climate_zone: str | None
) -> str:
    """Priority: state-aware location default > climate zone > DEFAULT_FILL_VALUES > DEFAULT_VALUE_OVERRIDES."""
    if param in location_defaults:
        chosen = location_defaults[param]
    elif param in DEFAULT_FILL_VALUES:
        chosen = DEFAULT_FILL_VALUES[param]
    else:  # a model output column the model did not produce
        chosen = DEFAULT_VALUE_OVERRIDES.get(param, "")
    if climate_zone and param in CLIMATE_ZONE_PARAMETERS and allowed:
        chosen = coerce_to_allowed(climate_zone, allowed) or chosen
    neutral = next((o for o in NEUTRAL_OPTIONS if o in allowed), allowed[0] if allowed else None)
    if (chosen is None or str(chosen).strip() == "") and allowed:
        chosen = neutral
    if allowed:
        chosen = coerce_to_allowed(chosen, allowed) or neutral
    return str(chosen) if chosen else "None"


def _consistent_static(param: str, df: pd.DataFrame, allowed: list[str], default: Any) -> tuple[Any, str] | None:
    """Static fills that follow location columns already present (e.g. derived ``City``, ``PUMA Metro Status``).

    Used for parameters a ResStock release has no data for, so they cannot be derived from the release itself.
    """
    base = default if isinstance(default, pd.Series) else pd.Series([default] * len(df), index=df.index, dtype=object)
    if param == "County Metro Status" and "PUMA Metro Status" in df.columns and \
            {"Metropolitan", "Non-Metropolitan"} <= set(allowed):
        metro = df["PUMA Metro Status"].astype(str)
        known = metro.str.startswith(("In metro area", "Not/partially in metro area"))
        mapped = metro.map(lambda s: "Non-Metropolitan" if s.startswith("Not/partially") else "Metropolitan")
        return base.where(~known, mapped), "static default matching PUMA Metro Status"
    if param == "Metropolitan and Micropolitan Statistical Area" and "City" in df.columns:
        cache: dict[str, str | None] = {}

        def metro_area(city: str) -> str | None:
            # "NY, Cheektowaga" -> "Buffalo-Cheektowaga-Niagara Falls, NY MSA": the city is one of its principal cities
            if city not in cache:
                state_code, _, name = city.partition(", ")
                cache[city] = next(
                    (o for o in allowed if name and "," in o and name in o.split(",", 1)[0].split("-")
                     and state_code in o.split(",", 1)[1].split()[0].split("-")),
                    None,
                )
            return cache[city]

        matched = df["City"].astype(str).map(metro_area)
        return matched.where(matched.notna(), base), "static default matching City"
    return None


def fill_missing_columns(
    df: pd.DataFrame,
    allowed_options: dict[str, list[str]],
    climate_zone: str | None = None,
    state: str | None = None,
    missing_parameters: list[str] | None = None,
    extra_cols_to_drop: list[str] | None = None,
    write_literal_none: bool = True,
    required_columns: list[str] | None = None,
    derivations: dict | None = None,
    context_values: dict[str, str] | None = None,
    fill_log: dict | None = None,
) -> pd.DataFrame:
    """Add missing required columns with defaults and drop columns URBANopt-BuildStock does not use.

    By default ``missing_parameters`` = required − present and ``extra_cols_to_drop`` = present − required −
    {Building, Feature ID}, with ``required_columns`` from ``required_columns.json``. Without ``state``, location
    defaults follow each row's ``State`` value when that column is present.
    """
    if ID_COLUMN not in df.columns:
        raise ValueError("Input CSV must contain a 'Building' column.")
    if missing_parameters is None or extra_cols_to_drop is None:
        required = required_columns if required_columns is not None else load_required_columns()
        if missing_parameters is None:
            missing_parameters = [c for c in required if c not in df.columns]
        if extra_cols_to_drop is None:
            keep = set(required) | {ID_COLUMN, FEATURE_ID_COLUMN}
            extra_cols_to_drop = [c for c in df.columns if c not in keep]

    df = df.copy()
    # Ensure 'Feature ID' first column (copy of 'Building')
    if FEATURE_ID_COLUMN in df.columns:
        df[FEATURE_ID_COLUMN] = df[ID_COLUMN]
        df = df[[FEATURE_ID_COLUMN] + [c for c in df.columns if c != FEATURE_ID_COLUMN]]
    else:
        df.insert(0, FEATURE_ID_COLUMN, df[ID_COLUMN])

    from usg.postprocess_derivations import apply_derivations, present

    partial: set[str] = set()
    if derivations:
        # ResStock-derived fills first (consistent with the known and predicted columns); static defaults below only
        # handle what the table could not fill. Existing values are never overwritten.
        df = apply_derivations(df, derivations, context=context_values, allowed=allowed_options, log=fill_log)
        partial = {c for c in missing_parameters if c in df.columns and not df[c].map(present).all()}

    state_defaults = get_state_aware_location_defaults(state, allowed_options) if state else {}
    row_states = None
    per_state: dict[str, dict[str, str]] = {}
    if not state and "State" in df.columns:
        row_states = df["State"].map(lambda v: "" if pd.isna(v) else str(v).strip().upper())
        per_state = {s: get_state_aware_location_defaults(s, allowed_options) for s in row_states.unique() if s}

    new_columns: dict[str, Any] = {}
    for param in missing_parameters:
        if param in df.columns and param not in partial:
            continue
        allowed = allowed_options.get(param, [])
        if row_states is not None and any(param in d for d in per_state.values()):
            choices = {s: _choose_fill_value(param, allowed, d, climate_zone) for s, d in per_state.items()}
            fallback = _choose_fill_value(param, allowed, {}, climate_zone)
            value = row_states.map(lambda s, c=choices, f=fallback: c.get(s, f))
        else:
            value = _choose_fill_value(param, allowed, state_defaults, climate_zone)
        by_state = param in state_defaults or any(param in d for d in per_state.values())
        label = ("state default" if by_state else
                 "project climate zone" if climate_zone and param in CLIMATE_ZONE_PARAMETERS else "static default")
        consistent = _consistent_static(param, df, allowed, value)
        if consistent is not None:
            value, label = consistent
        if param in partial:  # the derivation table filled some rows only
            empty = ~df[param].map(present)
            df.loc[empty, param] = value[empty] if isinstance(value, pd.Series) else value
            n_filled = int(empty.sum())
        else:
            new_columns[param] = value
            n_filled = len(df)
        if fill_log is not None:
            fill_log.setdefault(param, {})[label] = fill_log.get(param, {}).get(label, 0) + n_filled
    if new_columns:
        df = df.assign(**{k: v for k, v in new_columns.items()})

    df = df.drop(columns=[c for c in extra_cols_to_drop if c in df.columns])

    if write_literal_none:
        df = force_none_string_everywhere(df)
    return df


# ============================================================================
# STEP 2: SCHEMA VALIDATOR
# ============================================================================

DEFAULT_VOID_LIKE = {"", " ", "none", "null", "nan", "<na>", "n/a", "na", "void", "unknown"}

# Per-column override defaults used when fixing invalid values
DEFAULT_VALUE_OVERRIDES: dict[str, str] = {
    "AHS Region": "CBSA Atlanta-Sandy Springs-Roswell, GA",
    "AIANNH Area": "No",
    "ASHRAE IECC Climate Zone 2004": "4A",
    "ASHRAE IECC Climate Zone 2004 - Sub-CZ Split": "4A",
    "Area Median Income": "0-30%",
    "Bathroom Spot Vent Hour": "Hour0",
    "Battery": "None",
    "Bedrooms": "3",
    "Building America Climate Zone": "Mixed-Humid",
    "CEC Climate Zone": "1",
    "Ceiling Fan": "None",
    "Census Division": "East North Central",
    "Census Division RECS": "East North Central",
    "Census Region": "Midwest",
    "City": "AK, Anchorage",
    "Clothes Dryer": "Electric",
    "Clothes Dryer Usage Level": "100% Usage",
    "Clothes Washer": "Standard",
    "Clothes Washer Presence": "Yes",
    "Clothes Washer Usage Level": "100% Usage",
    "Cooking Range": "Electric Resistance",
    "Cooking Range Usage Level": "100% Usage",
    "Cooling Setpoint": "72F",
    "Cooling Setpoint Has Offset": "No",
    "Cooling Setpoint Offset Magnitude": "0F",
    "Cooling Setpoint Offset Period": "Day Setup",
    "Cooling Unavailable Days": "Never",
    "Corridor": "Not Applicable",
    "County": "AK, Aleutians East Borough",
    "County Metro Status": "Metropolitan",
    "County and PUMA": "G0100010, G01002100",
    "Custom State": "AK",
    "Dehumidifier": "None",
    "Dishwasher": "318 Rated kWh",
    "Dishwasher Usage Level": "100% Usage",
    "Door Area": "20 ft^2",
    "Doors": "Wood",
    "Duct Leakage and Insulation": "None",
    "Duct Location": "None",
    "Eaves": "2 ft",
    "Electric Vehicle Battery": "None",
    "Electric Vehicle Charge At Home": "None",
    "Electric Vehicle Charger": "None",
    "Electric Vehicle Miles Traveled": "None",
    "Electric Vehicle Outlet Access": "None",
    "Electric Vehicle Ownership": "None",
    "Energystar Climate Zone 2023": "North-Central",
    "Federal Poverty Level": "0-100%",
    "Generation And Emissions Assessment Region": "CAISO",
    "Geometry Attic Type": "Vented Attic",
    "Geometry Building Horizontal Location MF": "None",
    "Geometry Building Horizontal Location SFA": "None",
    "Geometry Building Level MF": "None",
    "Geometry Building Number Units MF": "None",
    "Geometry Building Number Units SFA": "None",
    "Geometry Building Type ACS": "Single-Family Detached",
    "Geometry Building Type Height": "Single-Family Detached",
    "Geometry Building Type RECS": "Single-Family Detached",
    "Geometry Floor Area": "1500-1999",
    "Geometry Floor Area Bin": "1500-2499",
    "Geometry Foundation Type": "Slab",
    "Geometry Garage": "None",
    "Geometry Space Combination": "Single-Family Detached, Slab, Vented Attic, No Garage",
    "Geometry Stories": "1",
    "Geometry Stories Low Rise": "1",
    "Geometry Story Bin": "1",
    "Geometry Wall Exterior Finish": "Vinyl, Light",
    "Geometry Wall Type": "Wood Frame",
    "Ground Thermal Conductivity": "1.1",
    "HVAC Cooling Autosizing Factor": "1.0",
    "HVAC Cooling Efficiency": "AC, SEER 13",
    "HVAC Cooling Partial Space Conditioning": "100% Conditioned",
    "HVAC Cooling Type": "Central AC",
    "HVAC Has Ducts": "Yes",
    "HVAC Has Shared System": "None",
    "HVAC Has Zonal Electric Heating": "No",
    "HVAC Heating Autosizing Factor": "1.0",
    "HVAC Heating Efficiency": "Fuel Furnace, 80% AFUE",
    "HVAC Heating Type": "Ducted Heating",
    "HVAC Heating Type And Fuel": "Natural Gas Fuel Furnace",
    "HVAC Secondary Heating Efficiency": "None",
    "HVAC Secondary Heating Fuel": "None",
    "HVAC Secondary Heating Partial Space Conditioning": "None",
    "HVAC Secondary Heating Type": "None",
    "HVAC Shared Efficiencies": "None",
    "HVAC System Is Faulted": "No",
    "HVAC System Is Scaled": "No",
    "HVAC System Single Speed AC Airflow": "None",
    "HVAC System Single Speed AC Charge": "None",
    "HVAC System Single Speed ASHP Airflow": "None",
    "HVAC System Single Speed ASHP Charge": "None",
    "Has PV": "No",
    "Heating Fuel": "Natural Gas",
    "Heating Setpoint": "68F",
    "Heating Setpoint Has Offset": "No",
    "Heating Setpoint Offset Magnitude": "0F",
    "Heating Setpoint Offset Period": "Day",
    "Heating Unavailable Days": "Never",
    "Holiday Lighting": "No Exterior Use",
    "Hot Water Distribution": "Uninsulated",
    "Hot Water Fixtures": "100% Usage",
    "Household Has Tribal Persons": "No",
    "ISO RTO Region": "None",
    "Income": "60000-69999",
    "Income RECS2015": "60000-79999",
    "Income RECS2020": "60000-99999",
    "Infiltration": "10 ACH50",
    "Insulation Ceiling": "R-30",
    "Insulation Floor": "None",
    "Insulation Foundation Wall": "None",
    "Insulation Rim Joist": "None",
    "Insulation Roof": "None",
    "Insulation Slab": "None",
    "Insulation Wall": "Wood Stud, R-13",
    "Interior Shading": "Summer = 0.7, Winter = 0.85",
    "Lighting": "100% LED",
    "Lighting Interior Use": "100% Usage",
    "Lighting Other Use": "100% Usage",
    "Location Region": "CR04",
    "Mechanical Ventilation": "None",
    "Metropolitan and Micropolitan Statistical Area": "None",
    "Misc Extra Refrigerator": "None",
    "Misc Freezer": "None",
    "Misc Gas Fireplace": "None",
    "Misc Gas Grill": "None",
    "Misc Gas Lighting": "None",
    "Misc Hot Tub Spa": "None",
    "Misc Pool": "None",
    "Misc Pool Heater": "None",
    "Misc Pool Pump": "None",
    "Misc Well Pump": "None",
    "Natural Ventilation": "Cooling Season, 7 days/wk",
    "Neighbors": "None",
    "Occupants": "3",
    "Orientation": "South",
    "Overhangs": "None",
    "PUMA": "AK, 00101",
    "PUMA Metro Status": "In metro area, not/partially in principal city",
    "PV Orientation": "None",
    "PV System Size": "None",
    "Plug Load Diversity": "100%",
    "Plug Loads": "100%",
    "REEDS Balancing Area": "1",
    "Radiant Barrier": "None",
    "Range Spot Vent Hour": "Hour17",
    "Refrigerator": "EF 17.6",
    "Roof Material": "Asphalt Shingles, Medium",
    "State": "MD",
    "State Metro Median Income": "80000-99999",
    "Tenure": "Owner",
    "Usage Level": "Medium",
    "Vacancy Status": "Occupied",
    "Vintage": "1980s",
    "Vintage ACS": "1980-99",
    "Water Heater Efficiency": "Natural Gas Standard",
    "Water Heater Fuel": "Natural Gas",
    "Water Heater In Unit": "Yes",
    "Water Heater Location": "Living Space",
    "Window Areas": "F15 B15 L15 R15",
    "Windows": "Double, Clear, Non-metal, Air",
}


@dataclass
class ValidationIssue:
    row_index: int
    feature_id: str | None
    column: str
    value: str
    issue_type: str
    message: str
    allowed_options: str
    fixed_to_default_value: str


def norm_str(x) -> str:
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return ""
    return str(x).strip()


def is_void_like(s: str, void_like: set) -> bool:
    return s.lower() in void_like


def pick_default_value(col: str, allowed: list[str], overrides: dict[str, str]) -> str | None:
    if col in overrides:
        v = overrides[col]
        if v in allowed:
            return v
        low = v.lower()
        for a in allowed:
            if a.lower() == low:
                return a
    if allowed:
        return allowed[0]
    return None


def validate_and_fix_schema(
    df: pd.DataFrame,
    allowed_options: dict[str, list[str]],
    feature_id_col: str = FEATURE_ID_COLUMN,
    void_like: set | None = None,
    fix_invalid_values: bool = True,
    default_value_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Validate values against ``allowed_options`` and optionally replace invalid ones with defaults."""
    void_like = void_like if void_like is not None else DEFAULT_VOID_LIKE
    default_value_overrides = default_value_overrides if default_value_overrides is not None else DEFAULT_VALUE_OVERRIDES

    allowed_sets = build_allowed_sets(allowed_options)
    issues: list[ValidationIssue] = []
    fixed_df = df.copy(deep=True).astype(object)

    schema_cols = [c for c in df.columns if c in allowed_options]
    feature_ids = df[feature_id_col] if feature_id_col in df.columns else pd.Series([None] * len(df), index=df.index)

    for col in schema_cols:
        exact_allowed, ci_map = allowed_sets[col]
        allowed_list = allowed_options.get(col, [])

        void_allowed = any(opt.strip().lower() in {"none", "void", "null"} for opt in allowed_list)
        default_fix = pick_default_value(col, allowed_list, default_value_overrides)

        for idx, raw in df[col].items():
            fid = feature_ids.loc[idx] if idx in feature_ids.index else None
            s = norm_str(raw)

            if is_void_like(s, void_like):
                if not void_allowed:
                    fixed_to = ""
                    if fix_invalid_values and default_fix:
                        fixed_df.at[idx, col] = default_fix
                        fixed_to = default_fix
                    issues.append(
                        ValidationIssue(
                            row_index=int(idx),
                            feature_id=str(fid) if fid is not None else None,
                            column=col,
                            value=s,
                            issue_type="VOID_NOT_ALLOWED",
                            message=f"Value is empty/None-like but '{col}' does not allow Void/None options.",
                            allowed_options=" | ".join(allowed_list[:50]),
                            fixed_to_default_value=fixed_to,
                        )
                    )
                continue

            canonical = None
            if s in exact_allowed:
                canonical = s
            elif s.lower() in ci_map:
                canonical = ci_map[s.lower()]

            if canonical is None:
                fixed_to = ""
                if fix_invalid_values and default_fix:
                    fixed_df.at[idx, col] = default_fix
                    fixed_to = default_fix
                issues.append(
                    ValidationIssue(
                        row_index=int(idx),
                        feature_id=str(fid) if fid is not None else None,
                        column=col,
                        value=s,
                        issue_type="NOT_IN_ALLOWED_OPTIONS",
                        message=f"Invalid option for '{col}'.",
                        allowed_options=" | ".join(allowed_list[:50]),
                        fixed_to_default_value=fixed_to,
                    )
                )

    return fixed_df, issues


# ============================================================================
# STEP 3: CONSISTENCY PROCESSOR
# ============================================================================


@dataclass
class AuditEvent:
    pass_num: int
    rule_id: str
    feature_id: str | None
    row_index: int
    severity: str
    status: str  # "FIXED" | "VIOLATION" | "SKIPPED"
    field: str | None
    old_value: str | None
    new_value: str | None
    message: str


def _get(row: pd.Series, field: str) -> str:
    return norm_str(row.get(field, ""))


def _starts_with_any(s: str, prefixes: list[str]) -> bool:
    s2 = (s or "").strip()
    return any(s2.startswith(p) for p in prefixes)


def eval_condition(cond: dict[str, Any], row: pd.Series) -> bool:
    """Evaluate a WHEN condition against a row."""
    if "all" in cond:
        return all(eval_condition(c, row) for c in cond["all"])
    if "any" in cond:
        return any(eval_condition(c, row) for c in cond["any"])

    field = cond["field"]
    op = cond["op"]
    val = cond.get("value")
    s = _get(row, field)

    if op == "equals":
        return s == str(val)
    if op == "in":
        return s in [str(x) for x in val]
    if op == "not_in":
        return s not in [str(x) for x in val]
    if op == "starts_with":
        return s.startswith(str(val))
    if op == "starts_with_any":
        return _starts_with_any(s, [str(x) for x in val])
    if op == "not_starts_with_any":
        return not _starts_with_any(s, [str(x) for x in val])
    if op == "starts_with_field":
        other = _get(row, str(val))
        return s.startswith(other) if other else False
    if op == "equals_field_mapped":
        return s in {str(v) for v in val.values()}

    raise ValueError(f"Unsupported condition op: {op}")


def eval_then_clause(clause: dict[str, Any], row: pd.Series) -> bool:
    """Evaluate THEN clauses (constraints)."""
    field = clause["field"]
    op = clause["op"]
    val = clause.get("value")
    s = _get(row, field)

    if op == "equals":
        return s == str(val)
    if op == "not_equals":
        return s != str(val)
    if op == "in":
        return s in [str(x) for x in val]
    if op == "not_in":
        return s not in [str(x) for x in val]
    if op == "starts_with_any":
        return _starts_with_any(s, [str(x) for x in val])
    if op == "not_starts_with_any":
        return not _starts_with_any(s, [str(x) for x in val])
    if op == "starts_with_field":
        other = _get(row, str(val))
        return s.startswith(other) if other else False
    if op == "equals_field_mapped":
        return s in {str(v) for v in (val or {}).values()}

    raise ValueError(f"Unsupported THEN op: {op}")


def set_value_if_allowed(
    df: pd.DataFrame,
    row_idx: int,
    field: str,
    new_value: str,
    allowed_sets: dict[str, tuple[set, dict[str, str]]],
) -> tuple[bool, str]:
    """Sets df.at[row_idx, field] to new_value if it's allowed."""
    if field not in allowed_sets:
        old = df.at[row_idx, field] if field in df.columns else ""
        df.at[row_idx, field] = new_value
        return (norm_str(old) != str(new_value).strip()), "Field not in options_lookup; set anyway."

    exact_allowed, ci_map = allowed_sets[field]
    canonical = None
    if new_value in exact_allowed:
        canonical = new_value
    elif new_value.lower() in ci_map:
        canonical = ci_map[new_value.lower()]

    if canonical is None:
        return False, f"Proposed fix value {new_value!r} is not in allowed options for '{field}'."

    old = df.at[row_idx, field] if field in df.columns else ""
    changed = norm_str(old) != str(canonical).strip()
    df.at[row_idx, field] = canonical
    return changed, f"Set to '{canonical}'."


def _eval_fix_condition(cond: dict[str, Any], row: pd.Series) -> bool:
    """Condition for fix actions (set_if): equals, not_equals, in, not_in."""
    field = cond.get("field", "")
    op = cond.get("op", "")
    val = cond.get("value")
    s = _get(row, field)

    if op == "equals":
        return s == str(val)
    if op == "not_equals":
        return s != str(val)
    if op == "in":
        return s in [str(x) for x in val]
    if op == "not_in":
        return s not in [str(x) for x in val]
    return False


def apply_fix_actions(
    df: pd.DataFrame,
    row_idx: int,
    row: pd.Series,
    rule: dict[str, Any],
    allowed_sets: dict[str, tuple[set, dict[str, str]]],
) -> list[tuple[str, str, str, str]]:
    """Apply fix actions from a rule. Returns list of (field, old, new, note)."""
    changes = []

    def attempt(field: str, new_val: str) -> None:
        old_val = _get(row, field)
        changed, note = set_value_if_allowed(df, row_idx, field, new_val, allowed_sets)
        if changed:
            changes.append((field, old_val, str(df.at[row_idx, field]), note))

    for fx in rule.get("fix", []) or []:
        action = fx.get("action")
        if action == "set":
            attempt(fx["field"], str(fx["value"]))
        elif action == "set_if":
            # {"action": "set_if", "when": {...condition...}, "field": "X", "value": "Y"}
            when_cond = fx.get("when", {})
            if when_cond and _eval_fix_condition(when_cond, row):
                attempt(fx["field"], str(fx["value"]))
        elif action == "set_if_in":
            # {"action": "set_if_in", "field": "X", "when_value_in": [...], "value": "Y"}
            if _get(row, fx["field"]) in [str(v) for v in fx.get("when_value_in", [])]:
                attempt(fx["field"], str(fx["value"]))
        elif action == "set_if_equals":
            # {"action": "set_if_equals", "field": "X", "when_value_equals": "Z", "value": "Y"}
            if _get(row, fx["field"]) == str(fx.get("when_value_equals", "")):
                attempt(fx["field"], str(fx["value"]))
        elif action == "set_field_mapped":
            src_val = _get(row, fx["source_field"])
            if src_val in fx["map"]:
                attempt(fx["field"], str(fx["map"][src_val]))
        elif action == "set_by_prefix":
            prefix = _get(row, fx["prefix_field"])
            defaults = fx.get("default_by_prefix", {})
            if prefix and prefix in defaults:
                attempt(fx["field"], str(defaults[prefix]))
        elif action:
            logging.getLogger(__name__).warning("Unknown fix action %r in rule - skipping", action)
    return changes


def run_consistency_engine(
    df: pd.DataFrame,
    allowed_options: dict[str, list[str]],
    rules: list[dict[str, Any]],
    max_passes: int = 5,
    stop_when_no_changes: bool = True,
    feature_id_col: str = FEATURE_ID_COLUMN,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[AuditEvent]]:
    """Run the consistency rules engine over ``df`` (modified in place and returned) for up to ``max_passes``."""
    allowed_sets = build_allowed_sets(allowed_options)
    audit: list[AuditEvent] = []

    if feature_id_col not in df.columns:
        df[feature_id_col] = ""

    def event(pass_num, rule_id, fid, row_idx, severity, status, message, field=None, old=None, new=None):
        return AuditEvent(pass_num, rule_id, fid, int(row_idx), severity, status, field, old, new, message)

    for pass_num in range(1, max_passes + 1):
        changes_in_pass = 0
        violations_in_pass = 0

        for row_idx in df.index:
            row = df.loc[row_idx]
            fid = _get(row, feature_id_col) or None

            for rule in rules:
                rule_id = rule.get("id", "rule_without_id")
                severity = rule.get("severity", "error")
                when_cond = rule.get("when", {})
                then_clauses = rule.get("then", []) or []

                try:
                    if when_cond and not eval_condition(when_cond, row):
                        continue
                except Exception as e:  # noqa: BLE001 - a malformed rule must not abort the run
                    audit.append(event(pass_num, rule_id, fid, row_idx, severity, "SKIPPED", f"Error evaluating WHEN: {e}"))
                    continue

                violated = False
                failed_msgs = []
                for clause in then_clauses:
                    try:
                        ok = eval_then_clause(clause, row)
                    except Exception as e:  # noqa: BLE001
                        ok = False
                        failed_msgs.append(f"THEN eval error ({clause}): {e}")
                    if not ok:
                        violated = True
                        failed_msgs.append(f"Constraint failed: {clause}")
                if not violated:
                    continue

                violations_in_pass += 1
                audit.append(
                    event(pass_num, rule_id, fid, row_idx, severity, "VIOLATION", " | ".join(failed_msgs)[:2000])
                )
                try:
                    changes = apply_fix_actions(df, row_idx, df.loc[row_idx], rule, allowed_sets)
                    changes_in_pass += len(changes)
                    for field, old_v, new_v, note in changes:
                        audit.append(event(pass_num, rule_id, fid, row_idx, severity, "FIXED", note, field, old_v, new_v))
                    if changes:
                        row = df.loc[row_idx]
                except Exception as e:  # noqa: BLE001
                    audit.append(event(pass_num, rule_id, fid, row_idx, severity, "SKIPPED", f"Error applying fix: {e}"))

        log(f"[pass {pass_num}/{max_passes}] violations={violations_in_pass:,} fixes_applied={changes_in_pass:,}", verbose)
        if stop_when_no_changes and changes_in_pass == 0:
            break

    return df, audit


# ============================================================================
# MAIN POSTPROCESSOR CLASS
# ============================================================================


class USGPostProcessor:
    """Fill missing columns -> validate against options_lookup.tsv -> enforce consistency rules."""

    def __init__(
        self,
        options_lookup_path: str | Path | None = None,
        consistency_rules_path: str | Path | None = None,
        verbose: bool = True,
        required_columns: list[str] | None = None,
        derivations: dict | str | Path | None = None,
    ):
        """``derivations``: a bundle's ``postprocess_derivations.json`` (or its dict) to fill non-predicted columns
        from ResStock consistently with the known and predicted ones; without it only static defaults are used."""
        default_options, default_rules = get_default_resource_paths()
        self.options_lookup_path = Path(options_lookup_path or default_options)
        self.consistency_rules_path = Path(consistency_rules_path or default_rules)
        self.verbose = verbose
        self.required_columns = required_columns if required_columns is not None else load_required_columns()

        log("Loading post-processor resources...", verbose)
        t = Timer("Load options_lookup.tsv", verbose)
        self.allowed_options = load_allowed_options(self.options_lookup_path)
        t.done(extra=f"[params={len(self.allowed_options):,}]")

        t = Timer("Load consistency_rules.json", verbose)
        rules_obj = json.loads(self.consistency_rules_path.read_text(encoding="utf-8"))
        self.rules = rules_obj.get("rules", [])
        self.execution = rules_obj.get("execution", {})
        t.done(extra=f"[rules={len(self.rules):,}]")
        self.last_report = None  # postprocess_report.PostprocessReport of the last process() call
        self.last_report_paths: list[str] = []
        from usg.postprocess_derivations import load_derivations

        self.derivations = load_derivations(derivations)
        self.last_fill_log: dict = {}  # column -> {fill method: cells} of the last process_dataframe() call

    def process_dataframe(
        self, df: pd.DataFrame, context: ProjectContext | None = None
    ) -> tuple[pd.DataFrame, list[ValidationIssue], list[AuditEvent]]:
        verbose = self.verbose
        context = context or ProjectContext()
        log("=== Step 1: Fill Missing Columns ===", verbose)
        t = Timer("Fill missing columns", verbose)
        # Location columns the input lacks take the project's resolved values (e.g. Building America Climate Zone)
        # instead of generic defaults.
        location = {c: v for c, v in context.conditioning_values().items() if c not in df.columns}
        from usg.postprocess_derivations import WEATHER_STATION

        self.last_fill_log = {}
        df = fill_missing_columns(
            df=df,
            allowed_options=self.allowed_options,
            climate_zone=context.climate_zone,
            state=context.state,
            required_columns=self.required_columns,
            derivations=self.derivations,
            context_values={**context.conditioning_values(), WEATHER_STATION: context.nearest_station,
                            "County and PUMA": context.county_and_puma},
            fill_log=self.last_fill_log,
        )
        for column, value in location.items():
            if column in df.columns and value in self.allowed_options.get(column, [value]):
                df[column] = value
        t.done(extra=f"[cols now={len(df.columns):,}]")

        log("=== Step 2: Schema Validation ===", verbose)
        t = Timer("Schema validation", verbose)
        df, schema_issues = validate_and_fix_schema(df=df, allowed_options=self.allowed_options, fix_invalid_values=True)
        t.done(extra=f"[issues={len(schema_issues):,}]")

        log("=== Step 3: Consistency Processing ===", verbose)
        t = Timer("Consistency processing", verbose)
        df, audit_events = run_consistency_engine(
            df=df,
            allowed_options=self.allowed_options,
            rules=self.rules,
            max_passes=int(self.execution.get("max_passes", 5)),
            stop_when_no_changes=bool(self.execution.get("stop_when_no_changes", True)),
            verbose=verbose,
        )
        t.done(extra=f"[audit_events={len(audit_events):,}]")
        return df, schema_issues, audit_events

    def process(
        self,
        input_csv_path: str | Path,
        output_csv_path: str | Path,
        geojson_path: str | Path | None = None,
        generate_reports: bool = True,
        reports_dir: str | Path | None = None,
        context: ProjectContext | None = None,
        imputed: pd.DataFrame | None = None,
        context_columns: list[str] | tuple[str, ...] = (),
        report_formats: list[str] | tuple[str, ...] = ("json", "html"),
    ) -> str:
        """Run the full pipeline on a CSV. ``context`` (or ``geojson_path``) supplies the project state/climate zone.

        Always builds a change report (``self.last_report``); with ``generate_reports`` it is written next to the
        other reports. ``imputed`` (True where the model filled a cell, row-aligned with the input) and
        ``context_columns`` let the report tell predicted values from inputs and project location values.
        """
        from usg.postprocess_report import build_postprocess_report, write_postprocess_report

        input_csv_path = Path(input_csv_path)
        output_csv_path = Path(output_csv_path)
        reports_dir = Path(reports_dir) if reports_dir else output_csv_path.parent
        verbose = self.verbose
        overall = Timer("TOTAL post-processing", verbose)

        log(f"Reading input CSV: {input_csv_path}", verbose)
        df = read_csv_strings(input_csv_path)
        before = df.copy()
        if context is None and geojson_path:
            context = read_project_context(geojson_path)
        if context is not None:
            log(f"  Project state: {context.state}  climate zone: {context.climate_zone}", verbose)

        df, schema_issues, audit_events = self.process_dataframe(df, context)
        self.last_report = build_postprocess_report(
            before, df, schema_issues, audit_events, self.rules, imputed=imputed, context_columns=context_columns,
            fill_log=self.last_fill_log,
        )
        self.last_report_paths = []

        if generate_reports:
            reports_dir.mkdir(parents=True, exist_ok=True)
            if schema_issues:
                report_path = reports_dir / "schema_validation_report.csv"
                pd.DataFrame([vars(x) for x in schema_issues]).to_csv(report_path, index=False)
                log(f"  Schema report saved to: {report_path}", verbose)
            if audit_events:
                audit_path = reports_dir / "consistency_audit_report.csv"
                pd.DataFrame([vars(a) for a in audit_events]).to_csv(audit_path, index=False)
                log(f"  Consistency report saved to: {audit_path}", verbose)
            self.last_report_paths = write_postprocess_report(
                self.last_report, reports_dir, formats=report_formats, meta={"input_csv": str(input_csv_path)}
            )
            log(f"  Change report saved to: {reports_dir / 'postprocess_report.html'}", verbose)

        log(f"Saving output: {output_csv_path}", verbose)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        overall.done()
        return f"Successfully processed {len(df)} buildings. Output saved to {output_csv_path}"


def get_default_resource_paths() -> tuple[Path, Path]:
    """Paths to the packaged options_lookup.tsv and consistency_rules.json."""
    return options_lookup_path(), consistency_rules_path()
