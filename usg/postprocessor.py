"""
Urban System Generator - Post-Processor Module

This module provides post-processing functionality to transform the ML inference output
into a validated, URBANopt-BuildStock compatible format.

The post-processing pipeline consists of three sequential steps:
1. Missing Column Filler - Adds missing columns required by URBANopt-BuildStock
2. Schema Validator - Validates and fixes values against options_lookup.tsv
3. Consistency Processor - Enforces cross-field dependencies and constraints

Author: NREL Urban System Generator Team
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# ============================================================================
# LOGGING HELPERS
# ============================================================================

def _now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()

    def done(self, extra: str = "") -> None:
        dt = time.perf_counter() - self.t0
        suffix = f" {extra}" if extra else ""
        log(f"{self.label} ✓ ({dt:.3f}s){suffix}")


# ============================================================================
# OPTIONS LOOKUP HELPERS (shared across all steps)
# ============================================================================

def load_allowed_options(options_lookup_tsv: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Read options_lookup.tsv and build {Parameter Name: [Option Name, ...]}.
    Robust: only trust first 2 columns (param, option).
    """
    options_lookup_tsv = Path(options_lookup_tsv)
    allowed: Dict[str, List[str]] = {}

    with options_lookup_tsv.open("r", encoding="utf-8", errors="ignore") as f:
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

    # De-dup while preserving order
    for k, v in list(allowed.items()):
        seen = set()
        dedup = []
        for x in v:
            if x not in seen:
                dedup.append(x)
                seen.add(x)
        allowed[k] = dedup

    return allowed


def coerce_to_allowed(value: Optional[str], allowed: List[str]) -> Optional[str]:
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


def build_allowed_sets(allowed_options: Dict[str, List[str]]) -> Dict[str, Tuple[set, Dict[str, str]]]:
    """
    For each param:
      - exact allowed set
      - case-insensitive map lower(opt) -> canonical opt
    """
    out: Dict[str, Tuple[set, Dict[str, str]]] = {}
    for p, opts in allowed_options.items():
        exact = set(opts)
        ci_map: Dict[str, str] = {}
        for o in opts:
            lo = o.lower()
            if lo not in ci_map:
                ci_map[lo] = o
        out[p] = (exact, ci_map)
    return out


# ============================================================================
# STEP 1: MISSING COLUMN FILLER
# ============================================================================

# Parameters that need to be added for URBANopt-BuildStock compatibility (this temporary fix to updated the old usg model to the new buildstock schema )
MISSING_PARAMETERS: List[str] = [
    "AHS Region", "AIANNH Area", "ASHRAE IECC Climate Zone 2004",
    "ASHRAE IECC Climate Zone 2004 - Sub-CZ Split", "Area Median Income",
    "Battery", "Building America Climate Zone", "CEC Climate Zone",
    "Census Division", "Census Division RECS", "Census Region", "City",
    "Cooling Unavailable Days", "County", "County Metro Status",
    "County and PUMA", "Custom State", "Electric Vehicle Battery",
    "Electric Vehicle Charge At Home", "Electric Vehicle Charger",
    "Electric Vehicle Miles Traveled", "Electric Vehicle Outlet Access",
    "Electric Vehicle Ownership", "Energystar Climate Zone 2023",
    "Federal Poverty Level", "Generation And Emissions Assessment Region",
    "Geometry Story Bin", "HVAC Cooling Autosizing Factor",
    "HVAC Heating Autosizing Factor", "HVAC Secondary Heating Type",
    "HVAC System Is Scaled", "Has PV", "Heating Unavailable Days",
    "ISO RTO Region", "Income", "Income RECS2015", "Income RECS2020",
    "Location Region", "Metropolitan and Micropolitan Statistical Area",
    "PUMA", "PUMA Metro Status", "PV Orientation", "PV System Size",
    "REEDS Balancing Area", "Radiant Barrier", "State",
    "State Metro Median Income", "Tenure", "Vacancy Status",
]

# Columns to drop from the inference output
EXTRA_COLS_TO_DROP = [
    "Electric Vehicle",
    "Solar Hot Water",
    "site_energy.total.energy_consumption.kwh",
]

# Default values for missing parameters
DEFAULT_FILL_VALUES: Dict[str, str] = {
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


def read_project_climate_zone(geojson_path: Union[str, Path]) -> Optional[str]:
    """Extract climate_zone from GeoJSON project or features."""
    p = Path(geojson_path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        proj = data.get("project", {})
        cz = proj.get("climate_zone")
        if cz:
            return str(cz).strip()

        feats = data.get("features", [])
        for f in feats:
            props = (f or {}).get("properties", {})
            if "climate_zone" in props and props["climate_zone"]:
                return str(props["climate_zone"]).strip()
    return None


def read_project_state(geojson_path: Union[str, Path]) -> Optional[str]:
    """
    Extract state abbreviation from GeoJSON weather_filename.
    
    Weather filenames follow the pattern: USA_NY_Buffalo-Greater.Buffalo.Intl.AP.725280_TMY3.epw
    where NY is the state abbreviation.
    
    Args:
        geojson_path: Path to GeoJSON file
        
    Returns:
        Two-letter state abbreviation (e.g., 'NY') or None if not found
    """
    import re
    
    p = Path(geojson_path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        return None
    
    # Check project-level weather_filename first
    proj = data.get("project", {})
    weather_filename = proj.get("weather_filename", "")
    
    # Also check Site Origin feature if not in project
    if not weather_filename:
        feats = data.get("features", [])
        for f in feats:
            props = (f or {}).get("properties", {})
            if props.get("type") == "Site Origin":
                weather_filename = props.get("weather_filename", "")
                if weather_filename:
                    break
    
    if not weather_filename:
        return None
    
    # Extract state from weather filename pattern: USA_XX_...
    match = re.match(r'^USA_([A-Z]{2})_', weather_filename)
    if match:
        return match.group(1)
    
    return None


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
}

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
}

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
}

# Location-related parameters that should be state-aware
LOCATION_PARAMETERS = [
    "State", "Custom State", "City", "County", "PUMA", 
    "Census Division", "Census Division RECS", "Census Region",
    "ISO RTO Region",
]


def get_state_aware_location_defaults(
    state: str, 
    allowed_options: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Get location-related default values that match the given state.
    
    Args:
        state: Two-letter state abbreviation (e.g., 'NY')
        allowed_options: Dictionary of allowed options from options_lookup.tsv
        
    Returns:
        Dictionary of parameter -> default value for location fields
    """
    defaults = {}
    
    if not state:
        return defaults
    
    state = state.upper()
    
    # State and Custom State
    if "State" in allowed_options and state in allowed_options["State"]:
        defaults["State"] = state
    
    if "Custom State" in allowed_options:
        if state in allowed_options["Custom State"]:
            defaults["Custom State"] = state
        elif "Others" in allowed_options["Custom State"]:
            defaults["Custom State"] = "Others"
    
    # City - find first city matching the state (format: "NY, Buffalo")
    if "City" in allowed_options:
        state_cities = [c for c in allowed_options["City"] if c.startswith(f'"{state},') or c.startswith(f'{state},')]
        if state_cities:
            # Clean up quotes if present
            defaults["City"] = state_cities[0].strip('"')
    
    # County - find first county matching the state (format: "NY, Erie County")
    if "County" in allowed_options:
        state_counties = [c for c in allowed_options["County"] if c.startswith(f'"{state},') or c.startswith(f'{state},')]
        if state_counties:
            defaults["County"] = state_counties[0].strip('"')
    
    # PUMA - find first PUMA matching the state (format: "NY, 00100")
    if "PUMA" in allowed_options:
        state_pumas = [p for p in allowed_options["PUMA"] if p.startswith(f'"{state},') or p.startswith(f'{state},')]
        if state_pumas:
            defaults["PUMA"] = state_pumas[0].strip('"')
    
    # Metropolitan and Micropolitan Statistical Area - find first matching state (format: "Albany-Schenectady-Troy, NY MSA")
    if "Metropolitan and Micropolitan Statistical Area" in allowed_options:
        # Match pattern: "..., NY MSA" or "..., NY MicroSA" or "..., NY-..."
        state_metros = [
            m for m in allowed_options["Metropolitan and Micropolitan Statistical Area"] 
            if f', {state} ' in m or f', {state}-' in m
        ]
        if state_metros:
            defaults["Metropolitan and Micropolitan Statistical Area"] = state_metros[0].strip('"')
    
    # Census Division
    census_div = STATE_TO_CENSUS_DIVISION.get(state)
    if census_div:
        if "Census Division" in allowed_options and census_div in allowed_options["Census Division"]:
            defaults["Census Division"] = census_div
        if "Census Division RECS" in allowed_options and census_div in allowed_options["Census Division RECS"]:
            defaults["Census Division RECS"] = census_div
    
    # Census Region
    census_reg = STATE_TO_CENSUS_REGION.get(state)
    if census_reg and "Census Region" in allowed_options and census_reg in allowed_options["Census Region"]:
        defaults["Census Region"] = census_reg
    
    # ISO RTO Region
    iso_rto = STATE_TO_ISO_RTO.get(state)
    if iso_rto and "ISO RTO Region" in allowed_options and iso_rto in allowed_options["ISO RTO Region"]:
        defaults["ISO RTO Region"] = iso_rto
    
    return defaults


def force_none_string_everywhere(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OUTPUT CSV contains literal text 'None' for any missing value."""
    df = df.replace({None: pd.NA})
    df = df.where(pd.notna(df), pd.NA)
    df = df.replace(
        to_replace=r"^\s*(nan|NaN|NAN|<NA>|None|null|NULL)\s*$",
        value=pd.NA,
        regex=True,
    )
    return df.fillna("None")


def fill_missing_columns(
    df: pd.DataFrame,
    allowed_options: Dict[str, List[str]],
    climate_zone: Optional[str] = None,
    state: Optional[str] = None,
    missing_parameters: Optional[List[str]] = None,
    extra_cols_to_drop: Optional[List[str]] = None,
    write_literal_none: bool = True,
) -> pd.DataFrame:
    """
    Add missing columns and fill them using DEFAULT_FILL_VALUES.
    
    Args:
        df: Input DataFrame from inference
        allowed_options: Dictionary of allowed options from options_lookup.tsv
        climate_zone: Optional climate zone from GeoJSON
        state: Optional two-letter state abbreviation from GeoJSON weather file
        missing_parameters: List of parameters to add
        extra_cols_to_drop: Columns to remove
        write_literal_none: If True, convert all NA values to 'None' string
    
    Returns:
        Processed DataFrame
    """
    missing_parameters = missing_parameters or MISSING_PARAMETERS
    extra_cols_to_drop = extra_cols_to_drop or EXTRA_COLS_TO_DROP
    
    # Get state-aware location defaults if state is provided
    state_location_defaults = {}
    if state:
        state_location_defaults = get_state_aware_location_defaults(state, allowed_options)
    
    # Ensure 'Feature ID' first column (copy of 'Building')
    if "Building" not in df.columns:
        raise ValueError("Input CSV must contain a 'Building' column.")
    if "Feature ID" in df.columns:
        df["Feature ID"] = df["Building"]
        cols = ["Feature ID"] + [c for c in df.columns if c != "Feature ID"]
        df = df[cols]
    else:
        df.insert(0, "Feature ID", df["Building"])
    
    # Add missing parameters
    for param in missing_parameters:
        if param in df.columns:
            continue

        allowed = allowed_options.get(param, [])
        
        # Priority: state-aware location defaults > climate zone > DEFAULT_FILL_VALUES
        if param in state_location_defaults:
            chosen = state_location_defaults[param]
        else:
            chosen = DEFAULT_FILL_VALUES.get(param, "")

        # Prefer geojson climate zone for these fields
        if climate_zone and param in {
            "ASHRAE IECC Climate Zone 2004",
            "ASHRAE IECC Climate Zone 2004 - Sub-CZ Split",
        } and allowed:
            cz_val = coerce_to_allowed(climate_zone, allowed)
            if cz_val:
                chosen = cz_val

        # If chosen is empty, fallback to first allowed option
        if (chosen is None or str(chosen).strip() == "") and allowed:
            chosen = allowed[0]

        # If chosen is not allowed, coerce or fallback
        if allowed:
            chosen = coerce_to_allowed(chosen, allowed) or allowed[0]

        df[param] = str(chosen) if chosen else "None"

    # Drop extra columns
    for col in extra_cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    if write_literal_none:
        df = force_none_string_everywhere(df)

    return df


# ============================================================================
# STEP 2: SCHEMA VALIDATOR
# ============================================================================

DEFAULT_VOID_LIKE = {
    "", " ", "none", "null", "nan", "<na>", "n/a", "na", "void", "unknown"
}

# Per-column override defaults used when fixing invalid values
DEFAULT_VALUE_OVERRIDES: Dict[str, str] = {
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
    feature_id: Optional[str]
    column: str
    value: str
    issue_type: str
    message: str
    allowed_options: str
    fixed_to_default_value: str


def norm_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def is_void_like(s: str, void_like: set) -> bool:
    return s.lower() in void_like


def pick_default_value(col: str, allowed: List[str], overrides: Dict[str, str]) -> Optional[str]:
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
    allowed_options: Dict[str, List[str]],
    feature_id_col: str = "Feature ID",
    void_like: Optional[set] = None,
    fix_invalid_values: bool = True,
    default_value_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
    """
    Validate DataFrame against schema and optionally fix invalid values.
    
    Args:
        df: Input DataFrame
        allowed_options: Dictionary of allowed options
        feature_id_col: Column name for feature IDs
        void_like: Set of values considered void/empty
        fix_invalid_values: If True, replace invalid values with defaults
        default_value_overrides: Custom default values per column
    
    Returns:
        Tuple of (fixed DataFrame, list of issues)
    """
    void_like = void_like or DEFAULT_VOID_LIKE
    default_value_overrides = default_value_overrides or DEFAULT_VALUE_OVERRIDES
    
    allowed_sets = build_allowed_sets(allowed_options)
    issues: List[ValidationIssue] = []
    fixed_df = df.copy(deep=True)

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

            # Check if value is in allowed options
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
    feature_id: Optional[str]
    row_index: int
    severity: str
    status: str  # "FIXED" | "VIOLATION" | "SKIPPED"
    field: Optional[str]
    old_value: Optional[str]
    new_value: Optional[str]
    message: str


def _get(row: pd.Series, field: str) -> str:
    v = row.get(field, "")
    return "" if v is None else str(v).strip()


def _starts_with_any(s: str, prefixes: List[str]) -> bool:
    s2 = (s or "").strip()
    return any(s2.startswith(p) for p in prefixes)


def eval_condition(cond: Dict[str, Any], row: pd.Series) -> bool:
    """Evaluate a condition against a row."""
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
        mapping = val
        return s in set(str(v) for v in mapping.values())

    raise ValueError(f"Unsupported condition op: {op}")


def eval_then_clause(clause: Dict[str, Any], row: pd.Series) -> bool:
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
        mapping = val or {}
        return s in set(str(v) for v in mapping.values())

    raise ValueError(f"Unsupported THEN op: {op}")


def set_value_if_allowed(
    df: pd.DataFrame,
    row_idx: int,
    field: str,
    new_value: str,
    allowed_sets: Dict[str, Tuple[set, Dict[str, str]]],
) -> Tuple[bool, str]:
    """Sets df.at[row_idx, field] to new_value if it's allowed."""
    if field not in allowed_sets:
        old = df.at[row_idx, field] if field in df.columns else ""
        df.at[row_idx, field] = new_value
        return (str(old).strip() != str(new_value).strip()), "Field not in options_lookup; set anyway."

    exact_allowed, ci_map = allowed_sets[field]
    canonical = None
    if new_value in exact_allowed:
        canonical = new_value
    elif new_value.lower() in ci_map:
        canonical = ci_map[new_value.lower()]
    
    if canonical is None:
        return False, f"Proposed fix value {new_value!r} is not in allowed options for '{field}'."

    old = df.at[row_idx, field] if field in df.columns else ""
    changed = str(old).strip() != str(canonical).strip()
    df.at[row_idx, field] = canonical
    return changed, f"Set to '{canonical}'."


def _eval_fix_condition(cond: Dict[str, Any], row: pd.Series) -> bool:
    """
    Evaluate a condition for fix actions (set_if, set_if_in, set_if_equals).
    Supports: equals, not_equals, in, not_in
    """
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
    
    # Default: condition not met
    return False


def apply_fix_actions(
    df: pd.DataFrame,
    row_idx: int,
    row: pd.Series,
    rule: Dict[str, Any],
    allowed_sets: Dict[str, Tuple[set, Dict[str, str]]],
) -> List[Tuple[str, str, str, str]]:
    """Apply fix actions from a rule. Returns list of (field, old, new, note)."""
    fixes = rule.get("fix", [])
    if not fixes:
        return []

    changes = []
    for fx in fixes:
        action = fx.get("action")

        if action == "set":
            field = fx["field"]
            new_val = str(fx["value"])
            old_val = _get(row, field)
            changed, note = set_value_if_allowed(df, row_idx, field, new_val, allowed_sets)
            if changed:
                changes.append((field, old_val, str(df.at[row_idx, field]), note))

        elif action == "set_if":
            # Conditional set: only apply if the "when" condition is met
            # Format: {"action": "set_if", "when": {...condition...}, "field": "X", "value": "Y"}
            when_cond = fx.get("when", {})
            if when_cond and _eval_fix_condition(when_cond, row):
                field = fx["field"]
                new_val = str(fx["value"])
                old_val = _get(row, field)
                changed, note = set_value_if_allowed(df, row_idx, field, new_val, allowed_sets)
                if changed:
                    changes.append((field, old_val, str(df.at[row_idx, field]), note))

        elif action == "set_if_in":
            # Set field to value if current value is in a list
            # Format: {"action": "set_if_in", "field": "X", "when_value_in": [...], "value": "Y"}
            field = fx["field"]
            when_vals = [str(v) for v in fx.get("when_value_in", [])]
            cur_val = _get(row, field)
            
            if str(cur_val) in when_vals:
                new_val = str(fx["value"])
                old_val = cur_val
                changed, note = set_value_if_allowed(df, row_idx, field, new_val, allowed_sets)
                if changed:
                    changes.append((field, old_val, str(df.at[row_idx, field]), note))

        elif action == "set_if_equals":
            # Set field to value if current value equals a specific value
            # Format: {"action": "set_if_equals", "field": "X", "when_value_equals": "Z", "value": "Y"}
            field = fx["field"]
            when_val = str(fx.get("when_value_equals", ""))
            cur_val = _get(row, field)
            
            if str(cur_val) == when_val:
                new_val = str(fx["value"])
                old_val = cur_val
                changed, note = set_value_if_allowed(df, row_idx, field, new_val, allowed_sets)
                if changed:
                    changes.append((field, old_val, str(df.at[row_idx, field]), note))

        elif action == "set_field_mapped":
            source_field = fx["source_field"]
            field = fx["field"]
            mapping = fx["map"]
            src_val = _get(row, source_field)
            if src_val in mapping:
                new_val = str(mapping[src_val])
                old_val = _get(row, field)
                changed, note = set_value_if_allowed(df, row_idx, field, new_val, allowed_sets)
                if changed:
                    changes.append((field, old_val, str(df.at[row_idx, field]), note))

        elif action == "set_by_prefix":
            field = fx["field"]
            prefix_field = fx["prefix_field"]
            defaults = fx.get("default_by_prefix", {})
            prefix = _get(row, prefix_field)
            if prefix and prefix in defaults:
                new_val = str(defaults[prefix])
                old_val = _get(row, field)
                changed, note = set_value_if_allowed(df, row_idx, field, new_val, allowed_sets)
                if changed:
                    changes.append((field, old_val, str(df.at[row_idx, field]), note))

        else:
            # Unknown action - log warning but don't fail
            if action:
                import logging
                logging.warning(f"Unknown fix action '{action}' in rule - skipping")

    return changes


def run_consistency_engine(
    df: pd.DataFrame,
    allowed_options: Dict[str, List[str]],
    rules: List[Dict[str, Any]],
    max_passes: int = 5,
    stop_when_no_changes: bool = True,
    feature_id_col: str = "Feature ID",
) -> Tuple[pd.DataFrame, List[AuditEvent]]:
    """
    Run consistency rules engine over DataFrame.
    
    Args:
        df: Input DataFrame
        allowed_options: Dictionary of allowed options
        rules: List of consistency rules
        max_passes: Maximum number of passes
        stop_when_no_changes: Stop if no changes in a pass
        feature_id_col: Column name for feature IDs
    
    Returns:
        Tuple of (fixed DataFrame, list of audit events)
    """
    allowed_sets = build_allowed_sets(allowed_options)
    audit: List[AuditEvent] = []
    
    if feature_id_col not in df.columns:
        df[feature_id_col] = ""

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

                # If WHEN is false: skip rule for this row
                try:
                    if when_cond and not eval_condition(when_cond, row):
                        continue
                except Exception as e:
                    audit.append(
                        AuditEvent(
                            pass_num=pass_num,
                            rule_id=rule_id,
                            feature_id=fid,
                            row_index=int(row_idx),
                            severity=severity,
                            status="SKIPPED",
                            field=None,
                            old_value=None,
                            new_value=None,
                            message=f"Error evaluating WHEN: {e}",
                        )
                    )
                    continue

                # If any THEN clause fails => violation
                violated = False
                failed_msgs = []

                for clause in then_clauses:
                    try:
                        ok = eval_then_clause(clause, row)
                    except Exception as e:
                        ok = False
                        failed_msgs.append(f"THEN eval error ({clause}): {e}")
                    if not ok:
                        violated = True
                        failed_msgs.append(f"Constraint failed: {clause}")

                if not violated:
                    continue

                violations_in_pass += 1
                audit.append(
                    AuditEvent(
                        pass_num=pass_num,
                        rule_id=rule_id,
                        feature_id=fid,
                        row_index=int(row_idx),
                        severity=severity,
                        status="VIOLATION",
                        field=None,
                        old_value=None,
                        new_value=None,
                        message=" | ".join(failed_msgs)[:2000],
                    )
                )

                # Attempt fix
                try:
                    row_latest = df.loc[row_idx]
                    changes = apply_fix_actions(df, row_idx, row_latest, rule, allowed_sets)

                    if changes:
                        changes_in_pass += len(changes)
                        for field, old_v, new_v, note in changes:
                            audit.append(
                                AuditEvent(
                                    pass_num=pass_num,
                                    rule_id=rule_id,
                                    feature_id=fid,
                                    row_index=int(row_idx),
                                    severity=severity,
                                    status="FIXED",
                                    field=field,
                                    old_value=old_v,
                                    new_value=new_v,
                                    message=note,
                                )
                            )
                except Exception as e:
                    audit.append(
                        AuditEvent(
                            pass_num=pass_num,
                            rule_id=rule_id,
                            feature_id=fid,
                            row_index=int(row_idx),
                            severity=severity,
                            status="SKIPPED",
                            field=None,
                            old_value=None,
                            new_value=None,
                            message=f"Error applying fix: {e}",
                        )
                    )

        log(f"[pass {pass_num}/{max_passes}] violations={violations_in_pass:,} fixes_applied={changes_in_pass:,}")

        if stop_when_no_changes and changes_in_pass == 0:
            break

    return df, audit


# ============================================================================
# MAIN POSTPROCESSOR CLASS
# ============================================================================

class USGPostProcessor:
    """
    Main class for post-processing inference output for URBANopt-BuildStock compatibility.
    
    This class orchestrates three processing steps:
    1. Missing Column Filler - Adds required columns
    2. Schema Validator - Validates against options_lookup.tsv
    3. Consistency Processor - Enforces cross-field constraints
    """
    
    def __init__(
        self,
        options_lookup_path: Union[str, Path],
        consistency_rules_path: Union[str, Path],
    ):
        """
        Initialize the post-processor.
        
        Args:
            options_lookup_path: Path to options_lookup.tsv
            consistency_rules_path: Path to consistency_rules.json
        """
        self.options_lookup_path = Path(options_lookup_path)
        self.consistency_rules_path = Path(consistency_rules_path)
        
        # Load resources
        log("Loading post-processor resources...")
        t = Timer("Load options_lookup.tsv")
        self.allowed_options = load_allowed_options(self.options_lookup_path)
        t.done(extra=f"[params={len(self.allowed_options):,}]")
        
        t = Timer("Load consistency_rules.json")
        rules_obj = json.loads(self.consistency_rules_path.read_text(encoding="utf-8"))
        self.rules = rules_obj.get("rules", [])
        self.execution = rules_obj.get("execution", {})
        t.done(extra=f"[rules={len(self.rules):,}]")
    
    def process(
        self,
        input_csv_path: Union[str, Path],
        output_csv_path: Union[str, Path],
        geojson_path: Optional[Union[str, Path]] = None,
        generate_reports: bool = True,
        reports_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Run the full post-processing pipeline.
        
        Args:
            input_csv_path: Path to input CSV (inference output)
            output_csv_path: Path to final output CSV
            geojson_path: Optional path to GeoJSON for climate zone extraction
            generate_reports: If True, generate validation reports
            reports_dir: Directory for reports (default: same as output)
        
        Returns:
            Status message
        """
        input_csv_path = Path(input_csv_path)
        output_csv_path = Path(output_csv_path)
        reports_dir = Path(reports_dir) if reports_dir else output_csv_path.parent
        
        overall = Timer("TOTAL post-processing")
        
        # Read input CSV
        log(f"Reading input CSV: {input_csv_path}")
        t = Timer("Read CSV")
        df = pd.read_csv(input_csv_path, dtype=str, keep_default_na=False)
        t.done(extra=f"[rows={len(df):,} cols={len(df.columns):,}]")
        
        # Get climate zone and state from GeoJSON if provided
        climate_zone = None
        state = None
        if geojson_path:
            log(f"Reading climate zone from GeoJSON: {geojson_path}")
            climate_zone = read_project_climate_zone(geojson_path)
            log(f"  Climate zone: {climate_zone}")
            
            log(f"Reading state from GeoJSON weather file...")
            state = read_project_state(geojson_path)
            log(f"  State: {state}")
        
        # Step 1: Fill missing columns
        log("\n=== Step 1: Fill Missing Columns ===")
        t = Timer("Fill missing columns")
        df = fill_missing_columns(
            df=df,
            allowed_options=self.allowed_options,
            climate_zone=climate_zone,
            state=state,
        )
        t.done(extra=f"[cols now={len(df.columns):,}]")
        
        # Step 2: Validate and fix schema
        log("\n=== Step 2: Schema Validation ===")
        t = Timer("Schema validation")
        df, schema_issues = validate_and_fix_schema(
            df=df,
            allowed_options=self.allowed_options,
            fix_invalid_values=True,
        )
        t.done(extra=f"[issues={len(schema_issues):,}]")
        
        if generate_reports and schema_issues:
            report_path = reports_dir / "schema_validation_report.csv"
            report_df = pd.DataFrame([vars(x) for x in schema_issues])
            report_df.to_csv(report_path, index=False)
            log(f"  Schema report saved to: {report_path}")
        
        # Step 3: Consistency processing
        log("\n=== Step 3: Consistency Processing ===")
        t = Timer("Consistency processing")
        max_passes = int(self.execution.get("max_passes", 5))
        stop_when_no_changes = bool(self.execution.get("stop_when_no_changes", True))
        
        df, audit_events = run_consistency_engine(
            df=df,
            allowed_options=self.allowed_options,
            rules=self.rules,
            max_passes=max_passes,
            stop_when_no_changes=stop_when_no_changes,
        )
        t.done(extra=f"[audit_events={len(audit_events):,}]")
        
        if generate_reports and audit_events:
            audit_path = reports_dir / "consistency_audit_report.csv"
            audit_df = pd.DataFrame([a.__dict__ for a in audit_events])
            audit_df.to_csv(audit_path, index=False)
            log(f"  Consistency report saved to: {audit_path}")
        
        # Save final output
        log(f"\nSaving output: {output_csv_path}")
        t = Timer("Write CSV")
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        t.done(extra=f"[rows={len(df):,} cols={len(df.columns):,}]")
        
        overall.done()
        
        return f"Successfully processed {len(df)} buildings. Output saved to {output_csv_path}"


def get_default_resource_paths() -> Tuple[Path, Path]:
    """Get default paths to post-processor resources."""
    # Try to find resources relative to this file
    module_dir = Path(__file__).parent  # usg/ directory
    options_lookup = module_dir / "resources" / "postprocessor" / "options_lookup.tsv"
    consistency_rules = module_dir / "resources" / "postprocessor" / "consistency_rules.json"
    return options_lookup, consistency_rules
