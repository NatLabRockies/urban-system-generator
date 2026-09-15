import json

import pytest

from usg.errors import InputValidationError
from usg.geojson import (
    GeoJSONProcessor as P,
)
from usg.geojson import (
    ProjectContext,
    _allowed_options,
    geojson_to_dataframe,
    project_context_from_geojson,
    read_project_context,
    state_from_weather_filename,
)
from usg.testing.fixtures import tiny_schema

ALLOWED = _allowed_options()

SYSTEM_TYPES = [
    f"Residential - {heat} and {cool}"
    for heat in ("electric resistance", "furnace", "boiler")
    for cool in ("no cooling", "central air conditioner", "room air conditioner", "evaporative cooler")
] + [
    "Residential - air-to-air heat pump", "Residential - mini-split heat pump", "Residential - ground-to-air heat pump",
    "Residential - something new", "VAV chiller with gas boiler reheat", None, "",
]  # fmt: skip
BUILDING_TYPES = ["Multifamily", "Single-Family", "Single-Family Attached", "Single-Family Detached", "Mobile Home",
                  "Office", "Food service", None, ""]
UNITS = [None, 0, 1, 2, 3, 4, 5, 8, 50, 500, "3", "abc", 2.0]


def _valid(parameter, value):
    assert value is None or value in ALLOWED[parameter], (parameter, value)


def test_mapper_outputs_are_valid_options_or_none():
    for year in range(1700, 2040):
        _valid("Vintage", P.map_vintage(year))
        _valid("Vintage ACS", P.map_vintage_acs(year))
    for area in [1, 250, 499.9, 500, 749, 750, 999, 1000, 1499, 1500, 1999, 2000, 2499, 2500, 2999, 3000, 3999, 4000, 1e6]:
        _valid("Geometry Floor Area", P.map_floor_area_to_category(area))
        _valid("Geometry Floor Area Bin", P.map_floor_area_to_bin(area))
    for btype in BUILDING_TYPES:
        for units in UNITS:
            props = {"building_type": btype, "number_of_residential_units": units}
            _valid("Geometry Building Type RECS", P.map_building_type(btype, props))
            _valid("Geometry Building Number Units SFA", P.map_units_sfa(units, props))
    for value in ["slab", "crawlspace - vented", "crawlspace - unvented", "crawlspace - conditioned",
                  "basement - unconditioned", "basement - conditioned", "ambient", "pier", None]:
        _valid("Geometry Foundation Type", P.map_foundation_type(value))
    for value in ["attic - vented", "attic - unvented", "attic - conditioned", "flat roof", "mansard", None]:
        _valid("Geometry Attic Type", P.map_attic_type(value))
    for value in ["electricity", "natural gas", "fuel oil", "propane", "wood", "other", "none", "coal", None]:
        _valid("Heating Fuel", P.map_heating_fuel(value))
    for value in SYSTEM_TYPES:
        _valid("HVAC Cooling Type", P.map_hvac_cooling_type(value))
    for value in [*range(-2, 45), "2", 2.0, "x", None, True]:
        _valid("Geometry Stories", P.map_stories(value))
        _valid("Bedrooms", P.map_bedrooms(value))


@pytest.mark.parametrize("path_attr", ["example_geojson", "combined_geojson"])
def test_mapped_example_features_are_valid(helpers, path_attr):
    processor = P()
    data = processor.read_geojson(getattr(helpers, path_attr))
    for feature in processor.extract_residential_buildings(data):
        for column, value in processor.map_properties(feature).items():
            if column != "Building":
                _valid(column, value)


def test_mapper_fixes():
    assert P.map_building_type("Office", {}) is None
    assert P.map_building_type("Mobile Home", {}) == "Mobile Home"
    assert P.map_building_type("Multifamily", {"number_of_residential_units": "3"}) == "Multi-Family with 2 - 4 Units"
    assert P.map_foundation_type("pier") is None
    assert P.map_units_sfa(1, {"building_type": "Single-Family Detached"}) == "None"
    assert P.map_units_sfa(3, {"building_type": "Single-Family Attached"}) == "3"
    assert P.map_units_sfa(1, {"building_type": "Single-Family Attached"}) is None
    assert P.map_bedrooms(0) == "1" and P.map_bedrooms(9) == "5"
    assert P.map_stories(16) is None and P.map_stories("2") == "2"
    assert P.map_hvac_cooling_type("Residential - furnace and central air conditioner") == "Central AC"
    assert P.map_hvac_cooling_type("Residential - boiler and no cooling") == "None"
    assert "Mobile Home" in P.RESIDENTIAL_TYPES


def test_project_context_from_example(helpers):
    ctx = read_project_context(helpers.example_geojson)
    assert ctx.state == "MD" and ctx.climate_zone == "4A"
    assert "USA_MD_Baltimore" in ctx.state_source
    assert ctx.conditioning_values() == {
        "State": "MD", "Census Division": "South Atlantic", "ASHRAE IECC Climate Zone 2004": "4A",
        "Building America Climate Zone": "Mixed-Humid",
    }
    assert read_project_context(helpers.combined_geojson).state == "NY"


def test_project_context_fallbacks():
    site = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-104.99, 39.74]},
            "properties": {"type": "Site Origin", "weather_filename": "USA_CO_Denver.epw", "climate_zone": "7A"}}
    ctx = project_context_from_geojson({"project": {}, "features": [site]})
    # the climate zone comes from the Site Origin's nearest weather station, never from a climate_zone property
    assert ctx.state == "CO" and ctx.climate_zone == "5B" and "Site Origin" in ctx.state_source
    assert project_context_from_geojson({"project": {"weather_filename": "CAN_ON_Toronto.epw"}}) == ProjectContext(
        weather_filename="CAN_ON_Toronto.epw"
    )
    assert project_context_from_geojson({"project": {"climate_zone": "9Z"}}).climate_zone is None
    assert state_from_weather_filename(r"C:\weather\USA_TX_Austin.epw") == "TX"
    assert state_from_weather_filename("/data/USA_ZZ_Nowhere.epw") is None


def test_legacy_readers_are_reexported(helpers, tmp_path):
    from usg.postprocessor import read_project_climate_zone, read_project_state

    assert read_project_state(helpers.example_geojson) == "MD"
    assert read_project_climate_zone(helpers.example_geojson) == "4A"
    assert read_project_state(tmp_path / "missing.json") is None


def test_geojson_to_dataframe_with_schema(helpers):
    schema = tiny_schema()
    df = geojson_to_dataframe(helpers.example_geojson, schema=schema)
    assert len(df) == 77
    assert list(df.columns[: len(schema) + 1]) == ["Building", *schema.names]
    assert (df["State"] == "MD").all() and df["Geometry Garage"].isna().all()
    assert "ASHRAE IECC Climate Zone 2004" not in df.columns  # not in the tiny schema
    assert "Geometry Foundation Type" in df.columns  # mapped extras are kept after the schema columns

    plain = geojson_to_dataframe(helpers.example_geojson)
    assert {"State", "ASHRAE IECC Climate Zone 2004", "Census Division"} <= set(plain.columns)
    override = geojson_to_dataframe(helpers.example_geojson, context=ProjectContext(state="VA"))
    assert (override["State"] == "VA").all() and "ASHRAE IECC Climate Zone 2004" not in override.columns


def test_non_residential_buildings_are_skipped(helpers, tmp_path):
    df = geojson_to_dataframe(helpers.combined_geojson)
    assert len(df) == 5
    assert set(df["Geometry Building Type RECS"]) <= set(ALLOWED["Geometry Building Type RECS"])

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"type": "Building", "building_type": "Office"}}
    ]}))
    with pytest.raises(InputValidationError, match="no residential"):
        geojson_to_dataframe(empty)
    with pytest.raises(InputValidationError, match="not found"):
        read_project_context(tmp_path / "missing.json")
