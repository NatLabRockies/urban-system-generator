import json

from usg.geojson import (
    ProjectContext,
    geojson_to_dataframe,
    project_context_from_geojson,
    read_project_context,
    site_origin_point,
)
from usg.location import LocationOverrides
from usg.testing.fixtures import tiny_schema

ZONE = "ASHRAE IECC Climate Zone 2004"
BA = "Building America Climate Zone"
SITE_ORIGIN = (39.302104, -76.6056777)  # examples/example_geojson.json


def _example(helpers) -> dict:
    return json.loads(helpers.example_geojson.read_text(encoding="utf-8"))


def _site_origin(data: dict) -> dict:
    return next(f for f in data["features"] if f["properties"].get("type") == "Site Origin")


def test_example_project_location(helpers):
    ctx = read_project_context(helpers.example_geojson)
    assert ctx.summary().startswith(
        "Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state)"
        " · nearest station Baltimore Blt Washngtn IntL"
    )
    assert (ctx.latitude, ctx.longitude, ctx.coordinates_source) == (*SITE_ORIGIN, "Site Origin")
    assert ctx.nearest_station_km < 20 and ctx.warnings == ()
    record = ctx.location_json()
    assert record["values"][BA] == "Mixed-Humid" and record["sources"]["State"].startswith("weather file (USA_MD_")
    # Buffalo: the project's climate_zone (6A) is not a source; the Site Origin's station gives 5A, without a warning
    buffalo = read_project_context(helpers.combined_geojson)
    assert (buffalo.state, buffalo.climate_zone, buffalo.building_america_zone) == ("NY", "5A", "Cold")
    assert buffalo.climate_zone_source.startswith("nearest station") and buffalo.warnings == ()


def test_only_the_weather_filename_and_site_origin_locate_a_project(helpers):
    data = _example(helpers)
    assert site_origin_point(data) == SITE_ORIGIN
    reference = project_context_from_geojson(data)

    # project coordinates, a project climate zone and the building footprints are not sources
    decoys = {**data, "project": {**data["project"], "latitude": 29.76, "longitude": -95.37, "climate_zone": "2A"}}
    assert project_context_from_geojson(decoys) == reference

    no_origin = {**data, "features": [f for f in data["features"] if f["properties"].get("type") != "Site Origin"]}
    assert site_origin_point(no_origin) is None
    ctx = project_context_from_geojson(no_origin)  # the weather filename in `project` still names the state
    assert (ctx.state, ctx.latitude, ctx.coordinates_source) == ("MD", None, None)
    assert (ctx.climate_zone, ctx.climate_zone_source) == ("4A", "from state (MD 99%)")
    assert project_context_from_geojson({**no_origin, "project": {"climate_zone": "4A"}}) == ProjectContext()

    site_only = project_context_from_geojson({**data, "project": {}})  # weather filename on the Site Origin
    assert site_only.state_source.startswith("Site Origin weather file (USA_MD_") and site_only.climate_zone == "4A"

    point = {"type": "Point", "coordinates": [SITE_ORIGIN[1], SITE_ORIGIN[0]]}  # GeoJSON order: lon, lat
    anonymous = {"features": [{"type": "Feature", "properties": {"type": "Site Origin"}, "geometry": point}]}
    ctx = project_context_from_geojson(anonymous)  # no weather filename: the station gives the state too
    assert ctx.state == "MD" and ctx.state_source.startswith("nearest station (")
    named = {"features": [{"type": "Feature", "properties": {"name": "Site Origin"}, "geometry": point}]}
    assert site_origin_point(named) == SITE_ORIGIN

    projected = {"features": [{"type": "Feature", "properties": {"type": "Site Origin"},
                               "geometry": {"type": "Point", "coordinates": [500000.0, 4300000.0]}}]}
    warnings: list[str] = []
    assert site_origin_point(projected, warnings) is None and "not longitude/latitude" in warnings[0]


def test_weather_filename_state_wins_over_the_site_origin_station(helpers):
    data = _example(helpers)
    _site_origin(data)["geometry"]["coordinates"] = [-80.8431, 35.2271]  # Charlotte, NC
    ctx = project_context_from_geojson(data)
    assert ctx.state == "MD" and any("disagrees" in w and "NC" in w for w in ctx.warnings)
    assert (ctx.climate_zone, ctx.climate_zone_source) == ("4A", "from state (MD 99%)")  # NC's 3A never occurs in MD


def test_explicit_overrides_win_over_the_geojson(helpers):
    ctx = read_project_context(helpers.example_geojson, LocationOverrides(state="VA", climate_zone="5A"))
    assert (ctx.state, ctx.state_source, ctx.climate_zone, ctx.building_america_zone) == ("VA", "explicit", "5A", "Cold")
    assert ctx.census_division == "South Atlantic" and ctx.warnings == ()
    moved = read_project_context(helpers.example_geojson, LocationOverrides(latitude=37.54, longitude=-77.43))
    assert moved.coordinates_source == "explicit" and (moved.latitude, moved.longitude) == (37.54, -77.43)
    assert moved.state == "MD" and any("disagrees" in w and "VA" in w for w in moved.warnings)


def test_every_building_gets_the_project_location(helpers):
    schema = tiny_schema()  # its only location column is State (vocab MD, VA, DC)
    assert (geojson_to_dataframe(helpers.example_geojson, schema=schema)["State"] == "MD").all()
    alaska = geojson_to_dataframe(helpers.example_geojson, schema=schema, context=ProjectContext(state="AK"))
    assert alaska["State"].isna().all()
    plain = geojson_to_dataframe(helpers.example_geojson)
    location = read_project_context(helpers.example_geojson).conditioning_values()
    assert set(location) == {"State", ZONE, BA, "Census Division"}
    assert all((plain[column] == value).all() for column, value in location.items())
    assert "PUMA Metro Status" not in plain.columns
    legacy = ProjectContext(state="VA", climate_zone="4A")  # hand-built contexts keep working
    assert legacy.conditioning_values() == {"State": "VA", ZONE: "4A", "Census Division": "South Atlantic"}
