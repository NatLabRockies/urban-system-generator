import json

import pandas as pd
import pytest

from usg.errors import InputValidationError
from usg.geojson import _allowed_options
from usg.location import (
    BUILDING_AMERICA_COLUMN,
    CENSUS_DIVISION_COLUMN,
    LOCATION_COLUMNS,
    LOCATION_LOOKUP_FORMAT,
    PUMA_METRO_COLUMN,
    LocationLookup,
    canonical_location_label,
    fill_location_columns,
    haversine_km,
    resolve_location,
)
from usg.resources import location_lookup_path

ZONE = "ASHRAE IECC Climate Zone 2004"
BA = BUILDING_AMERICA_COLUMN
DIVISION = CENSUS_DIVISION_COLUMN
BALTIMORE = (39.29, -76.61)
SITE_ORIGIN = (39.302104, -76.6056777)  # examples/example_geojson.json
WHITE_HOUSE = (38.8977, -77.0365)
DENVER = (39.74, -104.99)
CHARLOTTE = (35.2271, -80.8431)


@pytest.fixture(scope="module")
def lookup():
    return LocationLookup.load()


def test_packaged_lookup_is_compact_and_uses_resstock_labels(lookup):
    path = location_lookup_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["format"] == LOCATION_LOOKUP_FORMAT and raw["license"] == "CC-BY-4.0" and "NREL" in raw["attribution"]
    assert path.stat().st_size < 300_000
    assert LocationLookup.load() is lookup  # cached
    assert lookup.data_release == "2024.2" and len(lookup.stations) > 800 and len(lookup.states) == 49
    assert "AK" not in lookup.states and "HI" not in lookup.states
    allowed = _allowed_options()
    for column in LOCATION_COLUMNS:
        assert lookup.labels(column) and set(lookup.labels(column)) <= set(allowed[column]), column
    for station in lookup.stations:
        assert round(station.latitude, 3) == station.latitude and station.n_buildings > 0
        for shares in station.shares.values():
            assert sum(shares.values()) >= 0.985
            assert list(shares.values()) == sorted(shares.values(), reverse=True)


def test_nearest_stations(lookup):
    assert float(haversine_km(*BALTIMORE, *BALTIMORE)) == pytest.approx(0.0)
    assert float(haversine_km(40.7128, -74.0060, 34.0522, -118.2437)) == pytest.approx(3936, rel=0.01)  # NYC-LA
    nearest = lookup.nearest_stations(*BALTIMORE, k=3)
    assert len(nearest) == 3 and [d for _, d in nearest] == sorted(d for _, d in nearest)
    station, km = nearest[0]
    assert station.dominant("State") == ("MD", 1.0) and km < 20
    with pytest.raises(InputValidationError, match="invalid coordinates"):
        lookup.nearest_stations(91.0, 0.0)


def test_summary_line_and_empty_location():
    info = resolve_location(weather_filename="USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw",
                            latitude=SITE_ORIGIN[0], longitude=SITE_ORIGIN[1])
    assert info.summary() == (
        "Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state)"
        f" · nearest station {info.nearest_station} ({info.station_distance_km:.0f} km)"
    )
    assert info.sources()["State"] == "weather file (USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw)"
    assert resolve_location(weather_filename="USA_KY_Louisville.epw").summary() == (
        "Location: KY (weather file) · 4A (from state) · Mixed-Humid (from 4A) · East South Central (from state)"
    )
    empty = resolve_location()
    assert empty.values() == {} and empty.warnings == () and empty.summary() == "Location: state unknown · climate zone unknown"
    assert json.loads(json.dumps(info.to_json()))["values"]["State"] == "MD"


def test_precedence_explicit_then_weather_file_then_coordinates():
    coords = resolve_location(latitude=BALTIMORE[0], longitude=BALTIMORE[1])
    assert coords.values() == {"State": "MD", ZONE: "4A", BA: "Mixed-Humid", DIVISION: "South Atlantic"}
    assert coords.state_source.startswith("nearest station (") and coords.ashrae_zone_source.startswith("nearest station")
    assert coords.building_america_zone_source == "from 4A" and coords.census_division_source == "from state"
    assert coords.nearest_station and coords.station_distance_km < 20 and coords.warnings == ()
    assert coords.summary().endswith(f"nearest station {coords.nearest_station} ({coords.station_distance_km:.0f} km)")

    weather = resolve_location(latitude=BALTIMORE[0], longitude=BALTIMORE[1], weather_filename=r"C:\w\USA_VA_Richmond.epw")
    assert weather.state == "VA" and weather.state_source == "weather file (USA_VA_Richmond.epw)"
    assert weather.census_division == "South Atlantic" and weather.ashrae_zone == "4A"
    assert any("disagrees" in w and "MD 100%" in w for w in weather.warnings)

    explicit = resolve_location(state="de", climate_zone="4a", latitude=BALTIMORE[0], longitude=BALTIMORE[1],
                                weather_filename="USA_VA_Richmond.epw")
    assert (explicit.state, explicit.state_source, explicit.ashrae_zone, explicit.ashrae_zone_source) == (
        "DE", "explicit", "4A", "explicit")
    assert explicit.warnings == ()

    zone = resolve_location(climate_zone="5A", latitude=BALTIMORE[0], longitude=BALTIMORE[1])
    assert (zone.ashrae_zone, zone.ashrae_zone_source, zone.building_america_zone) == ("5A", "explicit", "Cold")
    assert zone.warnings == ()  # an explicit climate zone is a deliberate override: no station comparison


def test_station_zone_must_fit_the_state():
    # weather file VA, coordinates in Charlotte NC: the station's 3A never occurs in VA, so VA's own 4A is used
    guarded = resolve_location(latitude=CHARLOTTE[0], longitude=CHARLOTTE[1], weather_filename="USA_VA_Richmond.epw")
    assert guarded.state == "VA" and any("disagrees" in w and "NC" in w for w in guarded.warnings)
    assert (guarded.ashrae_zone, guarded.ashrae_zone_source) == ("4A", "from state (VA 100%)")
    unguarded = resolve_location(latitude=CHARLOTTE[0], longitude=CHARLOTTE[1])
    assert (unguarded.state, unguarded.ashrae_zone) == ("NC", "3A")


def test_md_dc_border():
    dc = resolve_location(latitude=WHITE_HOUSE[0], longitude=WHITE_HOUSE[1])
    # Reagan National's weather serves MD, DC and VA homes: no confident state, but the climate zones agree
    assert dc.state is None and dc.census_division is None
    assert (dc.ashrae_zone, dc.building_america_zone) == ("4A", "Mixed-Humid")
    assert any("several states" in w and "DC" in w for w in dc.warnings)
    arlington = resolve_location(latitude=WHITE_HOUSE[0], longitude=WHITE_HOUSE[1],
                                 weather_filename="USA_VA_Arlington-Ronald.Reagan.Washington.Natl.AP.724050_TMY3.epw")
    assert arlington.state == "VA" and arlington.warnings == ()  # ambiguous coordinates don't contradict the file
    assert resolve_location(state="DC").values() == {"State": "DC", ZONE: "4A", BA: "Mixed-Humid", DIVISION: "South Atlantic"}
    assert resolve_location(latitude=BALTIMORE[0], longitude=BALTIMORE[1]).state == "MD"


def test_state_values_only_when_dominant():
    ky = resolve_location(state="KY")
    assert (ky.ashrae_zone, ky.ashrae_zone_source) == ("4A", "from state (KY 100%)")
    assert (ky.building_america_zone, ky.building_america_zone_source) == ("Mixed-Humid", "from 4A")
    tx = resolve_location(state="TX")  # no ASHRAE or Building America zone covers 90% of Texas homes
    assert tx.ashrae_zone is None and tx.building_america_zone is None and tx.census_division == "West South Central"
    fl = resolve_location(state="FL")  # 1A and 2A, both Hot-Humid
    assert fl.ashrae_zone is None and fl.building_america_zone == "Hot-Humid"
    assert fl.building_america_zone_source.startswith("from state (FL")


def test_building_america_zone_from_ashrae_zone(lookup):
    assert lookup.building_america_for("4A") == ("Mixed-Humid", 1.0)
    assert lookup.building_america_for("3A") is None  # split between Mixed-Humid and Hot-Humid
    assert lookup.zones_compatible("3A", "Hot-Humid") and not lookup.zones_compatible("5A", "Hot-Humid")
    assert resolve_location(climate_zone="3A").building_america_zone is None

    atlanta = resolve_location(climate_zone="3A", latitude=33.749, longitude=-84.388)
    assert atlanta.building_america_zone == "Mixed-Humid" and atlanta.building_america_zone_source.startswith("nearest")
    guarded = resolve_location(climate_zone="3A", latitude=DENVER[0], longitude=DENVER[1])
    assert guarded.state == "CO" and guarded.building_america_zone is None  # Denver's "Cold" never occurs with 3A
    assert any("inconsistent" in w for w in guarded.warnings)

    mismatch = resolve_location(climate_zone="5A", building_america_zone="hot-humid")
    assert mismatch.building_america_zone == "Hot-Humid" and "never occurs" in mismatch.warnings[0]


def test_puma_metro_status_is_only_explicit():
    assert resolve_location(state="DC", latitude=WHITE_HOUSE[0], longitude=WHITE_HOUSE[1]).puma_metro_status is None
    info = resolve_location(puma_metro_status="in metro area, principal city")
    assert (info.puma_metro_status, info.puma_metro_status_source) == ("In metro area, principal city", "explicit")
    assert info.values() == {PUMA_METRO_COLUMN: "In metro area, principal city"}


@pytest.mark.parametrize(
    "lat, lon, message",
    [
        (35.0, -70.0, "more than 150 km"),  # Atlantic, 500 km off Cape Hatteras
        (26.0, -90.0, "more than 150 km"),  # Gulf of Mexico
        (30.0, -60.0, "outside the contiguous United States"),
        (61.2181, -149.9003, "looks like Alaska"),
        (21.3069, -157.8583, "looks like Hawaii"),
    ],
)
def test_unusable_coordinates_are_not_used(lat, lon, message):
    info = resolve_location(latitude=lat, longitude=lon)
    assert info.values() == {} and (info.latitude, info.longitude) == (lat, lon)
    assert len(info.warnings) == 1 and message in info.warnings[0]


def test_alaska_is_kept_with_a_warning():
    info = resolve_location(state="AK", climate_zone="7ak")
    assert info.values() == {"State": "AK", ZONE: "7AK", DIVISION: "Pacific"}
    assert any("no data for AK" in w for w in info.warnings)
    assert resolve_location(weather_filename="USA_AK_Anchorage.Intl.AP.702730_TMY3.epw").state == "AK"


def test_colocated_station_records_are_pooled(lookup):
    nj = next(s for s in lookup.stations if s.name.startswith("New York Central Prk"))
    assert nj.dominant("State")[0] == "NJ"
    nearest, km = lookup.nearest_stations(nj.latitude, nj.longitude, k=1)[0]
    assert nearest is nj and km == pytest.approx(0.0, abs=1e-6)
    # pooled with the NY record of the same Central Park station 0.1 km away
    assert resolve_location(latitude=nj.latitude, longitude=nj.longitude).state == "NY"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"state": "Maryland"}, "invalid state"),
        ({"climate_zone": "9Z"}, "invalid climate zone"),
        ({"building_america_zone": "Tropical"}, "invalid Building America climate zone"),
        ({"puma_metro_status": "urban"}, "invalid PUMA metro status"),
        ({"latitude": 39.0}, "together"),
        ({"latitude": 95, "longitude": 0}, "invalid coordinates"),
    ],
)
def test_invalid_explicit_values_raise(kwargs, message):
    with pytest.raises(InputValidationError, match=message):
        resolve_location(**kwargs)


def test_labels_are_canonical_and_non_us_weather_files_are_ignored():
    info = resolve_location(weather_filename="CAN_ON_Toronto.epw")
    assert info.values() == {} and info.warnings == ()
    assert canonical_location_label("State", " md ") == "MD" and canonical_location_label(ZONE, "7ak") == "7AK"
    assert canonical_location_label(BA, "subarctic") == "Subarctic" and canonical_location_label("State", "ZZ") is None
    assert canonical_location_label(ZONE, None) is None and canonical_location_label(ZONE, float("nan")) is None


def test_fill_location_columns_per_row():
    df = pd.DataFrame(
        {
            "Building": ["balt", "dc", "blank", "bad", "ocean", "kept"],
            "Latitude": [39.29, 38.8977, None, "abc", 35.0, 39.29],
            "LONGITUDE": [-76.61, -77.0365, None, "x", -70.0, -76.61],
            ZONE: [None, None, None, None, None, "5A"],
        },
        dtype=object,
    )
    fill = fill_location_columns(df)
    out = fill.data
    assert fill.mode == "coordinates" and fill.coordinate_columns == ("Latitude", "LONGITUDE")
    assert "Latitude" not in out.columns and "LONGITUDE" not in out.columns
    assert out["State"].tolist() == ["MD", None, None, None, None, "MD"]
    assert out[ZONE].tolist() == ["4A", "4A", None, None, None, "5A"]  # the row's own 5A is kept ...
    assert out[BA].tolist()[5] == "Cold"  # ... and the Building America zone follows it
    assert fill.filled["State"] == 2 and fill.conflicts == {ZONE: 1}
    joined = "\n".join(fill.warnings)
    for fragment in ("keep their input value", "invalid latitude/longitude", "more than 150 km", "several states"):
        assert fragment in joined
    assert fill.to_json()["mode"] == "coordinates" and fill.describe()[0].startswith("Location: derived per row")


def test_fill_location_columns_explicit_options_never_overwrite():
    df = pd.DataFrame({"Building": ["a", "b"], "State": ["MD", None], "latitude": [29.76, 29.76],
                       "longitude": [-95.37, -95.37]}, dtype=object)
    fill = fill_location_columns(df, state="VA")
    assert fill.mode == "explicit" and "latitude" not in fill.data.columns
    assert fill.data["State"].tolist() == ["MD", "VA"]
    assert fill.data[ZONE].tolist() == ["4A", "4A"]  # MD's and VA's own zones
    assert fill.conflicts == {"State": 1}
    assert any("not used" in w for w in fill.warnings)

    plain = df.drop(columns=["latitude", "longitude"])
    untouched = fill_location_columns(plain)
    assert untouched.mode is None and untouched.data is plain and untouched.to_json() is None
    single = fill_location_columns(pd.DataFrame([{"Vintage": "1990s"}]), climate_zone="5B")
    assert single.describe() == ["Location: state unknown · 5B (explicit) · Cold (from 5B)"]
