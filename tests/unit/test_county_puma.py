import pandas as pd
import pytest

from usg.county_puma import CountyPumaLookup
from usg.geojson import ProjectContext, read_project_context, with_county_and_puma
from usg.postprocessor import USGPostProcessor


@pytest.fixture(scope="module")
def lookup():
    return CountyPumaLookup.load()


def test_packaged_table(lookup):
    assert lookup.meta["format"] == "usg-county-puma-tracts/v1"
    assert lookup.meta["n_tracts"] > 70_000 and lookup.meta["n_county_puma"] > 4_400


@pytest.mark.parametrize(("lat", "lon", "state", "expected"), [
    (39.3021, -76.6057, "MD", "G2405100, G24000804"),  # Baltimore example Site Origin
    (42.8168, -78.8495, "NY", "G3600290, G36001204"),  # Buffalo example Site Origin
])
def test_nearest_tract(lookup, lat, lon, state, expected):
    match = lookup.nearest(lat, lon, state=state)
    assert match.county_and_puma == expected and match.state == state and match.distance_km < 2


def test_far_points_and_other_states_are_not_matched(lookup):
    assert lookup.nearest(38.0, -70.0) is None  # Atlantic Ocean
    assert lookup.nearest(39.3021, -76.6057, state="OH") is None


def test_project_context_from_the_site_origin(helpers):
    context = read_project_context(helpers.example_geojson)
    assert context.county_and_puma == "G2405100, G24000804"
    assert context.county_and_puma_source.startswith("Site Origin, nearest Census tract")
    assert "County and PUMA G2405100, G24000804" in context.summary()
    assert context.location_json()["county_and_puma"] == "G2405100, G24000804"
    assert with_county_and_puma(ProjectContext(state="MD")).county_and_puma is None  # no coordinates


def test_post_processing_derives_the_location_from_it():
    table = {
        "format": "usg-postprocess-derivations/v1", "derive_purity": 0.99, "order": ["County and PUMA", "County"],
        "columns": {
            "County and PUMA": {"kind": "anchor", "mode": None, "levels": []},
            "County": {"kind": "derived", "mode": None, "levels": [
                {"from": ["County and PUMA"], "purity": 1.0, "values": {"G2405100, G24000804": "MD, Baltimore city"}},
            ]},
        },
    }
    processor = USGPostProcessor(verbose=False, derivations=table)
    context = ProjectContext(state="MD", county_and_puma="G2405100, G24000804")
    out, _, _ = processor.process_dataframe(pd.DataFrame({"Building": ["a"]}), context)
    assert out.at[0, "County and PUMA"] == "G2405100, G24000804" and out.at[0, "County"] == "MD, Baltimore city"
    assert processor.last_fill_log["County and PUMA"] == {"project location": 1}
    assert processor.last_fill_log["County"] == {"derived from County and PUMA": 1}
