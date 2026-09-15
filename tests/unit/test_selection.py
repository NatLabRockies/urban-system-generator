import pandas as pd
import pytest

from usg.download import Cache
from usg.errors import DownloadError, InputValidationError, SelectionError
from usg.geojson import ProjectContext
from usg.registry import Registry
from usg.selection import ModelRequest, resolve, resolve_partitioned, state_key

MD_CTX = ProjectContext(state="MD", state_source="project weather file (USA_MD_x.epw)", climate_zone="4A")


@pytest.fixture
def world(bundle_factory, http_server, helpers, tmp_path):
    """Embedded national G-NN, cached gnn-MD, downloadable tabddpm-MD, tabddpm-VA with a broken URL."""
    root, base = http_server
    gnn_md = bundle_factory("gnn", "MD")
    tab_md = bundle_factory("tabddpm", "MD")
    gnn_nat = bundle_factory("gnn", "national")
    embedded = tmp_path / "embedded"
    helpers.copy(gnn_nat, embedded)
    cache = Cache(tmp_path / "cache")
    helpers.copy(gnn_md, cache.dir)
    models = [
        helpers.entry(gnn_md, url=f"{base}/unused.tar.gz", sha256="0" * 64),
        helpers.publish(tab_md, root, base),
        helpers.entry(gnn_nat, embedded=True),
        {**helpers.entry(tab_md), "model_id": "tabddpm-VA-2024.2-v1", "region": "VA",
         "url": f"{base}/missing.tar.gz", "sha256": "0" * 64},
    ]
    champions = {
        "MD": {"gnn": gnn_md.name, "tabddpm": tab_md.name},
        "VA": {"tabddpm": "tabddpm-VA-2024.2-v1"},
        "TX": {"gnn": gnn_nat.name},  # the national model is TX's recommended G-NN
        "national": {"gnn": gnn_nat.name},
    }
    registry = Registry(helpers.registry(models, champions, {"__default__": "gnn", "VA": "tabddpm"}))
    return {"registry": registry, "cache": cache, "embedded": embedded, "gnn_md": gnn_md}


def _resolve(world, **kwargs):
    return resolve(ModelRequest(**kwargs), world["registry"], world["cache"], embedded_dir=world["embedded"])


@pytest.mark.parametrize(
    "request_kwargs, model_id, region, source, n_warnings",
    [
        ({"model_id": "gnn-MD-2024.2-v1"}, "gnn-MD-2024.2-v1", "MD", "cache", 0),
        ({"model_id": "gnn-national-2024.2-v1"}, "gnn-national-2024.2-v1", "national", "embedded", 0),
        ({"state": "MD"}, "gnn-MD-2024.2-v1", "MD", "cache", 0),
        ({"state": "md", "model_type": "gnn"}, "gnn-MD-2024.2-v1", "MD", "cache", 0),
        ({"state": "auto", "context": MD_CTX}, "gnn-MD-2024.2-v1", "MD", "cache", 0),
        ({"state": "auto"}, "gnn-national-2024.2-v1", "national", "embedded", 0),
        ({"state": "national"}, "gnn-national-2024.2-v1", "national", "embedded", 0),
        ({"state": "MD", "model_type": "tabddpm"}, "tabddpm-MD-2024.2-v1", "MD", "download", 0),
        ({"state": "AK"}, "gnn-national-2024.2-v1", "national", "embedded", 1),
        ({"state": "VA"}, "gnn-national-2024.2-v1", "national", "embedded", 1),  # default tabddpm, download 404
        ({"state": "VA", "offline": True}, "gnn-national-2024.2-v1", "national", "embedded", 1),
        ({"state": "AK", "model_type": "tabddpm"}, "gnn-national-2024.2-v1", "national", "embedded", 2),
        ({"state": "TX"}, "gnn-national-2024.2-v1", "national", "embedded", 0),  # national is TX's champion
    ],
)
def test_resolution_table(world, request_kwargs, model_id, region, source, n_warnings):
    resolved = _resolve(world, **request_kwargs)
    assert (resolved.model_id, resolved.region, resolved.source) == (model_id, region, source)
    assert len(resolved.warnings) == n_warnings, resolved.warnings
    assert (resolved.bundle_path / "manifest.json").is_file()
    assert resolved.reason


def test_reason_mentions_project_state(world):
    resolved = _resolve(world, context=MD_CTX)
    assert "USA_MD" in resolved.reason and "gnn" in resolved.reason


def test_national_champion_for_a_state_is_a_recommendation_not_a_fallback(world):
    resolved = _resolve(world, state="TX")
    assert resolved.warnings == []
    assert "recommended model for TX" in resolved.reason and "fell back" not in resolved.reason


def test_bundle_dir_wins_and_warns_on_conflicts(world):
    resolved = _resolve(world, bundle_dir=world["gnn_md"], state="VA", model_type="tabddpm", model_id="gnn-x")
    assert resolved.source == "bundle_dir" and resolved.model_id == "gnn-MD-2024.2-v1"
    assert len(resolved.warnings) == 2


@pytest.mark.parametrize(
    "request_kwargs, error",
    [
        ({"state": "AK", "allow_fallback": False}, SelectionError),
        ({"state": "VA", "allow_fallback": False}, DownloadError),
        ({"state": "ZZ"}, InputValidationError),
        ({"model_type": "xgboost"}, InputValidationError),
        ({"model_id": "gnn-TX-2024.2-v1"}, SelectionError),
    ],
)
def test_resolution_errors(world, request_kwargs, error):
    with pytest.raises(error):
        _resolve(world, **request_kwargs)


def test_empty_registry_without_embedded_model(tmp_path, helpers):
    with pytest.raises(SelectionError, match="no gnn model"):
        resolve(ModelRequest(), Registry(helpers.registry([])), Cache(tmp_path / "c"), embedded_dir=tmp_path / "none")


def test_empty_registry_falls_back_to_embedded(world, helpers):
    resolved = resolve(ModelRequest(state="MD"), Registry(helpers.registry([])), world["cache"],
                       embedded_dir=world["embedded"])
    assert resolved.model_id == "gnn-national-2024.2-v1" and resolved.warnings


def test_state_key():
    assert state_key("md") == "MD" and state_key(" VA ") == "VA"
    assert state_key(None) is None and state_key("") is None and state_key("Maryland") is None


def test_resolve_partitioned(world):
    states = pd.Series(["MD", "AK", None, "md", "Maryland"])
    out = resolve_partitioned(ModelRequest(), states, world["registry"], world["cache"], embedded_dir=world["embedded"])
    assert set(out) == {"MD", "AK", None}
    assert out["MD"].model_id == "gnn-MD-2024.2-v1"
    assert out["AK"].model_id == out[None].model_id == "gnn-national-2024.2-v1"

    fixed = resolve_partitioned(ModelRequest(state="MD"), states, world["registry"], world["cache"],
                                embedded_dir=world["embedded"])
    assert {r.model_id for r in fixed.values()} == {"gnn-MD-2024.2-v1"}
