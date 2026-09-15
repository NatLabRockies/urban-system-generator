import json

import pytest

from usg.errors import RegistryError
from usg.registry import Registry


def _entry(model_id, **extra):
    model_type, region = model_id.split("-")[:2]
    return {"model_id": model_id, "model_type": model_type, "region": region, "url": f"https://x/{model_id}.tar.gz",
            "sha256": "0" * 64, **extra}


def _raw(**overrides):
    raw = {
        "registry_format": "usg-registry/v1",
        "default_model_type": {"__default__": "gnn", "MD": "tabddpm"},
        "champions": {"MD": {"gnn": "gnn-MD-2024.2-v1", "tabddpm": "tabddpm-MD-2024.2-v2"},
                      "national": {"gnn": "gnn-national-2024.2-v1"}},
        "models": [
            _entry("gnn-MD-2024.2-v1"),
            _entry("tabddpm-MD-2024.2-v1", yanked=True),
            _entry("tabddpm-MD-2024.2-v2", headline={"accuracy": 0.8}),
            _entry("gnn-national-2024.2-v1", embedded=True),
        ],
    }
    raw.update(overrides)
    return raw


def test_packaged_registry_ships_national_models_for_every_state(monkeypatch):
    monkeypatch.delenv("USG_REGISTRY", raising=False)
    reg = Registry.load()
    assert {m.model_type for m in reg.models()} == {"gnn", "tabddpm"}
    assert reg.default_type("MD") == "gnn"
    for region in ("national", "MD", "CA", "DC", "WY"):
        for model_type in ("gnn", "tabddpm"):
            assert reg.champion(region, model_type).scope == "national"


def test_queries():
    reg = Registry(_raw())
    assert reg.get("tabddpm-MD-2024.2-v2").headline == {"accuracy": 0.8}
    assert reg.champion("md", "tabddpm").model_id == "tabddpm-MD-2024.2-v2"
    assert reg.champion("National", "gnn").embedded
    assert reg.champion("VA", "gnn") is None
    assert reg.default_type("MD") == "tabddpm" and reg.default_type("VA") == "gnn"
    assert {e.model_id for e in reg.models(region="MD", model_type="tabddpm")} == {
        "tabddpm-MD-2024.2-v1", "tabddpm-MD-2024.2-v2"
    }
    assert reg.is_champion("gnn-MD-2024.2-v1") and not reg.is_champion("tabddpm-MD-2024.2-v1")
    with pytest.raises(RegistryError):
        reg.get("gnn-VA-2024.2-v1")


def test_state_champion_may_be_the_national_model():
    raw = _raw()
    raw["champions"]["VA"] = {"gnn": "gnn-national-2024.2-v1"}  # no regional model beat the national one for VA
    reg = Registry(raw)
    assert reg.champion("VA", "gnn").model_id == "gnn-national-2024.2-v1"
    assert reg.champion("VA", "tabddpm") is None


def test_yanked_champion_is_not_returned():
    raw = _raw()
    raw["champions"]["MD"]["tabddpm"] = "tabddpm-MD-2024.2-v1"
    assert Registry(raw).champion("MD", "tabddpm") is None


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.update(registry_format="usg-registry/v0"), "registry_format"),
        (lambda r: r["models"].append(_entry("gnn-MD-2024.2-v1")), "duplicate"),
        (lambda r: r["champions"].update(VA={"gnn": "gnn-VA-2024.2-v1"}), "not listed"),
        (lambda r: r["champions"].update(VA={"gnn": "gnn-MD-2024.2-v1"}), "does not match"),
        (lambda r: r["models"].append({**_entry("gnn-VA-2024.2-v1"), "region": "MD"}), "does not match"),
        (lambda r: r["models"].append(_entry("gnn-Virginia-2024.2-v1")), "invalid model_id"),
        (lambda r: r["default_model_type"].update(VA="xgboost"), "not a model type"),
        (lambda r: r["champions"].update(ZZ={}), "unknown region"),
    ],
)
def test_invalid_registries(mutate, match):
    raw = _raw()
    mutate(raw)
    with pytest.raises(RegistryError, match=match):
        Registry(raw)


def test_load_from_path_env_and_urls(tmp_path, monkeypatch, http_server):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(_raw()))
    assert len(Registry.load(path)) == 4
    assert len(Registry.load(path.as_uri())) == 4
    monkeypatch.setenv("USG_REGISTRY", str(path))
    assert Registry.load().source == str(path)

    root, base = http_server
    (root / "reg.json").write_text(json.dumps(_raw()))
    assert len(Registry.load(f"{base}/reg.json")) == 4
    with pytest.raises(RegistryError, match="cannot fetch"):
        Registry.load(f"{base}/missing.json")
    with pytest.raises(RegistryError, match="not found"):
        Registry.load(tmp_path / "nope.json")
    assert RegistryError.exit_code == 3
