"""Session fixtures: tiny model bundles built at test time, a local HTTP server, and registry helpers."""

from __future__ import annotations

import functools
import http.server
import shutil
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_GEOJSON = REPO_ROOT / "examples" / "example_geojson.json"
COMBINED_GEOJSON = REPO_ROOT / "examples" / "example_project_combined.json"


@pytest.fixture(autouse=True)
def _no_embedded_bundles(tmp_path_factory, monkeypatch):
    """Tests never see bundles embedded in a local checkout (they are gitignored, so CI has none)."""
    monkeypatch.setenv("USG_EMBEDDED_DIR", str(tmp_path_factory.mktemp("embedded")))


def build_tiny_bundle(out_dir: Path, model_type: str = "gnn", region: str = "MD", version: int = 1) -> Path:
    import torch

    from usg.bundle import write_bundle
    from usg.models import build_model
    from usg.testing.fixtures import tiny_config, tiny_dataframe, tiny_schema

    torch.manual_seed(0)
    schema = tiny_schema()
    config = tiny_config(model_type)
    model_id = f"{model_type}-{region}-2024.2-v{version}"
    national = region == "national"
    return write_bundle(
        Path(out_dir) / model_id,
        model=build_model(schema, config),
        schema=schema,
        config=config,
        manifest={
            "model_id": model_id,
            "region": {
                "scope": "national" if national else "state",
                "target_state": None if national else region,
                "training_states": ["MD", "VA", "DC"],
            },
        },
        metrics={"schema": "usg-metrics/v1"},
        model_card=f"# {model_id}\n",
        selftest_inputs=tiny_dataframe(4, seed=5).drop(columns=["Vintage", "sqft"]),
    )


@pytest.fixture(scope="session")
def bundle_factory(tmp_path_factory):
    """``make(model_type, region, version)`` -> path of a (cached) tiny bundle. Copy before modifying it."""
    root = tmp_path_factory.mktemp("bundles")
    built: dict[tuple, Path] = {}

    def make(model_type: str = "gnn", region: str = "MD", version: int = 1) -> Path:
        key = (model_type, region, version)
        if key not in built:
            built[key] = build_tiny_bundle(root, *key)
        return built[key]

    return make


@pytest.fixture(scope="session")
def gnn_bundle(bundle_factory) -> Path:
    return bundle_factory("gnn", "MD")


@pytest.fixture(scope="session")
def tabddpm_bundle(bundle_factory) -> Path:
    return bundle_factory("tabddpm", "MD")


@pytest.fixture(scope="session")
def gnn_imputer(gnn_bundle):
    from usg.bundle import Bundle

    return Bundle.open(gnn_bundle).load_imputer()


@pytest.fixture(scope="session")
def tabddpm_imputer(tabddpm_bundle):
    from usg.bundle import Bundle

    return Bundle.open(tabddpm_bundle).load_imputer()


@pytest.fixture(scope="session")
def http_server(tmp_path_factory):
    """``(root_dir, base_url)`` of a local static file server."""
    root = tmp_path_factory.mktemp("http")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def registry_entry(bundle_path: Path, url: str | None = None, sha256: str | None = None, **extra) -> dict:
    from usg.bundle.manifest import parse_model_id, sha256_file

    model_type, region, _, _ = parse_model_id(bundle_path.name)
    return {
        "model_id": bundle_path.name,
        "model_type": model_type,
        "region": region,
        "url": url,
        "sha256": sha256,
        "manifest_sha256": sha256_file(bundle_path / "manifest.json"),
        **extra,
    }


def registry_json(models: list[dict], champions: dict | None = None, default_type: dict | None = None) -> dict:
    return {
        "registry_format": "usg-registry/v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "data_release": "test",
        "default_model_type": default_type or {"__default__": "gnn"},
        "champions": champions or {},
        "models": models,
    }


def publish(bundle_path: Path, http_root: Path, base_url: str) -> dict:
    """Pack a bundle into the HTTP root and return its registry entry."""
    from usg.bundle import pack_bundle

    archive = http_root / f"{bundle_path.name}.tar.gz"
    sha = pack_bundle(bundle_path, archive)
    return registry_entry(bundle_path, url=f"{base_url}/{archive.name}", sha256=sha, size_bytes=archive.stat().st_size)


def copy_bundle(bundle_path: Path, dest_parent: Path) -> Path:
    dest_parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copytree(bundle_path, dest_parent / bundle_path.name))


@pytest.fixture
def helpers():
    """Namespace of plain helper functions for registries and bundles."""

    class Helpers:
        entry = staticmethod(registry_entry)
        registry = staticmethod(registry_json)
        publish = staticmethod(publish)
        copy = staticmethod(copy_bundle)
        example_geojson = EXAMPLE_GEOJSON
        combined_geojson = COMBINED_GEOJSON

    return Helpers
