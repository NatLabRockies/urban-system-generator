import json

import pandas as pd
from click.testing import CliRunner

from usg.cli import cli
from usg.testing.fixtures import tiny_dataframe


def _world(tmp_path, bundle_factory, http_server, helpers):
    root, base = http_server
    cache = tmp_path / "cache"
    md = helpers.copy(bundle_factory("gnn", "MD"), cache)
    va = bundle_factory("gnn", "VA")
    nat = helpers.copy(bundle_factory("gnn", "national"), cache)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(helpers.registry(
        [helpers.entry(md, headline={"accuracy": 0.5}), helpers.publish(va, root, base), helpers.entry(nat)],
        {"MD": {"gnn": md.name}, "VA": {"gnn": va.name}, "national": {"gnn": nat.name}},
    )))
    env = {"USG_CACHE_DIR": str(cache), "USG_REGISTRY": str(registry), "USG_OFFLINE": ""}
    return env, cache


def _invoke(env, *args):
    return CliRunner().invoke(cli, [str(a) for a in args], env=env)


def test_models_lifecycle(tmp_path, bundle_factory, http_server, helpers):
    env, cache = _world(tmp_path, bundle_factory, http_server, helpers)

    listing = _invoke(env, "models", "list")
    assert listing.exit_code == 0, listing.output
    assert "* gnn-MD-2024.2-v1" in listing.output and "accuracy=0.5" in listing.output
    rows = {r["model_id"]: r for r in json.loads(_invoke(env, "models", "list", "--json").output)["models"]}
    assert rows["gnn-VA-2024.2-v1"]["status"] == "remote" and rows["gnn-MD-2024.2-v1"]["status"] == "cached"
    assert "gnn-VA" not in _invoke(env, "models", "list", "--state", "MD").output

    offline = _invoke({**env, "USG_OFFLINE": "1"}, "models", "download", "--state", "VA")
    assert offline.exit_code == 3 and "offline" in offline.output
    downloaded = _invoke(env, "models", "download", "--state", "VA")
    assert downloaded.exit_code == 0, downloaded.output
    assert (cache / "gnn-VA-2024.2-v1" / "manifest.json").exists()
    assert _invoke(env, "models", "download").exit_code == 2

    path = _invoke(env, "models", "path", "gnn-VA-2024.2-v1")
    assert path.exit_code == 0 and path.output.strip() == str(cache / "gnn-VA-2024.2-v1")
    assert _invoke(env, "models", "path").output.strip() == str(cache)
    assert _invoke(env, "models", "path", "gnn-TX-2024.2-v1").exit_code == 3

    info = _invoke(env, "models", "info", "gnn-MD-2024.2-v1", "--card")
    assert info.exit_code == 0 and "Local status: cached" in info.output and "# gnn-MD-2024.2-v1" in info.output

    verify = _invoke(env, "models", "verify", "gnn-MD-2024.2-v1", "--full", "--selftest")
    assert verify.exit_code == 0, verify.output
    assert "self-test passed" in verify.output
    with open(cache / "gnn-MD-2024.2-v1" / "metrics.json", "a") as f:
        f.write(" ")
    corrupted = _invoke(env, "models", "verify", str(cache / "gnn-MD-2024.2-v1"), "--full")
    assert corrupted.exit_code == 4 and "checksum" in corrupted.output

    assert _invoke(env, "models", "clear-cache", "gnn-VA-2024.2-v1", "--yes").exit_code == 0
    assert not (cache / "gnn-VA-2024.2-v1").exists()
    cleared = _invoke(env, "models", "clear-cache", "--yes")
    assert cleared.exit_code == 0 and "removed 2" in cleared.output


def test_complete_partitions_mixed_state_csv(tmp_path, bundle_factory, http_server, helpers):
    env, _ = _world(tmp_path, bundle_factory, http_server, helpers)
    df = tiny_dataframe(12, seed=3).drop(columns=["Heating Fuel"])
    df["State"] = ["MD", "VA", "DC"] * 4
    df.insert(0, "Building", range(12))
    src = tmp_path / "mixed.csv"
    df.to_csv(src, index=False)
    out = tmp_path / "out.csv"
    result = _invoke(env, "complete", "-i", src, "-o", out)
    assert result.exit_code == 0, result.output
    for model_id in ("gnn-MD-2024.2-v1", "gnn-VA-2024.2-v1", "gnn-national-2024.2-v1"):
        assert f"Model: {model_id}" in result.output
    assert "no gnn model is published for DC" in result.output
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert {m["model_id"]: m["n_rows"] for m in manifest["models"]} == {
        "gnn-MD-2024.2-v1": 4, "gnn-VA-2024.2-v1": 4, "gnn-national-2024.2-v1": 4
    }
    assert pd.read_csv(out)["Heating Fuel"].notna().all()

    single = _invoke(env, "complete", "-i", src, "-o", out, "--state", "MD", "--no-fallback")
    assert single.exit_code == 0 and "gnn-VA" not in single.output
