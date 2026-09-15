import io
import json
import tarfile

import numpy as np
import pytest
import torch

from usg.bundle import Bundle, extract_bundle, pack_bundle, write_bundle
from usg.errors import BundleError, CompatibilityError
from usg.models import build_model
from usg.testing.fixtures import tiny_config, tiny_dataframe, tiny_schema


def _write(tmp_path, model_type, model_id=None, fp16=True):
    torch.manual_seed(0)
    schema = tiny_schema()
    config = tiny_config(model_type)
    net = build_model(schema, config)
    inputs = tiny_dataframe(8, seed=5).drop(columns=["Vintage", "HVAC Heating Type", "sqft"])
    model_id = model_id or f"{model_type}-MD-2024.2-v1"
    path = write_bundle(
        tmp_path / model_id,
        model=net,
        schema=schema,
        config=config,
        manifest={"model_id": model_id, "region": {"scope": "state", "target_state": "MD", "training_states": ["MD", "VA", "DC"]}},
        metrics={"schema": "usg-metrics/v1"},
        model_card=f"# {model_id}\n",
        selftest_inputs=inputs,
        fp16=fp16,
    )
    return path, net, schema


@pytest.mark.parametrize("model_type", ["gnn", "tabddpm"])
def test_write_open_selftest_and_impute(tmp_path, model_type):
    path, _, _ = _write(tmp_path, model_type)
    bundle = Bundle.open(path, verify="full")
    assert (path / ".verified").exists()
    assert bundle.manifest.target_state == "MD"
    assert bundle.run_selftest() == []
    imputer = bundle.load_imputer()
    df = tiny_dataframe(20, seed=9).drop(columns=["Heating Fuel", "Geometry Garage"])
    res = imputer.impute(df, seed=1)
    assert res.values["Heating Fuel"].notna().all()
    assert (res.values["State"].to_numpy() == df["State"].to_numpy()).all()
    assert res.imputed["Heating Fuel"].all() and not res.imputed["Vintage"].any()
    assert res.probabilities["Heating Fuel"].shape == (20, 5)
    np.testing.assert_allclose(res.values["sqft"], df["sqft"])  # known numeric passes through exactly
    assert len(res.top_k("Heating Fuel", 2)[0]) == 2


def test_coerced_input_keeps_user_value_in_output(tmp_path):
    path, _, _ = _write(tmp_path, "gnn")
    imputer = Bundle.open(path).load_imputer()
    df = tiny_dataframe(4, seed=1).drop(columns=["Heating Fuel"])
    df["Bedrooms"] = "7"
    res = imputer.impute(df)
    assert (res.values["Bedrooms"] == "7").all()
    assert not res.imputed["Bedrooms"].any()
    assert {i.kind for i in res.issues} == {"coerced"}


def test_same_seed_is_deterministic(tmp_path):
    path, _, _ = _write(tmp_path, "tabddpm")
    imputer = Bundle.open(path).load_imputer()
    df = tiny_dataframe(10, seed=2).drop(columns=["Heating Fuel", "Bedrooms"])
    a = imputer.impute(df, seed=3, n_samples=4)
    b = imputer.impute(df, seed=3, n_samples=4)
    assert a.values.equals(b.values)


@pytest.mark.parametrize("fp16, atol", [(False, 1e-5), (True, 5e-3)])
def test_loaded_network_matches_live_model(tmp_path, fp16, atol):
    path, live, schema = _write(tmp_path, "gnn", fp16=fp16)
    loaded = Bundle.open(path).load_network()
    batch, _ = schema.encode(tiny_dataframe(16, seed=4).drop(columns=["Vintage"]))
    args = [torch.as_tensor(a) for a in (batch.cat, batch.cat_known, batch.num, batch.num_known)]
    live.eval()
    with torch.no_grad():
        a = live(*args)["cat_logits"]
        b = loaded(*args)["cat_logits"]
    valid = a > -1e8
    torch.testing.assert_close(a[valid], b[valid], atol=atol, rtol=0)


def test_corrupted_file_is_detected(tmp_path):
    path, _, _ = _write(tmp_path, "gnn")
    Bundle.open(path, verify="full")
    with open(path / "metrics.json", "a") as f:
        f.write(" ")
    with pytest.raises(BundleError, match="changed since verification|checksum"):
        Bundle.open(path, verify="fast")
    with pytest.raises(BundleError, match="checksum"):
        Bundle.open(path, verify="full")


def test_tampered_schema_is_rejected(tmp_path):
    path, _, _ = _write(tmp_path, "gnn")
    schema = json.loads((path / "schema.json").read_text())
    schema["columns"][1]["vocab"].append("Mobile Home")
    (path / "schema.json").write_text(json.dumps(schema))
    with pytest.raises(BundleError, match="schema_hash"):
        Bundle.open(path, verify="none")


def test_incompatible_requirements(tmp_path):
    path, _, _ = _write(tmp_path, "gnn")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["requires"] = {"usg": ">=99"}
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CompatibilityError):
        Bundle.open(path, verify="none")
    manifest["requires"] = {}
    manifest["arch_version"] = 42
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CompatibilityError):
        Bundle.open(path, verify="none")


def test_invalid_model_id_rejected(tmp_path):
    with pytest.raises(BundleError, match="invalid model_id"):
        _write(tmp_path, "gnn", model_id="gnn-Maryland-2024.2-v1")


def test_archive_roundtrip_is_deterministic(tmp_path):
    path, _, _ = _write(tmp_path, "gnn")
    sha1 = pack_bundle(path, tmp_path / "a.tar.gz")
    sha2 = pack_bundle(path, tmp_path / "b.tar.gz")
    assert sha1 == sha2
    extracted = extract_bundle(tmp_path / "a.tar.gz", tmp_path / "cache", expected_name=path.name)
    assert Bundle.open(extracted, verify="full").run_selftest() == []


def test_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(BundleError, match="unsafe"):
        extract_bundle(archive, tmp_path / "cache")
    assert not (tmp_path / "evil.txt").exists()
