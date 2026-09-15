"""Write a model bundle (used by the model exporter and by test fixtures)."""

from __future__ import annotations

import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from usg.__about__ import __version__
from usg.bundle.manifest import (
    CALIBRATION,
    CONFIG,
    DERIVATIONS,
    MANIFEST,
    METRICS,
    MODEL_CARD,
    SCHEMA,
    SELFTEST,
    SENSITIVITY,
    WEIGHTS,
    parse_model_id,
    sha256_file,
    write_json,
)
from usg.constants import BUNDLE_FORMAT
from usg.errors import BundleError, CompatibilityError
from usg.schema import Schema


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                clean[k] = None
            elif isinstance(v, (np.floating, float)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            else:
                clean[k] = v if isinstance(v, (int, str, bool)) else str(v)
        out.append(clean)
    return out


def write_bundle(
    out_dir: str | Path,
    *,
    model,
    schema: Schema,
    config: dict,
    manifest: dict,
    metrics: dict,
    model_card: str,
    selftest_inputs: pd.DataFrame,
    calibration: dict | None = None,
    sensitivity: dict | None = None,
    derivations: dict | None = None,
    fp16: bool = True,
    overwrite: bool = False,
    selftest_samples: int | None = None,
) -> Path:
    """Serialize ``model`` + metadata into ``out_dir`` atomically and record a self-test.

    ``manifest`` must contain at least ``model_id``; format, hashes, file checksums, timestamps and package
    versions are filled in here.
    """
    import torch
    from safetensors.torch import save_file

    from usg.bundle.reader import Bundle
    from usg.models import SUPPORTED_ARCH

    out_dir = Path(out_dir)
    model_id = manifest.get("model_id", "")
    model_type, _, _, _ = parse_model_id(model_id)
    if config.get("model_type") != model_type:
        raise BundleError(f"config model_type {config.get('model_type')!r} does not match model_id {model_id!r}")
    arch_version = int(config.get("arch_version", 1))
    if arch_version not in SUPPORTED_ARCH.get(model_type, ()):
        raise CompatibilityError(f"{model_type} arch_version {arch_version} is not a released architecture")
    if out_dir.exists() and not overwrite:
        raise BundleError(f"bundle directory already exists: {out_dir}")

    tmp = out_dir.parent / f".{out_dir.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        write_json(tmp / SCHEMA, schema.to_json())
        write_json(tmp / CONFIG, config)
        write_json(tmp / METRICS, metrics)
        (tmp / MODEL_CARD).write_text(model_card, encoding="utf-8")
        if calibration is not None:
            write_json(tmp / CALIBRATION, calibration)
        if sensitivity is not None:
            write_json(tmp / SENSITIVITY, sensitivity)
        if derivations is not None:
            write_json(tmp / DERIVATIONS, derivations)

        state = {}
        for key, tensor in model.state_dict().items():
            t = tensor.detach().to("cpu")
            if fp16 and t.is_floating_point():
                t = t.half()
            state[key] = t.contiguous()
        schema_hash = schema.schema_hash()
        save_file(
            state,
            str(tmp / WEIGHTS),
            metadata={"model_id": model_id, "schema_hash": schema_hash, "bundle_format": BUNDLE_FORMAT},
        )

        man = dict(manifest)
        man["bundle_format"] = BUNDLE_FORMAT
        man["model_type"] = model_type
        man["arch_version"] = arch_version
        man["schema_hash"] = schema_hash
        man["vocab_hash"] = schema.vocab_hash()
        man.setdefault("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        created_by = dict(man.get("created_by") or {})
        created_by.setdefault("usg_version", __version__)
        created_by.setdefault("torch_version", torch.__version__)
        created_by.setdefault("python", platform.python_version())
        man["created_by"] = created_by
        man.setdefault("requires", {"usg": f">={__version__}"})
        man["weights_dtype"] = "float16" if fp16 else "float32"
        man["files"] = {}
        write_json(tmp / MANIFEST, man)

        # Self-test is recorded from the bundle as written (fp16 weights upcast), exactly as users load it.
        bundle = Bundle.open(tmp, verify="none")
        imputer = bundle.load_imputer(device="cpu", threads=None)
        n_samples = selftest_samples or config.get("inference", {}).get("default_samples")
        inputs = selftest_inputs.reset_index(drop=True).astype(object)
        result = imputer.impute(inputs, n_samples=n_samples, seed=0)
        imputable = [c for c in schema.imputable_names]
        write_json(
            tmp / SELFTEST,
            {
                "inputs": _records(inputs.where(inputs.notna(), None)),
                "seed": 0,
                "n_samples": result.n_samples,
                "torch_version": torch.__version__,
                "tolerance": 1e-4,
                "expected": {
                    "values": _records(result.values[imputable].reset_index(drop=True)),
                    "probabilities": {c: np.round(p, 6).tolist() for c, p in result.probabilities.items()},
                },
            },
        )

        files = {}
        for f in sorted(tmp.iterdir()):
            if f.name in (MANIFEST,) or f.name.startswith("."):
                continue
            files[f.name] = {"sha256": sha256_file(f), "size": f.stat().st_size}
        man["files"] = files
        write_json(tmp / MANIFEST, man)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir
