"""Open and verify model bundles; load their imputer."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
    REQUIRED_FILES,
    SCHEMA,
    SELFTEST,
    SENSITIVITY,
    WEIGHTS,
    Manifest,
    read_json,
    read_safetensors_metadata,
    sha256_file,
)
from usg.errors import BundleError, CompatibilityError
from usg.schema import Schema

if TYPE_CHECKING:
    from usg.backends.base import TorchImputer

VERIFIED_MARKER = ".verified"
VerifyMode = Literal["full", "fast", "none"]


class Bundle:
    def __init__(self, path: Path, manifest: Manifest, schema: Schema, config: dict):
        self.path = path
        self.manifest = manifest
        self.schema = schema
        self.config = config
        self.warnings: list[str] = []

    def __repr__(self) -> str:
        return f"Bundle({self.manifest.model_id!r}, path={str(self.path)!r})"

    @property
    def model_id(self) -> str:
        return self.manifest.model_id

    @property
    def model_type(self) -> str:
        return self.manifest.model_type

    # ---- opening -------------------------------------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path, *, verify: VerifyMode = "fast") -> Bundle:
        path = Path(path)
        if not path.is_dir():
            raise BundleError(f"bundle directory not found: {path}")
        manifest = Manifest.from_json(read_json(path / MANIFEST))

        from usg.models import SUPPORTED_ARCH

        supported = SUPPORTED_ARCH.get(manifest.model_type)
        if supported is None or manifest.arch_version not in supported:
            raise CompatibilityError(
                f"{manifest.model_id}: {manifest.model_type} arch_version {manifest.arch_version} is not supported "
                f"by urban-system-generator {__version__}; upgrade the package"
            )
        if manifest.requires:
            from packaging.specifiers import InvalidSpecifier, SpecifierSet

            try:
                ok = SpecifierSet(manifest.requires).contains(__version__, prereleases=True)
            except InvalidSpecifier:
                raise BundleError(f"invalid requires specifier {manifest.requires!r}") from None
            if not ok:
                raise CompatibilityError(
                    f"{manifest.model_id} requires urban-system-generator {manifest.requires}, installed {__version__}"
                )

        bundle_warnings: list[str] = []
        if verify != "none":
            cls._verify_files(path, manifest, verify)

        schema = Schema.from_json(read_json(path / SCHEMA))
        if schema.schema_hash() != manifest.schema_hash:
            raise BundleError(f"{manifest.model_id}: schema.json does not match manifest schema_hash")
        config = read_json(path / CONFIG)
        if config.get("model_type") != manifest.model_type:
            raise BundleError(f"{manifest.model_id}: config.json model_type does not match manifest")

        if verify != "none" or (path / WEIGHTS).exists():
            meta = read_safetensors_metadata(path / WEIGHTS)
            if meta.get("model_id") != manifest.model_id or meta.get("schema_hash") != manifest.schema_hash:
                raise BundleError(f"{manifest.model_id}: weights metadata does not match manifest")

        bundle = cls(path, manifest, schema, config)
        bundle.warnings = bundle_warnings
        return bundle

    @staticmethod
    def _verify_files(path: Path, manifest: Manifest, mode: VerifyMode) -> None:
        missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            raise BundleError(f"{manifest.model_id}: missing required files {missing}")
        unlisted = [name for name in REQUIRED_FILES if name not in manifest.files]
        if unlisted:
            raise BundleError(f"{manifest.model_id}: manifest does not list {unlisted}")
        marker = path / VERIFIED_MARKER
        manifest_sha = sha256_file(path / MANIFEST)
        if mode == "fast" and marker.is_file():
            try:
                if json.loads(marker.read_text()).get("manifest_sha256") == manifest_sha:
                    for name, info in manifest.files.items():
                        f = path / name
                        if not f.is_file() or f.stat().st_size != int(info["size"]):
                            raise BundleError(f"{manifest.model_id}: {name} changed since verification")
                    return
            except (json.JSONDecodeError, OSError):
                pass
        for name, info in manifest.files.items():
            f = path / name
            if not f.is_file():
                raise BundleError(f"{manifest.model_id}: listed file {name} is missing")
            if sha256_file(f) != info["sha256"]:
                raise BundleError(f"{manifest.model_id}: checksum mismatch for {name} (corrupted or modified)")
        try:
            marker.write_text(json.dumps({"manifest_sha256": manifest_sha}))
        except OSError:  # read-only location (e.g. inside an installed wheel)
            pass

    # ---- auxiliary files -----------------------------------------------------------------------------
    def _optional_json(self, name: str) -> dict | None:
        f = self.path / name
        return read_json(f) if f.is_file() else None

    def metrics(self) -> dict:
        return self._optional_json(METRICS) or {}

    def calibration(self) -> dict | None:
        return self._optional_json(CALIBRATION)

    def sensitivity(self) -> dict | None:
        return self._optional_json(SENSITIVITY)

    def derivations(self) -> dict | None:
        """Post-processing fill table (``usg-postprocess-derivations/v1``), if the bundle ships one."""
        return self._optional_json(DERIVATIONS)

    def selftest(self) -> dict:
        return read_json(self.path / SELFTEST)

    def model_card(self) -> str:
        f = self.path / MODEL_CARD
        return f.read_text(encoding="utf-8") if f.is_file() else ""

    # ---- model loading -------------------------------------------------------------------------------
    def load_network(self, device: str = "cpu", threads: int | None = 1):
        from usg._torch import get_torch

        get_torch(threads)
        from safetensors.torch import load_file

        from usg.models import build_model

        net = build_model(self.schema, self.config)
        state = load_file(str(self.path / WEIGHTS), device="cpu")
        state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
        try:
            net.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise BundleError(f"{self.model_id}: weights do not match the architecture: {exc}") from None
        return net.eval().to(device)

    def load_imputer(self, device: str = "cpu", threads: int | None = 1) -> TorchImputer:
        from usg._torch import resolve_device
        from usg.backends import get_imputer_cls

        net = self.load_network(device=resolve_device(device) if device == "auto" else device, threads=threads)
        return get_imputer_cls(self.model_type)(self, net)

    # ---- self-test -----------------------------------------------------------------------------------
    def run_selftest(self, imputer: TorchImputer | None = None) -> list[str]:
        """Re-run the recorded self-test inputs; returns a list of problems (empty = pass)."""
        import torch

        st = self.selftest()
        imputer = imputer or self.load_imputer()
        inputs = pd.DataFrame(st["inputs"], dtype=object)
        result = imputer.impute(inputs, n_samples=st.get("n_samples"), seed=st.get("seed", 0))
        tol = float(st.get("tolerance", 1e-4))
        strict = self.model_type == "gnn" or st.get("torch_version") == torch.__version__
        problems: list[str] = []
        expected_values = pd.DataFrame(st["expected"]["values"], dtype=object)
        for col in expected_values.columns:
            spec = self.schema.column(col)
            got = result.values[col].reset_index(drop=True)
            exp = expected_values[col]
            if spec.type == "numeric":
                g = pd.to_numeric(got, errors="coerce").to_numpy(dtype=float)
                e = pd.to_numeric(exp, errors="coerce").to_numpy(dtype=float)
                scale = np.maximum(np.abs(e), 1.0)
                if strict and not np.allclose(g, e, atol=0, rtol=0, equal_nan=True) and np.nanmax(np.abs(g - e) / scale) > 1e-3:
                    problems.append(f"numeric column {col!r} differs from self-test expectation")
            elif strict and [None if pd.isna(v) else str(v) for v in got] != [None if pd.isna(v) else str(v) for v in exp]:
                problems.append(f"column {col!r} values differ from self-test expectation")
        for col, exp in st["expected"].get("probabilities", {}).items():
            got = result.probabilities[col]
            exp_arr = np.asarray(exp, dtype=float)
            if got.shape != exp_arr.shape:
                problems.append(f"probabilities for {col!r} have shape {got.shape}, expected {exp_arr.shape}")
            elif strict and not np.allclose(got, exp_arr, atol=tol):
                problems.append(f"probabilities for {col!r} differ from self-test (max |Δ|={np.abs(got - exp_arr).max():.2e})")
            elif not np.allclose(got.sum(axis=1), 1.0, atol=1e-3):
                problems.append(f"probabilities for {col!r} do not sum to 1")
        if not strict and not problems:
            warnings.warn(
                f"{self.model_id}: self-test compared loosely (bundle built with torch {st.get('torch_version')}, "
                f"running {torch.__version__})",
                stacklevel=2,
            )
        return problems


def load_imputer(path: str | Path, device: str = "cpu", verify: VerifyMode = "fast", threads: int | None = 1):
    return Bundle.open(path, verify=verify).load_imputer(device=device, threads=threads)
