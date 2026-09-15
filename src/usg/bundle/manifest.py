"""Bundle file names, manifest parsing and hashing helpers (no torch)."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from usg.constants import BUNDLE_FORMAT, MODEL_ID_RE
from usg.errors import BundleError, CompatibilityError

MANIFEST = "manifest.json"
SCHEMA = "schema.json"
CONFIG = "config.json"
WEIGHTS = "model.safetensors"
SELFTEST = "selftest.json"
METRICS = "metrics.json"
CALIBRATION = "confidence_calibration.json"
SENSITIVITY = "sensitivity.json"
DERIVATIONS = "postprocess_derivations.json"  # how post-processing fills columns the model does not predict
MODEL_CARD = "MODEL_CARD.md"
REQUIRED_FILES = (SCHEMA, CONFIG, WEIGHTS, SELFTEST, METRICS, MODEL_CARD)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise BundleError(f"missing bundle file: {path}") from None
    except json.JSONDecodeError as exc:
        raise BundleError(f"invalid JSON in {path}: {exc}") from None


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    """Read the ``__metadata__`` block from a safetensors header without importing torch."""
    try:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            if length > 100_000_000:
                raise BundleError(f"implausible safetensors header size in {path}")
            header = json.loads(f.read(length))
    except (OSError, struct.error, json.JSONDecodeError) as exc:
        raise BundleError(f"unreadable weights file {path}: {exc}") from None
    return header.get("__metadata__") or {}


def parse_model_id(model_id: str) -> tuple[str, str, str, int]:
    """``gnn-MD-2024.2-v1`` -> ``("gnn", "MD", "2024.2", 1)``."""
    m = MODEL_ID_RE.match(model_id)
    if not m:
        raise BundleError(f"invalid model_id {model_id!r} (expected e.g. 'gnn-MD-2024.2-v1')")
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


@dataclass(frozen=True)
class Manifest:
    raw: dict

    @classmethod
    def from_json(cls, obj: dict) -> Manifest:
        fmt = str(obj.get("bundle_format", ""))
        if not fmt.startswith("usg-bundle/v"):
            raise BundleError(f"not a usg bundle manifest (bundle_format={fmt!r})")
        major = fmt.split("/v", 1)[1].split(".", 1)[0]
        if f"usg-bundle/v{major}" != BUNDLE_FORMAT:
            raise CompatibilityError(f"bundle format {fmt!r} is not supported (this package reads {BUNDLE_FORMAT})")
        for key in ("model_id", "model_type", "schema_hash", "files"):
            if key not in obj:
                raise BundleError(f"manifest is missing required field {key!r}")
        model_type, _, _, _ = parse_model_id(obj["model_id"])
        if model_type != obj["model_type"]:
            raise BundleError(f"model_id {obj['model_id']!r} does not match model_type {obj['model_type']!r}")
        return cls(obj)

    @property
    def model_id(self) -> str:
        return self.raw["model_id"]

    @property
    def model_type(self) -> str:
        return self.raw["model_type"]

    @property
    def arch_version(self) -> int:
        return int(self.raw.get("arch_version", 1))

    @property
    def schema_hash(self) -> str:
        return self.raw["schema_hash"]

    @property
    def vocab_hash(self) -> str | None:
        return self.raw.get("vocab_hash")

    @property
    def files(self) -> dict[str, dict]:
        return self.raw["files"]

    @property
    def region(self) -> dict:
        return self.raw.get("region", {})

    @property
    def scope(self) -> str:
        return self.region.get("scope", "national" if parse_model_id(self.model_id)[1] == "national" else "state")

    @property
    def target_state(self) -> str | None:
        return self.region.get("target_state")

    @property
    def requires(self) -> str | None:
        return (self.raw.get("requires") or {}).get("usg")

    @property
    def created_by(self) -> dict:
        return self.raw.get("created_by", {})
