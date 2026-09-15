"""Model registry (``usg-registry/v1``): which bundles exist, where to download them, and which is the default.

The packaged ``resources/registry.json`` is used unless ``source`` or ``$USG_REGISTRY`` points at another file or URL.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from usg.constants import ENV_REGISTRY, MODEL_ID_RE, MODEL_TYPES, NATIONAL, REGISTRY_FORMAT, STATES
from usg.errors import RegistryError

DEFAULT_KEY = "__default__"


def normalize_region(region: str) -> str:
    """``"md"`` -> ``"MD"``, ``"National"`` -> ``"national"``; raises for anything else."""
    text = str(region).strip()
    if text.lower() == NATIONAL:
        return NATIONAL
    if text.upper() in STATES:
        return text.upper()
    raise RegistryError(f"unknown region {region!r} (expected a two-letter state code or 'national')")


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    model_type: str
    region: str
    scope: str
    url: str | None = None
    sha256: str | None = None
    manifest_sha256: str | None = None
    size_bytes: int | None = None
    arch_version: int = 1
    bundle_format: str | None = None
    schema_hash: str | None = None
    release_tag: str | None = None
    requires_usg: str | None = None
    headline: dict = field(default_factory=dict)
    embedded: bool = False
    yanked: bool = False

    @classmethod
    def from_json(cls, obj: dict) -> RegistryEntry:
        try:
            model_id = str(obj["model_id"])
        except (KeyError, TypeError):
            raise RegistryError(f"registry model entry without model_id: {obj!r}") from None
        m = MODEL_ID_RE.match(model_id)
        if not m:
            raise RegistryError(f"registry entry has invalid model_id {model_id!r}")
        model_type = obj.get("model_type", m.group(1))
        if model_type != m.group(1) or model_type not in MODEL_TYPES:
            raise RegistryError(f"{model_id}: model_type {model_type!r} does not match the model_id")
        region = normalize_region(obj.get("region", m.group(2)))
        if region != m.group(2):
            raise RegistryError(f"{model_id}: region {region!r} does not match the model_id")
        size = obj.get("size_bytes")
        return cls(
            model_id=model_id,
            model_type=model_type,
            region=region,
            scope=obj.get("scope") or (NATIONAL if region == NATIONAL else "state"),
            url=obj.get("url"),
            sha256=obj.get("sha256"),
            manifest_sha256=obj.get("manifest_sha256"),
            size_bytes=int(size) if size is not None else None,
            arch_version=int(obj.get("arch_version", 1)),
            bundle_format=obj.get("bundle_format"),
            schema_hash=obj.get("schema_hash"),
            release_tag=obj.get("release_tag"),
            requires_usg=obj.get("requires_usg"),
            headline=dict(obj.get("headline") or {}),
            embedded=bool(obj.get("embedded", False)),
            yanked=bool(obj.get("yanked", False)),
        )

    def to_json(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "region": self.region,
            "scope": self.scope,
            "arch_version": self.arch_version,
            "bundle_format": self.bundle_format,
            "schema_hash": self.schema_hash,
            "release_tag": self.release_tag,
            "url": self.url,
            "sha256": self.sha256,
            "manifest_sha256": self.manifest_sha256,
            "size_bytes": self.size_bytes,
            "requires_usg": self.requires_usg,
            "headline": self.headline,
            "embedded": self.embedded,
            "yanked": self.yanked,
        }


def _read_source(source: str | Path) -> tuple[Any, str]:
    text = str(source)
    if text.startswith(("http://", "https://", "file://")):
        try:
            with urllib.request.urlopen(text, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8")), text
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RegistryError(f"cannot fetch model registry {text}: {exc}") from None
    path = Path(text).expanduser()
    try:
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    except FileNotFoundError:
        raise RegistryError(f"model registry not found: {path}") from None
    except (OSError, ValueError) as exc:
        raise RegistryError(f"cannot read model registry {path}: {exc}") from None


class Registry:
    def __init__(self, raw: dict, source: str = "<memory>"):
        if not isinstance(raw, dict) or raw.get("registry_format") != REGISTRY_FORMAT:
            fmt = raw.get("registry_format") if isinstance(raw, dict) else None
            raise RegistryError(f"{source}: unsupported registry_format {fmt!r} (expected {REGISTRY_FORMAT!r})")
        self.raw = raw
        self.source = source
        self._entries: dict[str, RegistryEntry] = {}
        for obj in raw.get("models") or []:
            entry = RegistryEntry.from_json(obj)
            if entry.model_id in self._entries:
                raise RegistryError(f"{source}: duplicate model_id {entry.model_id!r}")
            self._entries[entry.model_id] = entry
        self._champions: dict[str, dict[str, str]] = {}
        for region, by_type in (raw.get("champions") or {}).items():
            key = normalize_region(region)
            for model_type, model_id in (by_type or {}).items():
                if model_id not in self._entries:
                    raise RegistryError(f"{source}: champion {model_id!r} for {key}/{model_type} is not listed")
                entry = self._entries[model_id]
                # a state's champion is its own model, or the national model when no regional model beat it
                region_ok = entry.region == key or entry.scope == NATIONAL
                if entry.model_type != model_type or not region_ok:
                    raise RegistryError(f"{source}: champion {model_id!r} does not match {key}/{model_type}")
                self._champions.setdefault(key, {})[model_type] = model_id
        self._default_type = {
            (k if k == DEFAULT_KEY else normalize_region(k)): v for k, v in (raw.get("default_model_type") or {}).items()
        }
        for key, value in self._default_type.items():
            if value not in MODEL_TYPES:
                raise RegistryError(f"{source}: default_model_type[{key!r}] = {value!r} is not a model type")

    def __repr__(self) -> str:
        return f"Registry({self.source!r}, models={len(self._entries)})"

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def load(cls, source: str | Path | None = None) -> Registry:
        """Load from ``source`` (path or URL), else ``$USG_REGISTRY``, else the packaged registry."""
        if source is None:
            source = os.environ.get(ENV_REGISTRY) or None
        if source is None:
            from usg.resources import registry_path

            source = registry_path()
        raw, where = _read_source(source)
        return cls(raw, where)

    @property
    def data_release(self) -> str | None:
        return self.raw.get("data_release")

    @property
    def vocab_hash(self) -> str | None:
        return self.raw.get("vocab_hash")

    def get(self, model_id: str) -> RegistryEntry:
        try:
            return self._entries[model_id]
        except KeyError:
            raise RegistryError(f"model {model_id!r} is not in the registry ({self.source})") from None

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._entries

    def models(self, region: str | None = None, model_type: str | None = None) -> list[RegistryEntry]:
        key = normalize_region(region) if region is not None else None
        return [
            e
            for e in self._entries.values()
            if (key is None or e.region == key) and (model_type is None or e.model_type == model_type)
        ]

    def champion(self, region: str, model_type: str) -> RegistryEntry | None:
        """The recommended (non-yanked) model for ``region``/``model_type``, if any."""
        model_id = self._champions.get(normalize_region(region), {}).get(model_type)
        if model_id is None:
            return None
        entry = self._entries[model_id]
        return None if entry.yanked else entry

    def is_champion(self, model_id: str) -> bool:
        return any(model_id in by_type.values() for by_type in self._champions.values())

    def default_type(self, region: str) -> str:
        key = normalize_region(region)
        return self._default_type.get(key) or self._default_type.get(DEFAULT_KEY) or "gnn"
