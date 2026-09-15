"""Resolve which model bundle to run.

Precedence: ``bundle_dir`` > ``model_id`` > region (explicit state | ``national`` | ``auto`` = project state, else
national) > model type (requested, else the registry default for the region) > source (embedded bundle -> cache ->
download). Fallbacks (with warnings, unless ``allow_fallback=False``): no model for the region -> national; download
failure or offline without a cached copy -> the embedded national G-NN.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from usg.bundle.manifest import MANIFEST, Manifest, parse_model_id, read_json
from usg.constants import MODEL_TYPES, NATIONAL, STATES
from usg.download import Cache
from usg.errors import (
    CompatibilityError,
    DownloadError,
    InputValidationError,
    RegistryError,
    SelectionError,
)
from usg.registry import Registry, RegistryEntry, normalize_region
from usg.schema import canonical_label

if TYPE_CHECKING:
    from usg.geojson import ProjectContext


@dataclass
class ModelRequest:
    model_type: str | None = None
    state: str | None = "auto"
    model_id: str | None = None
    bundle_dir: str | Path | None = None
    context: ProjectContext | None = None
    offline: bool = False
    allow_fallback: bool = True


@dataclass
class ResolvedModel:
    bundle_path: Path
    model_id: str
    region: str
    reason: str
    warnings: list[str] = field(default_factory=list)
    source: str | None = None  # bundle_dir | embedded | cache | download
    archive_sha256: str | None = None

    @property
    def model_type(self) -> str:
        return parse_model_id(self.model_id)[0]


def state_key(value) -> str | None:
    """Canonical state code for a cell value, or ``None`` when missing / not a state."""
    label = canonical_label(value)
    if label is None:
        return None
    label = label.upper()
    return label if label in STATES else None


def _is_auto(state: str | None) -> bool:
    return state is None or str(state).strip().lower() == "auto"


def _region(req: ModelRequest) -> tuple[str, str]:
    if _is_auto(req.state):
        ctx = req.context
        if ctx is not None and ctx.state:
            return ctx.state, f"state {ctx.state} from {ctx.state_source or 'the project'}"
        return NATIONAL, "no project state found, using the national model"
    try:
        region = normalize_region(req.state)
    except RegistryError:
        raise InputValidationError(f"invalid state {req.state!r}: use a two-letter code, 'national' or 'auto'") from None
    return region, "national model requested" if region == NATIONAL else f"state {region} requested"


def _embedded_national_gnn(registry: Registry, embedded_dir: Path) -> str | None:
    champ = registry.champion(NATIONAL, "gnn")
    if champ is not None and (embedded_dir / champ.model_id / MANIFEST).is_file():
        return champ.model_id
    if not embedded_dir.is_dir():
        return None
    found = sorted(p.name for p in embedded_dir.glob("gnn-national-*") if (p / MANIFEST).is_file())
    return found[-1] if found else None


def _check_requires(entry: RegistryEntry) -> None:
    if not entry.requires_usg:
        return
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    from usg.__about__ import __version__

    try:
        ok = SpecifierSet(entry.requires_usg).contains(__version__, prereleases=True)
    except InvalidSpecifier:
        raise RegistryError(f"{entry.model_id}: invalid requires_usg {entry.requires_usg!r}") from None
    if not ok:
        raise CompatibilityError(
            f"{entry.model_id} requires urban-system-generator {entry.requires_usg}, installed {__version__}; "
            "upgrade with `pip install -U urban-system-generator`"
        )


def _from_bundle_dir(req: ModelRequest) -> ResolvedModel:
    path = Path(req.bundle_dir).expanduser()
    manifest = Manifest.from_json(read_json(path / MANIFEST))
    region = parse_model_id(manifest.model_id)[1]
    warnings = []
    if req.model_type and req.model_type != manifest.model_type:
        warnings.append(f"--model {req.model_type} ignored: {path} is a {manifest.model_type} bundle")
    if not _is_auto(req.state):
        wanted, _ = _region(req)
        if wanted != region and region != NATIONAL:  # a national bundle still takes the state as an input
            warnings.append(f"--state {wanted} ignored: {path} is the {region} model")
    return ResolvedModel(path, manifest.model_id, region, f"bundle directory {path}", warnings, source="bundle_dir")


def _materialize(
    model_id: str,
    entry: RegistryEntry | None,
    region: str,
    reason: str,
    warnings: list[str],
    req: ModelRequest,
    registry: Registry,
    cache: Cache,
    embedded_dir: Path,
) -> ResolvedModel:
    sha = entry.sha256 if entry else None
    embedded = embedded_dir / model_id
    if (embedded / MANIFEST).is_file():
        return ResolvedModel(embedded, model_id, region, f"{reason}; {model_id} (embedded)", warnings, "embedded", sha)
    if cache.is_cached(model_id, entry.manifest_sha256 if entry else None):
        return ResolvedModel(cache.path(model_id), model_id, region, f"{reason}; {model_id} (cached)", warnings, "cache", sha)
    if entry is None:
        raise SelectionError(f"{model_id} is not in the registry ({registry.source}), not embedded and not cached")
    _check_requires(entry)
    try:
        path = cache.ensure(entry, offline=req.offline)
    except DownloadError as exc:
        fallback = _embedded_national_gnn(registry, embedded_dir) if req.allow_fallback else None
        if fallback is None or fallback == model_id:
            raise
        warnings.append(f"{exc}; falling back to the embedded {fallback}")
        return ResolvedModel(
            embedded_dir / fallback, fallback, NATIONAL, f"{reason}; {model_id} unavailable, using embedded {fallback}",
            warnings, "embedded",
        )
    return ResolvedModel(path, model_id, region, f"{reason}; {model_id} (downloaded)", warnings, "download", sha)


def resolve(
    req: ModelRequest,
    registry: Registry | None = None,
    cache: Cache | None = None,
    *,
    embedded_dir: str | Path | None = None,
) -> ResolvedModel:
    if req.model_type is not None and req.model_type not in MODEL_TYPES:
        raise InputValidationError(f"unknown model type {req.model_type!r}; choose one of {MODEL_TYPES}")
    if req.bundle_dir is not None:
        return _from_bundle_dir(req)
    from usg.resources import embedded_bundles_dir

    embedded_dir = Path(embedded_dir) if embedded_dir is not None else embedded_bundles_dir()
    registry = registry if registry is not None else Registry.load()
    cache = cache if cache is not None else Cache()
    warnings: list[str] = []

    if req.model_id:
        model_type, region, _, _ = parse_model_id(req.model_id)
        if req.model_type and req.model_type != model_type:
            warnings.append(f"--model {req.model_type} ignored: --model-id {req.model_id} is a {model_type} model")
        entry = registry.get(req.model_id) if req.model_id in registry else None
        if entry is not None and entry.yanked:
            warnings.append(f"{req.model_id} has been yanked from the registry; prefer the current champion")
        return _materialize(
            req.model_id, entry, region, f"model {req.model_id} requested", warnings, req, registry, cache, embedded_dir
        )

    region, reason = _region(req)
    model_type = req.model_type or registry.default_type(region)
    reason += f"; model type {model_type} ({'requested' if req.model_type else 'default for ' + region})"
    entry = registry.champion(region, model_type)
    if entry is not None and region != NATIONAL and entry.scope == NATIONAL:
        # deliberate recommendation (no regional model beat the national one) — not a fallback, so no warning
        reason += f"; the national {model_type} model is the recommended model for {region}"
        region = NATIONAL
    if entry is None and region != NATIONAL:
        message = f"no {model_type} model is published for {region}"
        if not req.allow_fallback:
            raise SelectionError(f"{message} (fallback disabled); use --state national or --model-dir")
        if not req.model_type:
            model_type = registry.default_type(NATIONAL)
        entry = registry.champion(NATIONAL, model_type)
        warnings.append(f"{message}; using the national {model_type} model")
        reason += f"; fell back to national {model_type}"
        region = NATIONAL
    if entry is None:
        fallback = _embedded_national_gnn(registry, embedded_dir) if req.allow_fallback else None
        if fallback is None:
            raise SelectionError(
                f"no {model_type} model for {region} in the registry ({registry.source}) and no embedded model; "
                "pass --model-dir <bundle> or --registry <file|url>"
            )
        warnings.append(f"no {model_type} model for {region} in the registry; using the embedded {fallback}")
        return ResolvedModel(
            embedded_dir / fallback, fallback, NATIONAL, f"{reason}; {fallback} (embedded fallback)", warnings, "embedded"
        )
    return _materialize(entry.model_id, entry, region, reason, warnings, req, registry, cache, embedded_dir)


def resolve_partitioned(
    req: ModelRequest,
    states: Iterable,
    registry: Registry | None = None,
    cache: Cache | None = None,
    *,
    embedded_dir: str | Path | None = None,
) -> dict[str | None, ResolvedModel]:
    """One resolution per distinct state in ``states`` (key ``None`` = rows without a recognizable state).

    Only ``state="auto"`` partitions; an explicit state, model id or bundle directory applies to every row.
    """
    keys = sorted({state_key(v) for v in states}, key=lambda k: (k is None, k or ""))
    if not keys:
        keys = [None]
    if req.bundle_dir is None:
        registry = registry if registry is not None else Registry.load()
        cache = cache if cache is not None else Cache()
    if req.bundle_dir is not None or req.model_id or not _is_auto(req.state):
        one = resolve(req, registry, cache, embedded_dir=embedded_dir)
        return {k: one for k in keys}
    out: dict[str | None, ResolvedModel] = {}
    for key in keys:
        sub = req if key is None else replace(req, state=key)
        out[key] = resolve(sub, registry, cache, embedded_dir=embedded_dir)
    return out
