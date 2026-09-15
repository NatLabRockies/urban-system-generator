"""``usg models``: browse the registry and manage the local model cache."""

from __future__ import annotations

import json
from pathlib import Path

import click

from usg.cli._common import CONTEXT_SETTINGS, handle_errors, ok
from usg.constants import MODEL_TYPES

_REGISTRY_OPTION = click.option("--registry", default=None, help="Registry file or URL ($USG_REGISTRY).")


def _local_path(model_id: str) -> tuple[Path | None, str]:
    from usg.download import Cache
    from usg.resources import embedded_bundles_dir

    embedded = embedded_bundles_dir() / model_id
    if (embedded / "manifest.json").is_file():
        return embedded, "embedded"
    cache = Cache()
    if cache.is_cached(model_id):
        return cache.path(model_id), "cached"
    return None, "remote"


def _size(n: int | None) -> str:
    return "" if n is None else f"{n / 1e6:.1f} MB"


@click.group(context_settings=CONTEXT_SETTINGS)
def models():
    """Browse, download, verify and remove model bundles."""


@models.command("list")
@click.option("--state", "region", default=None, help="Only this region (two-letter state or 'national').")
@click.option("--model", "model_type", type=click.Choice(MODEL_TYPES), default=None, help="Only this model type.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@_REGISTRY_OPTION
@handle_errors
def list_models(region, model_type, as_json, registry):
    """List registry models (* = recommended for its region) and their local status."""
    from usg.download import Cache
    from usg.registry import Registry

    reg = Registry.load(registry)
    rows = []
    for entry in reg.models(region=region, model_type=model_type):
        _, status = _local_path(entry.model_id)
        rows.append({**entry.to_json(), "champion": reg.is_champion(entry.model_id),
                     "status": "yanked" if entry.yanked else status})
    listed = {r["model_id"] for r in rows}
    for path in Cache().list():
        if path.name not in listed and region is None and model_type is None:
            rows.append({"model_id": path.name, "status": "cached (not in registry)", "champion": False})
    if as_json:
        click.echo(json.dumps({"registry": reg.source, "models": rows}, indent=2))
        return
    click.echo(f"Registry: {reg.source}")
    if not rows:
        click.echo("No models listed. Pass --registry <file|url> or use --model-dir with a local bundle.")
        return
    for r in rows:
        headline = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in (r.get("headline") or {}).items()
        )
        click.echo(f"{'*' if r['champion'] else ' '} {r['model_id']:<28} {r['status']:<10} "
                   f"{_size(r.get('size_bytes')):>9}  {headline}")


@models.command("download")
@click.argument("model_ids", nargs=-1)
@click.option("--state", "region", default=None, help="Download the recommended model for this region.")
@click.option("--model", "model_type", type=click.Choice(MODEL_TYPES), default=None,
              help="Model type for --state (default: the registry default).")
@_REGISTRY_OPTION
@handle_errors
def download(model_ids, region, model_type, registry):
    """Download models into the cache (by id, or the recommended model for --state)."""
    from usg.download import Cache
    from usg.errors import SelectionError
    from usg.registry import Registry

    reg = Registry.load(registry)
    entries = [reg.get(m) for m in model_ids]
    if region:
        kind = model_type or reg.default_type(region)
        champ = reg.champion(region, kind)
        if champ is None:
            raise SelectionError(f"no recommended {kind} model for {region} in {reg.source}")
        entries.append(champ)
    if not entries:
        raise click.UsageError("give one or more MODEL_IDS or --state")
    cache = Cache()
    for entry in entries:
        click.echo(f"Downloading {entry.model_id} ({_size(entry.size_bytes) or 'unknown size'})...")
        ok(f"{entry.model_id} -> {cache.ensure(entry)}")


@models.command("info")
@click.argument("model_id")
@click.option("--card", is_flag=True, help="Print the model card.")
@_REGISTRY_OPTION
@handle_errors
def model_info(model_id, card, registry):
    """Registry entry and local manifest for one model."""
    from usg.bundle import Bundle
    from usg.registry import Registry

    reg = Registry.load(registry)
    if model_id in reg:
        click.echo(json.dumps(reg.get(model_id).to_json(), indent=2))
    path, status = _local_path(model_id)
    click.echo(f"Local status: {status}{f' ({path})' if path else ''}")
    if path is None:
        if model_id not in reg:
            from usg.errors import SelectionError

            raise SelectionError(f"{model_id} is neither in the registry nor available locally")
        return
    bundle = Bundle.open(path, verify="none")
    raw = bundle.manifest.raw
    for key in ("created_at", "arch_version", "weights_dtype", "schema_hash", "vocab_hash"):
        click.echo(f"  {key}: {raw.get(key)}")
    click.echo(f"  region: {json.dumps(bundle.manifest.region)}")
    if raw.get("metrics_summary"):
        click.echo(f"  metrics_summary: {json.dumps(raw['metrics_summary'])}")
    if card:
        click.echo("\n" + bundle.model_card())


@models.command("verify")
@click.argument("target")
@click.option("--full", is_flag=True, help="Re-hash every file (ignore the verification marker).")
@click.option("--selftest", is_flag=True, help="Also load the model and re-run its recorded self-test.")
@handle_errors
def verify(target, full, selftest):
    """Verify a bundle given its model id or directory."""
    from usg.bundle import Bundle
    from usg.errors import BundleError, SelectionError

    path = Path(target)
    if not path.is_dir():
        path, _ = _local_path(target)
        if path is None:
            raise SelectionError(f"{target} is not a directory and not available locally")
    bundle = Bundle.open(path, verify="full" if full else "fast")
    ok(f"{bundle.model_id}: files verified ({'full' if full else 'fast'})")
    if selftest:
        problems = bundle.run_selftest()
        if problems:
            raise BundleError(f"{bundle.model_id} self-test failed: " + "; ".join(problems))
        ok(f"{bundle.model_id}: self-test passed")


@models.command("clear-cache")
@click.argument("model_id", required=False)
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
@handle_errors
def clear_cache(model_id, yes):
    """Remove one cached model, or the whole cache."""
    from usg.download import Cache

    cache = Cache()
    what = model_id or f"all cached models in {cache.dir}"
    if not yes:
        click.confirm(f"Remove {what}?", abort=True)
    ok(f"removed {cache.clear(model_id)} bundle(s)")


@models.command("path")
@click.argument("model_id", required=False)
@handle_errors
def path_cmd(model_id):
    """Print the cache directory, or the local directory of MODEL_ID."""
    from usg.download import Cache
    from usg.errors import SelectionError

    if model_id is None:
        click.echo(Cache().dir)
        return
    path, _ = _local_path(model_id)
    if path is None:
        raise SelectionError(f"{model_id} is not available locally; run `usg models download {model_id}`")
    click.echo(path)
