# Models, registry and selection

## Model types

| Type | `--model` | What it is | Probabilities | Speed |
|---|---|---|---|---|
| G-NN | `gnn` | Masked any-order imputation network: predicts every hidden column from the known ones (optionally refining the most confident columns first). | softmax over each column's options | fast, deterministic with `argmax`/`refine` decoding |
| TabDDPM | `tabddpm` | Tabular diffusion model: draws `--samples` complete buildings consistent with the known inputs and reports a representative (medoid) sample. | smoothed frequencies over the drawn samples | slower; more coherent joint samples |

Both share the same schema and outputs. Use `--samples 16` or more when you want TabDDPM probabilities or native
confidence scores.

## Regions

The published models are **national**: `gnn-national-2024.2-v1` and `tabddpm-national-2024.2-v1`, trained on every
ResStock 2024.2 baseline building (48 contiguous states + DC). Location enters as conditioning inputs, and the
registry names the national model as the recommended model for every state. The registry format also supports
per-state champions should regional models be published later. Alaska and Hawaii are not in ResStock; they use the
national model with a warning.

Every model also takes location **conditioning** inputs: `State`, `ASHRAE IECC Climate Zone 2004`, `Building America
Climate Zone`, `Census Division` and `PUMA Metro Status`. Every building of a GeoJSON project gets the same location,
filled from, in order of precedence:

* explicit options (`--state`, `--climate-zone`, `--latitude/--longitude`);
* the weather filename (`USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw` → `MD`, from
  `project.weather_filename` or the *Site Origin* feature);
* the nearest ResStock weather station to the *Site Origin* point. It gives the climate zones, and the state when there
  is no weather filename.

`project.climate_zone` and building footprints are not used. Anything uncertain stays unknown. See
[location.md](location.md).

## Selection

Which bundle runs is decided in this order:

1. `--model-dir DIR` — use that bundle directory, nothing else is consulted.
2. `--model-id ID` — exactly that model (embedded, cached, or downloaded from the registry).
3. **Region** — `--state XX`, `--state national`, or `--state auto` (default). With `auto`, the region is the
   project's resolved state (weather file, else coordinates); for `usg complete`, the CSV's `State` column, which
   latitude/longitude columns can fill; otherwise national.
4. **Type** — `--model`, else the registry's `default_model_type` for the region.
5. **Source** — the registry *champion* (recommended model) for region + type, taken from the embedded bundles
   shipped in the wheel, else the local cache, else downloaded.

Fallbacks, each reported as a warning (and recorded in `run_manifest.json`):

* no model published for the region (e.g. `AK`) → the national model;
* download failure or offline without a cached copy → the embedded national G-NN (when the wheel ships one).

`--no-fallback` turns fallbacks into errors (exit code 3). The chosen model and the reason are always printed, e.g.

```
Model: tabddpm-national-2024.2-v1 — state MD from weather file (USA_MD_Baltimore…epw); model type tabddpm (requested); the national tabddpm model is the recommended model for MD; tabddpm-national-2024.2-v1 (embedded)
```

**Mixed-state CSVs.** `usg complete` with `--state auto` and a `State` column holding several states resolves one
model per state and completes each partition with its own model; rows without a recognizable state use the
national model. Output row order is preserved.

## Cache and offline use

Downloaded bundles live in `$USG_CACHE_DIR` (default: the platform cache directory, e.g.
`~/Library/Caches/usg/models` on macOS, `~/.cache/usg/models` on Linux):

```
<model_id>/            extracted, verified bundle
tmp/<id>.<pid>.part    partial downloads (deleted on failure)
locks/<id>.lock        per-model file lock (safe for parallel URBANopt runs)
```

Downloads are verified against the registry's archive `sha256` (a mismatch deletes the partial file) and the
bundle's own file checksums before the bundle is moved into place.

A wheel that embeds both national models works offline as is. Otherwise, for air-gapped machines:

```bash
usg models download gnn-national-2024.2-v1 tabddpm-national-2024.2-v1
usg models path gnn-national-2024.2-v1    # copy that directory, or the whole `usg models path` cache
export USG_OFFLINE=1                       # never touch the network (same as --offline)
usg workflow -i buildings.json            # uses embedded/cached models only
usg workflow -i buildings.json --model-dir /shared/bundles/gnn-national-2024.2-v1   # or point at a bundle
```

## `usg models`

| Command | Purpose |
|---|---|
| `usg models list [--state XX] [--model T] [--json]` | registry models (`*` = champion) with status `embedded`/`cached`/`remote`/`yanked` and headline metrics |
| `usg models download [IDS...] [--state XX --model T]` | fetch into the cache |
| `usg models info ID [--card]` | registry entry, local manifest, model card |
| `usg models verify ID\|DIR [--full] [--selftest]` | re-check checksums; optionally load the model and replay its self-test |
| `usg models clear-cache [ID] [--yes]` | delete cached bundles |
| `usg models path [ID]` | print the cache directory or a model's local directory |

## Registry format (`usg-registry/v1`)

The package ships `usg/resources/registry.json`. Point to another file or URL with `--registry` or
`$USG_REGISTRY`.

```json
{
  "registry_format": "usg-registry/v1",
  "generated_at": "2026-06-01T00:00:00Z",
  "data_release": "2024.2",
  "vocab_hash": "sha256:…",
  "default_model_type": {"__default__": "gnn"},
  "champions": {
    "MD": {"gnn": "gnn-national-2024.2-v1", "tabddpm": "tabddpm-national-2024.2-v1"},
    "national": {"gnn": "gnn-national-2024.2-v1", "tabddpm": "tabddpm-national-2024.2-v1"}
  },
  "models": [
    {
      "model_id": "gnn-national-2024.2-v1",
      "model_type": "gnn",
      "region": "national",
      "scope": "national",
      "arch_version": 1,
      "bundle_format": "usg-bundle/v1",
      "schema_hash": "sha256:…",
      "release_tag": "models-2024.2-v1",
      "url": "https://github.com/NatLabRockies/urban-system-generator/releases/download/models-2024.2-v1/gnn-national-2024.2-v1.tar.gz",
      "sha256": "<archive sha256>",
      "manifest_sha256": "<sha256 of manifest.json>",
      "size_bytes": 30900000,
      "requires_usg": ">=1.0",
      "headline": {"accuracy": 0.75, "skill": 0.28},
      "embedded": true,
      "yanked": false
    }
  ]
}
```

Validation rules enforced by `Registry`: `model_id` matches the id pattern and agrees with `model_type` and `region`;
ids are unique; every champion references a listed model of the same type whose region is that region or national;
`default_model_type` values
are model types. A yanked champion is ignored (fallbacks apply); a yanked model can still be requested by id (with a
warning). `sha256` is required to download; `manifest_sha256`, when present, must match the extracted bundle.

## Embedded bundles

Wheels may ship bundles under `usg/resources/bundles/<model_id>/` (fetched and checksum-verified at wheel
build time, never committed). Embedded bundles are preferred over the cache and are the offline fallback. Mark them
`"embedded": true` in the registry, then before building the wheel:

```bash
python scripts/fetch_embedded_bundles.py                                        # from the release assets
python scripts/fetch_embedded_bundles.py --from-dir path/to/exported/bundles   # from a local export
```

Both check each bundle against the registry's `manifest_sha256` and verify every file.

## Environment variables

| Variable | Effect |
|---|---|
| `USG_CACHE_DIR` | model cache directory |
| `USG_REGISTRY` | registry file or URL |
| `USG_OFFLINE` | `1`/`true`: never download |
| `USG_EMBEDDED_DIR` | directory used instead of the wheel's embedded bundles (e.g. a shared folder of bundles) |
| `USG_NUM_THREADS` | torch CPU threads (overrides `--threads`; default 1 so URBANopt's Ruby subprocess never hangs) |
| `USG_DEBUG` | show Python tracebacks for unexpected CLI errors |
