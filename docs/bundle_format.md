# Bundle format `usg-bundle/v1` (normative)

A **bundle** is a self-describing directory containing one trained model plus everything needed to load it, check
it and explain it. This document describes the format exactly as implemented in
`usg/bundle/manifest.py`, `writer.py`, `reader.py` and `archive.py`. Producers (the model training pipeline) must follow it;
`usg.bundle.write_bundle` is the reference writer.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

## 1. Model id

```
<model_type>-<region>-<data_release>-v<version>        e.g. gnn-MD-2024.2-v1, tabddpm-national-2024.2-v3
regex: ^(gnn|tabddpm)-([A-Z]{2}|national)-(\d{4}\.\d+)-v(\d+)$
```

* `model_type` MUST equal the manifest `model_type`.
* `region` is a two-letter state code (a state-targeted model) or `national`.
* A published model id is immutable: new weights MUST get a new `v<version>`.

## 2. Directory layout

The directory name MUST equal the model id.

| File | Required | Content |
|---|---|---|
| `manifest.json` | yes | identity, compatibility, provenance, checksums (§3) |
| `schema.json` | yes | feature schema `usg-schema/v1` (§4) |
| `config.json` | yes | architecture + inference settings (§5) |
| `model.safetensors` | yes | weights (§6) |
| `selftest.json` | yes | recorded inputs and expected outputs (§7) |
| `metrics.json` | yes | evaluation results, `{"schema": "usg-metrics/v1", ...}` |
| `MODEL_CARD.md` | yes | human-readable model card |
| `confidence_calibration.json` | no | calibration table, see [confidence_reports.md](confidence_reports.md#calibration-table-format) |
| `sensitivity.json` | no | sensitivity table, see [confidence_reports.md](confidence_reports.md#sensitivity-table-format) |
| `postprocess_derivations.json` | no | how post-processing fills the required columns the model does not predict, learned from the same ResStock release (`usg-postprocess-derivations/v1`, see [cli_usage.md](cli_usage.md#usg-process)) |
| `.verified` | no | local verification marker written by readers (§8); never distributed |

Files whose names start with `.` are ignored by checksums and archives. Additional files MAY be present; they are
covered by checksums when listed in `manifest.files`.

## 3. `manifest.json`

```json
{
  "bundle_format": "usg-bundle/v1",
  "model_id": "gnn-MD-2024.2-v1",
  "model_type": "gnn",
  "arch_version": 1,
  "schema_hash": "sha256:…",
  "vocab_hash": "sha256:…",
  "weights_dtype": "float16",
  "created_at": "2026-05-01T12:00:00Z",
  "created_by": {"usg_version": "1.0.0", "torch_version": "2.4.0", "python": "3.11.9", "…": "…"},
  "requires": {"usg": ">=1.0.0"},
  "region": {"scope": "state", "target_state": "MD", "training_states": ["MD", "VA", "DC", "DE", "PA"]},
  "files": {
    "schema.json": {"sha256": "<hex>", "size": 12345},
    "…": {}
  }
}
```

| Field | Required | Rules |
|---|---|---|
| `bundle_format` | yes | `usg-bundle/v<major>[.<minor>]`; readers accept only their major version (`v1`). Other prefixes are not bundles. |
| `model_id` | yes | §1 |
| `model_type` | yes | `gnn` or `tabddpm`; must match `model_id` |
| `schema_hash` | yes | §4.2; must equal the hash recomputed from `schema.json` |
| `files` | yes | every file of the bundle except `manifest.json` and dotfiles: `{name: {"sha256": hex, "size": bytes}}`; MUST include all required files |
| `arch_version` | no (default 1) | architecture version of `model_type`; the reader must support it |
| `vocab_hash` | recommended | §4.2 |
| `requires.usg` | no | PEP 440 specifier the installed package must satisfy (pre-releases allowed) |
| `weights_dtype` | no | `float16` or `float32` |
| `region` | recommended | `scope` (`state`/`national`), `target_state` (null for national), `training_states`, optional `n_buildings`, `target_weight`, `strategy` |
| `created_at`, `created_by` | recommended | provenance; writers fill package/torch/python versions |
| other | no | e.g. `data`, `split_hash`, `license`, `metrics_summary`, `recommended_known_inputs` — informational |

`write_bundle` fills `bundle_format`, `model_type`, `arch_version`, `schema_hash`, `vocab_hash`, `weights_dtype`,
`files`, and defaults for `created_at`, `created_by` and `requires` (`>=` the writer's package version).

## 4. `schema.json` (`usg-schema/v1`)

```json
{
  "schema_format": "usg-schema/v1",
  "data_release": "2024.2",
  "missing": {"na_tokens": [""], "none_is_value": true},
  "columns": [
    {"name": "State", "role": "conditioning", "type": "categorical", "vocab": ["AL", "AR", "…"]},
    {"name": "Vintage", "role": "imputable", "type": "categorical", "vocab": ["<1940", "1940s", "…"], "ordinal": true,
     "counts": [123, 456], "temperature": 1.07},
    {"name": "Annual Site Energy", "role": "conditioning", "type": "numeric",
     "transform": {"kind": "quantile_normal", "quantiles": [], "references": []}, "clip": [0.0, 1000000.0], "unit": "kWh"}
  ],
  "groups": {"hvac": ["Heating Fuel", "HVAC Heating Type"]},
  "metadata": {"geojson_observable": ["Vintage", "…"], "dependencies": [["Vintage", "Vintage ACS"]]}
}
```

### 4.1 Columns

* `name` — ResStock parameter name (unique). Order matters: categorical and numeric columns are indexed separately, in
  list order.
* `role` — `imputable` (the model predicts it) or `conditioning` (input only, never predicted; e.g. `State`,
  `ASHRAE IECC Climate Zone 2004`, `Building America Climate Zone`, `Census Division`, `PUMA Metro Status`,
  `Annual Site Energy`).
* `type` — `categorical` (requires a non-empty, duplicate-free `vocab` of strings; `ordinal` optional) or `numeric`
  (requires `transform`; `clip` `[lo, hi]` and `unit` optional).
* `counts`, `temperature`, `description` — metadata, excluded from the schema hash. `temperature` is a per-column
  softmax temperature fitted on validation data; backends apply it, consumers MUST NOT apply it again.
* `groups` members and `metadata` are free-form but `groups` MUST reference existing columns. Readers ignore unknown
  `metadata` keys.

### 4.2 Hashes

Both hashes are `"sha256:" + hex(sha256(json))` over a canonical JSON serialization
(`sort_keys=True, separators=(",", ":"), ensure_ascii=False`, UTF-8) of one list per column:

* `schema_hash`: `[name, role, type, vocab|null, transform|null, clip|null]`
* `vocab_hash`: `[name, type, vocab|null]` — identical across all bundles of one data release and feature set.

### 4.3 Encoding contract

* Cell values are canonicalized: missing/NaN/empty → missing; `"None"` is a real option and kept; integral floats
  collapse (`3.0`, `"3.0"` → `"3"`); strings are NFC-normalized and stripped.
* Categorical code = index in `vocab` (exact match, then canonical form, then case-insensitive); `-1` = missing. Models
  reserve index `len(vocab)` as their mask token.
* Unrecognized labels are treated as missing (reported as input issues) or rejected (`unknown="error"`).
* Numeric values are clipped, then mapped to standard-normal scores with the `quantile_normal` transform
  (piecewise-linear interpolation between `quantiles` and `references`); predictions are mapped back with the inverse.

## 5. `config.json`

```json
{
  "model_type": "gnn",
  "arch_version": 1,
  "arch": {"d_small": 16, "d_large": 64, "d_numeric": 16, "hidden": 512, "n_blocks": 4, "expansion": 2, "dropout": 0.1},
  "inference": {"decode": "refine", "rounds": 8, "default_samples": 1, "max_samples": 1024},
  "diffusion": {"timesteps": 1000, "schedule": "cosine"}
}
```

`model_type` MUST equal the manifest's. `arch` is passed to the architecture constructor of
`usg.models` (`build_model(schema, config)`). `inference` keys: `decode` (G-NN: `argmax`/`refine`/`sample`;
TabDDPM: `medoid`/`mode` via `aggregate`), `rounds`, `default_samples`, `max_samples`, `sample_steps`. `diffusion` is
TabDDPM-only.

## 6. `model.safetensors`

* A `state_dict` of the network built by `build_model(schema, config)`; loading is strict (no missing/unexpected keys).
* Floating tensors are stored as float16 (default) or float32 and upcast to float32 on load.
* The safetensors `__metadata__` MUST contain `model_id` and `schema_hash` equal to the manifest's (and
  `bundle_format`). Readers check this from the header without loading tensors.

## 7. `selftest.json`

```json
{
  "inputs": [{"State": "MD", "Geometry Building Type RECS": "Single-Family Detached", "Heating Fuel": null}],
  "seed": 0,
  "n_samples": 1,
  "torch_version": "2.4.0",
  "tolerance": 0.0001,
  "expected": {
    "values": [{"Vintage": "1990s", "…": "…"}],
    "probabilities": {"Vintage": [[0.01, 0.2, "…"]]}
  }
}
```

Written by `write_bundle` from the bundle *as saved* (fp16 weights upcast). `expected.values` covers imputable
columns; `probabilities` are rounded to 6 decimals. `Bundle.run_selftest()` re-runs `inputs` with `seed`/`n_samples`
and compares: strictly (values equal, probabilities within `tolerance`) for G-NN or when the running torch version
equals `torch_version`; otherwise loosely (shapes and normalization only, with a warning), since sampling kernels may
differ between torch versions.

## 8. Verification and compatibility

`Bundle.open(path, verify=...)`:

1. parse `manifest.json`; reject unknown `bundle_format` majors (`CompatibilityError`);
2. check `arch_version` is supported for `model_type` and `requires.usg` is satisfied (`CompatibilityError`);
3. `verify="full"`: every required file present and listed; every listed file's sha256 matches; then write
   `.verified` = `{"manifest_sha256": <sha256 of manifest.json>}` (best effort).
   `verify="fast"`: if `.verified` matches the current manifest, only file presence and sizes are checked; otherwise
   behaves like `full`. `verify="none"`: skip file checks;
4. parse `schema.json` and require `schema_hash` to match; `config.json` `model_type` must match;
5. check the weights header metadata.

Errors are `BundleError` (CLI exit code 4) or its subclass `CompatibilityError`.

## 9. Archives

Bundles are distributed as `.tar.gz` produced by `pack_bundle(bundle_dir, out)`:

* exactly one top-level directory named after the model id, containing the bundle's regular files (dotfiles
  excluded), sorted by name;
* reproducible: PAX format, `mtime=0`, `uid=gid=0`, empty owner names, mode `0644`, gzip header `mtime=0` and empty
  filename — packing the same directory twice yields identical bytes;
* the archive sha256 is recorded in the registry (`sha256`), the manifest sha256 as `manifest_sha256`.

`extract_bundle(archive, dest_parent, expected_name)` rejects absolute paths, `..` components, links/devices, more
than one top-level directory or a name different from `expected_name`, extracts into a staging directory and renames
it into place atomically.
