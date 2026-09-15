# `usg` command-line reference

Installing `urban-system-generator` provides the `usg` command (alias `urban-system-generator`; `python -m usg.cli` works
too). Every command accepts `-h/--help`.

```
usg [--version] COMMAND [OPTIONS]

  geojson2csv     Convert an URBANopt GeoJSON file to a building CSV
  predict-single  Predict missing attributes for a single building
  complete        Complete missing attributes for a CSV of buildings
  confidence      Complete a CSV and write per-building confidence reports
  process         Post-process a completed CSV for URBANopt-BuildStock
  workflow        GeoJSON -> completed, URBANopt-ready CSV in one step
  info            Show the selected model and packaged resources
  models          Browse, download, verify and remove model bundles
```

## Model options

`predict-single`, `complete`, `confidence`, `workflow` and `info` share these options (see [models.md](models.md)):

| Option | Default | Meaning |
|---|---|---|
| `--model [gnn\|tabddpm]` | registry default for the region | model type |
| `--state XX\|national\|auto` | `auto` | region; `auto` = GeoJSON project state / CSV `State` column / national |
| `--model-id ID` | | exact model, e.g. `tabddpm-national-2024.2-v1` |
| `-d, --model-dir DIR` | | use a local bundle directory (bypasses the registry) |
| `--samples N` | bundle default | samples per building (TabDDPM) |
| `--seed N` | `0` | random seed (runs are reproducible for a given seed, model and torch version) |
| `--device` | `cpu` | `cpu`, `cuda`, `mps` or `auto` |
| `--threads N` | `1` | torch CPU threads (`USG_NUM_THREADS` overrides) |
| `--offline` | | never download (`USG_OFFLINE=1`) |
| `--no-fallback` | | fail instead of falling back to the national/embedded model |
| `--registry FILE\|URL` | packaged | model registry (`USG_REGISTRY`) |
| `--unknown-labels [missing\|error]` | `missing` | invalid option values are treated as missing (and reported) or rejected |

A two-letter `--state` also sets the `State` input; `auto` and `national` only affect model selection.

## Location options

`workflow`, `geojson2csv`, `process`, `complete`, `confidence` and `predict-single` accept these options. The
precedence, sources and thresholds are in [location.md](location.md).

| Option | Meaning |
|---|---|
| `--state XX` | state; the model option above on `workflow`/`complete`/`confidence`/`predict-single`, location-only on `geojson2csv`/`process` |
| `--climate-zone ZONE` | ASHRAE IECC 2004 climate zone, e.g. `4A` (case-insensitive) |
| `--latitude DEG --longitude DEG` | site coordinates, both or neither, resolved through the nearest ResStock weather station |

* **GeoJSON input.** Every building gets the same project location, read only from the weather filename
  (`USA_<ST>_…`, in `project` or on the Site Origin) and the *Site Origin* point, whose nearest ResStock weather
  station gives the climate zones. The options override both; `project.climate_zone` and building footprints are not
  used.
* **CSV input.** The options apply to every row. A value already in the CSV is never overwritten: a differing value is
  kept and reported.
* **Invalid values.** An invalid option value exits with code 2.

Commands print one location line, and `run_manifest.json` records it under `location`:

```
Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state) · nearest station Baltimore Blt Washngtn IntL (16 km)
```

## `usg geojson2csv`

```bash
usg geojson2csv -i buildings.json -o buildings.csv [-d BUNDLE_DIR] [--state XX] [--climate-zone ZONE]
                 [--latitude DEG --longitude DEG] [-v]
```

Maps residential buildings (`Single-Family Detached/Attached`, `Single-Family`, `Multifamily`, `Mobile Home`) to
ResStock parameters:

* `Geometry Building Type RECS`, `Vintage`, `Vintage ACS`, `Geometry Floor Area`, `Geometry Floor Area Bin`,
  `Geometry Stories`, `Bedrooms`, `Geometry Building Number Units SFA`, `Geometry Foundation Type`,
  `Geometry Attic Type`, `HVAC Cooling Type`, `Heating Fuel`;
* plus the project's location columns: `State`, `ASHRAE IECC Climate Zone 2004`, `Building America Climate Zone`,
  `Census Division`, and `PUMA Metro Status` only when given.

Unmappable values are left empty for the model. With `-d`, every column of that bundle's schema is added; a location
label the model has no data for (e.g. `AK`) is left empty. No model is loaded.

## `usg predict-single`

```bash
usg predict-single -a "Vintage" "2000s" -a "Bedrooms" "3" -a "State" "CO"
usg predict-single -j building.json -o completed.json --top-k 3
```

`-a KEY VALUE` is repeatable and combined with `-j`. Without `-o`, prints each attribute (`[✓]` = provided) with the
top-k alternatives and their probabilities, and each predicted value's confidence (`conf 0.84 high`) followed by the
inputs that would help most. With `-o`, writes the completed attributes plus a `_meta` object (`model_id`, `imputed`,
`top_k`, `seed`, `n_samples`, `confidence`) and `<output stem>_confidence_report.{json,html}`.
`--no-confidence-report` turns confidence off. A `State` attribute selects the state model when `--state auto`.

Location options fill the building's location columns without overwriting attributes you gave. So do `latitude` and
`longitude` attributes when no `State` is given. For example:
`usg predict-single -a Vintage 1990s --latitude 39.29 --longitude -76.61`.

## `usg complete`

```bash
usg complete -i incomplete.csv -o completed.csv [--id-col Building] [--no-confidence-report]
              [--confidence-method native|ensemble|both] [--passes 24] [--drop-prob 0.35]
              [--report-dir DIR] [--report-format json,html] [model options] [-v]
```

Writes `Building` + every schema imputable column + required conditioning inputs present in the CSV (e.g. `State`) +
other ResStock parameters from the input that post-processing needs. With `--state auto` and a multi-state `State`
column, each state is completed with its own model.

Location handling (see [location.md](location.md#csv-and-single-building-input)):

* When the CSV has `latitude`/`longitude` columns (any case) and no `State` values, each row's location columns are
  resolved from its coordinates.
* `--state/--climate-zone/--latitude/--longitude` apply to every row.
* Cells that already hold a value are never overwritten; differences are reported.
* The coordinate columns are not passed to the model.

Also writes:

* `confidence_report.json` and `confidence_report.html` (in `--report-dir`, default next to the output) unless
  `--no-confidence-report`; see [confidence_reports.md](confidence_reports.md)
* `<output stem>_input_issues.csv` when values were not valid options or columns were not recognized
* `run_manifest.json` next to the output (models with checksums, seed, samples, package and torch versions, warnings)

## `usg confidence`

`complete` with reports always on, plus `--sensitivity FILE`, `--calibration FILE` and `--exclude ATTR` (repeatable).
See [confidence_reports.md](confidence_reports.md).

## `usg process`

```bash
usg process -i completed.csv -o uo_buildstock_mapping.csv [-g buildings.json] [--state XX] [--climate-zone ZONE]
             [--latitude DEG --longitude DEG]
             [--options-lookup FILE] [--consistency-rules FILE] [--no-reports] [--reports-dir DIR] [-d BUNDLE] [-v]
```

1. **Fill** every column listed in `resources/postprocessor/required_columns.json` that is missing, and drop the
   columns URBANopt-BuildStock does not use. `Feature ID` (a copy of `Building`) becomes the first column.
   * **Derived from ResStock first.** The model bundle's `postprocess_derivations.json` (from `workflow`'s model, or
     `-d BUNDLE`, else the embedded national G-NN) is learned from the same ResStock release. For each column the
     model does not predict, it tries in order:
     1. **Derive the value** from known or predicted attributes when ResStock determines it for at least 99% of
        homes. Examples: `Geometry Story Bin` from the building type and height; `AHS Region`, `County`, `PUMA`,
        `ISO RTO Region` and `REEDS Balancing Area` from `County and PUMA`; the emissions region from `REEDS
        Balancing Area`.
     2. **Choose an anchor** otherwise:
        * `County and PUMA` comes from the Site Origin's nearest 2010 Census tract. Without coordinates it is the most
          common one for the state and weather station.
        * `Income` is the most common one for the occupants and that location.

        The columns that depend on an anchor are then derived from it, so all location (and income) attributes stay
        consistent.
     3. **Use the most common value** for the most informative available attributes; the per-state default when
        nothing else informs a column; the national most common value last.
   * Values you provided are never overwritten, and values that are not valid options are skipped.
   * Columns absent from that ResStock release get static defaults, preferring options that mean "absent" (`None`,
     `Never`, `No`, `Not Available`).
   * Static defaults are state-, climate- and row-aware.
   * The location comes from `-g` and the location options (the options win). It supplies the state, the climate
     zones and the census division.
   * Location columns missing from the CSV, such as `Building America Climate Zone`, take the resolved values.
2. **Validate** values against `options_lookup.tsv`, replacing invalid ones with defaults →
   `schema_validation_report.csv`.
3. **Enforce** cross-field consistency rules → `consistency_audit_report.csv`.
4. **Report what changed.** The report compares the file before and after post-processing:
   * `postprocess_changes.csv` has one row per changed value: building, attribute, before → after, source and the
     step or rule responsible.
   * `postprocess_report.{json,html}` summarize the run:
     * how many predicted values and how many of your inputs changed;
     * changes by rule and by attribute;
     * rule violations still in the final file;
     * the required columns that were added and the default each one received.

   `workflow` knows which values the model predicted and labels each value as *predicted*, *your input* or *project
   location*. `process` alone cannot tell them apart, so it counts every value as input. The terminal prints a one-line
   summary. A run is flagged for review when more than 5% of predicted values, or any of your inputs, were changed.

## `usg workflow`

```bash
usg workflow -i buildings.json [-o uo_buildstock_mapping.csv] [--skip-postprocess] [--keep-intermediate]
              [--no-reports] [--no-confidence-report] [--confidence-method M] [--passes N] [--drop-prob P]
              [--report-format json,html] [model options] [-v]
```

Steps:

1. Resolve the project location from the weather filename and the Site Origin point, and print it, e.g.
   `Location: MD (weather file) · 4A (nearest station) · …`. Location options override the GeoJSON; see
   [location.md](location.md).
2. Select the model and print why.
3. Convert the GeoJSON to model inputs, including every known location column the model schema has.
4. Impute.
5. Post-process with the project location.
6. Write the confidence report, reconciled with the final file, unless `--no-confidence-report`. A one-line summary
   and the inputs that would help most are printed.
7. Write `run_manifest.json`, including the `location` record.

Intermediate files
(`<stem>_intermediate.csv`, `<stem>_inference_output.csv`) are written next to the output and removed unless
`--keep-intermediate`. The default output is `<geojson stem>_uo_buildstock_mapping.csv` next to the GeoJSON.

## `usg info`

```bash
usg info [model options] [--export attributes.json]
```

Shows post-processor resources, the model that would be selected (and why), its region, schema summary, whether it
ships calibration/sensitivity tables and its headline metrics. `--export` writes the attribute lists.

## `usg models`

See [models.md](models.md#usg-models): `list`, `download`, `info`, `verify`, `clear-cache`, `path`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | unexpected error |
| 2 | invalid command-line usage |
| 3 | model selection, registry or download error |
| 4 | bundle missing, corrupted or incompatible |
| 5 | invalid input (unreadable file, unknown state, invalid labels with `--unknown-labels error`, missing ID column) |

## Environment variables

`USG_CACHE_DIR`, `USG_REGISTRY`, `USG_OFFLINE`, `USG_NUM_THREADS`, `USG_DEBUG` — see
[models.md](models.md#environment-variables).

## Examples

```bash
# URBANopt project, model chosen from the project's weather file
usg workflow -i urban_district.json -o uo_buildstock_mapping.csv

# Step by step
usg geojson2csv -i district.json -o district_raw.csv
usg complete -i district_raw.csv -o district_completed.csv --model tabddpm --samples 32
usg process -i district_completed.csv -o uo_buildstock_mapping.csv -g district.json

# Reproducible, offline, pinned model
usg workflow -i district.json --model-id gnn-national-2024.2-v1 --offline --seed 42
```
