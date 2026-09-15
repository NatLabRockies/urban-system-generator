# Upgrading from urban-system-generator 0.1.x to 1.0

Version 1.0 keeps the PyPI name (`urban-system-generator`), the command (`usg`) and the Python package (`usg`). The
single TensorFlow model is replaced by versioned, checksum-verified PyTorch model bundles (national G-NN and TabDDPM),
which are selected automatically and ship inside the package.

## Install

```bash
pip install -U urban-system-generator    # Python 3.10+
```

TensorFlow, Keras and scikit-learn are no longer needed; the package depends on PyTorch (CPU is enough).

## Commands

The command names are unchanged:

| Command | What changed in 1.0 |
|---|---|
| `usg geojson2csv -i … -o …` | adds the project's location columns; `-d BUNDLE` adds every schema column |
| `usg predict-single -a K V …` | shows confidence and top-k alternatives; `-o` JSON gains a `_meta` object and confidence reports |
| `usg complete -i … -o …` | writes confidence reports, `run_manifest.json` and `*_input_issues.csv` |
| `usg confidence …` | default method is now `native` (calibrated); `--confidence-method ensemble` is the 0.1 behaviour |
| `usg process -i … -o … -g …` | derives non-predicted columns from ResStock; writes a change report |
| `usg workflow -i … -o …` | model chosen automatically; confidence and change reports by default |
| `usg info [--export]` | plus `usg models list/info/verify/download/path/clear-cache` |
| `-d/--model-dir <legacy model folder>` | now a bundle directory; legacy `.keras`/`.pkl` folders are not supported |
| `-m/-c/-n/-e` (model, scalers, encoding) | removed; use `--model`, `--model-id`, `--model-dir` |

New options wherever a model runs: `--model`, `--state`, `--model-id`, `--samples`, `--seed`, `--device`, `--threads`,
`--offline`, `--no-fallback`, `--registry`, `--unknown-labels`, `--climate-zone`, `--latitude/--longitude`.

Exit codes are now specific (3 selection/download, 4 bundle, 5 input) instead of always 1.

## Python API

| 0.1.x | 1.0 |
|---|---|
| `usg.inference.USGInference(model_path, cat_scaler_path, num_scaler_path, encoding_dict_path)` | `usg.Bundle.open(path).load_imputer()` or `usg.pipeline.load_model(usg.resolve(usg.ModelRequest(state="MD")))` |
| `inference.predict_missing_single(known)` | `imputer.impute(pd.DataFrame([known])).values.iloc[0]` |
| `inference.process_buildings_batch(in_csv, out_csv)` | `usg.pipeline.run_complete(in_csv, out_csv, ModelRequest())` or `complete_dataframe(df, imputer)` |
| `process_buildings_batch_with_confidence(...)` | `complete_dataframe(df, imputer, confidence=ConfidenceOptions())` |
| `usg.confidence.ConfidenceEstimator(inference, ...)` | `usg.confidence.ConfidenceEstimator(imputer, ...)`; `confidence_for_building(known)` still exists |
| `usg.geojson_processor.GeoJSONProcessor` | `usg.geojson.GeoJSONProcessor`, plus `geojson_to_dataframe(path, schema, context)` and `read_project_context(path)` |
| `usg.postprocessor.USGPostProcessor(options, rules)` | `usg.postprocessor.USGPostProcessor()` (paths optional); `process(..., context=ProjectContext)` |
| `usg.postprocessor.read_project_state/read_project_climate_zone` | unchanged (re-exported from `usg.geojson`) |
| `usg.report.write_reports/reconcile_with_final` | unchanged names in `usg.report` |
| `scripts/build_calibration.py` | calibration tables ship inside the model bundles |
| `usg/resources/model_attributes.py` | the bundle's `schema.json` (`bundle.schema.names`) |

## Behaviour changes

**GeoJSON mapping.** Values the old mapper invented are now left empty so the model infers them:

* An unknown `building_type` gives no value (was `Other Category`, not a valid option). `Mobile Home` buildings are
  now processed.
* An unknown `foundation_type` gives no value (was `Other Category`). An unknown `attic_type` gives no value (was
  `None`). `flat roof` maps to `None`.
* An unknown `heating_system_fuel_type` gives no value (was `Other Fuel`). `other` maps to `Other Fuel`, and `none` to
  `None`.
* `system_type`: every URBANopt residential system maps to a cooling type (`… and no cooling` → `None`,
  `mini-split heat pump` → `Non-Ducted Heat Pump`, …). Unknown systems give no value (was `None`).
* `Geometry Building Number Units SFA` is the unit count only for attached homes (`None` for other types).
* `Bedrooms` is clipped to 1–5 (5 means five or more). `Geometry Stories` values that are not options are left empty.
* Location columns come from the project's weather filename and Site Origin, and condition the model.

Every mapped value is guaranteed to be a valid `options_lookup.tsv` option.

**Post-processing.** The hard-coded `MISSING_PARAMETERS`/`EXTRA_COLS_TO_DROP` lists are replaced by
`resources/postprocessor/required_columns.json`:
* missing required columns are filled, and all other columns except `Building`/`Feature ID` are dropped;
* columns the model does not predict are derived from ResStock, consistently with the known and predicted attributes;
* a change report records every change.

CSVs are read so that the option `None` is never turned into a missing value.

**Workflow.**
* Intermediate CSVs are written next to the output, not next to the GeoJSON.
* A `run_manifest.json` records the models (with checksums), seed, sample count and versions.

**Confidence.**
* The default method is `native`: calibrated confidence models shipped in each bundle, with no extra model passes.
* Reports are written by default.

**Threads.** torch runs single-threaded by default, as the old TensorFlow configuration did, so URBANopt's Ruby
subprocess cannot hang. Raise it with `--threads` or `USG_NUM_THREADS`.
