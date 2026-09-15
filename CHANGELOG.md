# Changelog

## 1.0.0

A rewrite of Urban System Generator. These stay the same: the PyPI name `urban-system-generator`, the `usg` command
and the `usg` Python package. See [upgrading from 0.1.x](docs/upgrading_to_1.0.md).

### Added
- **Models.**
  - National G-NN and TabDDPM models (PyTorch), trained on all 549,715 ResStock 2024.2 baseline buildings and shipped
    inside the package, so everything works offline.
  - Choose one with `--model gnn|tabddpm`, `--model-id` or `--model-dir`.
- **Model bundles** (`usg-bundle/v1`) with checksums, self-tests, model cards and a registry. New commands:
  `usg models list|info|verify|download|path|clear-cache`.
- **Location inputs** from the GeoJSON weather filename and Site Origin:
  - the state;
  - the climate zones, from the nearest ResStock weather station;
  - County and PUMA, from the nearest 2010 Census tract.
  - The overrides `--state`, `--climate-zone`, `--latitude/--longitude` and CSV `latitude`/`longitude` columns are
    also supported.
- **Confidence.** A calibrated confidence report for every predicted value (HTML/JSON), written by default.
- **Post-processing.**
  - Required columns the model does not predict are derived from ResStock, consistently with the known and predicted
    attributes (e.g. every location column from one County and PUMA).
  - A change report records every change (`postprocess_report.*`, `postprocess_changes.csv`).
- **Run records.** `run_manifest.json` for reproducibility, and `*_input_issues.csv` for input values that are not
  valid options.
- **Exit codes:** 3 for model selection or download, 4 for bundle, 5 for invalid input.

### Changed
- Requires Python 3.10 or newer and PyTorch. TensorFlow, Keras and scikit-learn are no longer needed.
- GeoJSON mapping leaves unknown values empty for the model to infer, instead of inventing them.
- Post-processing required columns come from `resources/postprocessor/required_columns.json`.
- Static post-processing defaults prefer "absent" options (`None`, `Never`, `No`) over the alphabetically first
  option.

### Removed
- The single Keras model with its scalers and encoding files, and the `-m/-c/-n/-e` options that pointed to them.
