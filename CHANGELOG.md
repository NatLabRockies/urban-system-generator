# Changelog

## 1.1.1

### Fixed
- **An attic always has an insulated ceiling.** A vented or unvented attic with
  `Insulation Ceiling = None` reaches HPXML as an R-0 assembly, which it refuses
  (`AssemblyEffectiveRValue: The value '0.0' must be greater than '0'`), failing that building's
  simulation. The pairing does not occur in the stock — 0 of 216,171 vented-attic buildings and 0 of
  18,011 unvented ones — so a consistency rule now repairs it. Found when the TabDDPM model produced it
  on a real district and lost one building of 77.

### Changed
- `--samples` is documented for what it does: it sets the draws behind the confidence report and does
  not change the value written to the output. Use `--decode` for that. It is also the way to make
  TabDDPM faster, which matters more than the help text implied: TabDDPM denoises over 100 sequential
  steps per draw, so 32 draws cost 3,200 network passes per building against the G-NN's one.
- The README and docs refer to the models that actually ship (`-v2`), and the sample output matches a
  real 1.1 run. `docs/models.md` gains a section on choosing between the two models and what each costs.
- Tests needing URBANopt, OpenStudio or EnergyPlus carry a `urbanopt` marker and are excluded from the
  default run, so CI never needs those installs. `RELEASING.md` records the checks a person has to make
  instead, including running the acceptance script against **every** model — TabDDPM shipped in 1.1.0
  having never reached URBANopt, which is how the ceiling defect got out.

## 1.1.0

New national models and a post-processing step that restores the attributes they no longer predict. Output
columns, file formats and the CLI are unchanged; existing commands keep working.

### Added
- **New national models**, `gnn-national-2024.2-v2` and `tabddpm-national-2024.2-v2`. Measured on the
  attributes usg fills in for you, across 4,000 held-out buildings with the same known inputs, the G-NN
  is right **75.4% of the time against the previous model's 72.9%**. Skill — the share of the achievable
  gain over always guessing an attribute's most common value — improves from 0.238 to 0.285, with no
  attribute group worse. The model is also a third the size: 5.5M parameters against 16.6M, and 20.8 MB
  of weights against 66 MB.
- **Restoration of non-predicted attributes.** The models no longer predict attributes that are constant
  across the stock, duplicate another column, or are an exact function of other columns. Post-processing
  puts all of them back by table lookup, verified single-valued on every building in the release, so the
  URBANopt mapping CSV is complete as before. A combination the stock does not contain is reported rather
  than guessed.
- **`--decode argmax|sample|refine`** on `complete`, `workflow` and `predict-single`. `argmax` (the default)
  takes each attribute's most likely value, which is the best guess for one building but repeats it across
  a district. `sample` draws in proportion to the predicted probabilities, so a district keeps the stock's
  real mix of equipment and insulation at some cost in per-building accuracy.
- **Derivation from known inputs.** What a caller's own inputs already determine is filled before the model
  is asked, and a value that contradicts the stock is reported in `*_input_issues.csv` and kept, not
  overwritten.

### Fixed
- A value such as `None` is now rewritten to the spelling `options_lookup.tsv` actually contains (`Void`,
  or `None` where that is the option). URBANopt matches option strings exactly and failed the simulation on
  a mismatch; ten parameters were affected, including `Cooling Setpoint`, `Insulation Wall` and
  `HVAC Has Ducts`.
- Multi-family buildings no longer receive a garage, a garage duct location, or a single-family space
  combination. The stock only ever gives a garage to single-family detached, which is now derived from the
  caller's own building type.
- `number_of_residential_units` from a GeoJSON now reaches `Geometry Building Number Units MF` for
  multi-family buildings.

### Changed
- The per-model accuracy in `usg models list` is not comparable between v1 and v2, because the two are
  scored over different attributes. v1's 0.754 covers 112 attributes including 20 that hold one value
  across the whole stock and are free to predict; v2's 0.672 covers only the 73 it predicts, and excludes
  the 42 it restores afterwards. Neither is the number a caller experiences. Measured over the attributes
  usg actually fills in, accuracy went from 0.729 to 0.754.
- The previous models stay listed and downloadable by id; they are no longer shipped inside the package.

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
