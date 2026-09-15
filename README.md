# Urban System Generator (USG)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI](https://img.shields.io/pypi/v/urban-system-generator.svg)](https://pypi.org/project/urban-system-generator/)

**NLR Software Record:** SWR 25-36

Urban System Generator (USG) completes missing residential building characteristics for
[URBANopt](https://docs.urbanopt.net/) projects with machine-learning models trained on NREL ResStock. Give it an
URBANopt GeoJSON, or a CSV with whatever you know about each building. It returns a complete, validated
URBANopt-BuildStock mapping CSV, with a confidence report for every predicted value and a report of everything
post-processing changed.

**Version 1.0** is a rewrite of the 0.1.x releases. The package name and the `usg` command are unchanged. See
[upgrading from 0.1.x](docs/upgrading_to_1.0.md).

## Features

* **Two pretrained national models:**
  * **G-NN** is fast and gives the most accurate value per building.
  * **TabDDPM**, a tabular diffusion model, gives realistic variety across a building stock.

  Both were trained on all 549,715 ResStock 2024.2 buildings and ship inside the package, so everything works
  offline.
* **Location-aware.** The state comes from the weather filename, the climate zones from the Site Origin's nearest
  ResStock weather station, and County and PUMA from the Site Origin's nearest Census tract.
* **GeoJSON mapping** of URBANopt building properties to ResStock parameters.
* **Post-processing** to URBANopt-BuildStock:
  * required columns the model does not predict are derived from ResStock, consistently with the known and predicted
    attributes;
  * values are validated against `options_lookup.tsv` and consistency rules are applied;
  * a report records every change.
* **Confidence reports** (HTML/JSON) for every predicted value, calibrated on held-out ResStock buildings.
* **Reproducible runs.** Every run writes `run_manifest.json` with the model checksums, seed and versions.

## Install

```bash
pip install urban-system-generator            # Python 3.10+; CPU PyTorch is enough
usg --version
```

From source:

```bash
git clone https://github.com/NatLabRockies/urban-system-generator.git
cd urban-system-generator
pip install -e ".[dev]"
```

## Quickstart

```bash
# GeoJSON -> URBANopt-ready CSV + confidence and post-processing reports
usg workflow -i buildings.json -o uo_buildstock_mapping.csv

# Same project with the diffusion model
usg workflow -i buildings.json -o uo_buildstock_mapping.csv --model tabddpm --samples 32

# Complete a CSV of partially known buildings
usg complete -i incomplete.csv -o completed.csv

# One building, with confidence and the most likely alternatives
usg predict-single -a "Geometry Building Type RECS" "Single-Family Detached" -a Vintage 1990s -a State CO

# Post-process an existing completed CSV
usg process -i completed.csv -o uo_buildstock_mapping.csv -g buildings.json
```

```
Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state) · nearest station Baltimore Blt Washngtn IntL (16 km) · County and PUMA G2405100, G24000804 (Site Origin, nearest Census tract 0.2 km)
Model: gnn-national-2024.2-v1 — state MD from weather file (USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw); model type gnn (default for MD); the national gnn model is the recommended model for MD; gnn-national-2024.2-v1 (embedded)
Converted 77 residential building(s) from buildings.json
...
Confidence: mean 0.76, expected accuracy 0.75 — 4399 high / 1345 medium / 1956 low predicted values
✓ Confidence report: confidence_report.html
Post-processing changed 0 of 7,700 predicted values (0.0%) and 0 of your 924 input values; filled 0 empty cells and added 45 required columns the model does not predict
✓ Post-processing report: postprocess_report.html
✓ Workflow complete!
```

Outputs next to the CSV:
- the final mapping;
- `confidence_report.{json,html}`;
- `postprocess_report.{json,html}` and `postprocess_changes.csv`: every value post-processing changed, and why;
- `run_manifest.json`;
- `<name>_input_issues.csv` if some input values were not valid options;
- `schema_validation_report.csv` and `consistency_audit_report.csv`, when there are issues.

A run is flagged for review when post-processing changes more than 5% of predicted values or any of your inputs.

Full reference: [docs/cli_usage.md](docs/cli_usage.md).

## Models

| Model | `--model` | Best for |
|---|---|---|
| `gnn-national-2024.2-v1` (default) | `gnn` | the most accurate value for each building |
| `tabddpm-national-2024.2-v1` | `tabddpm` | realistic variety across a building stock (draws from the joint distribution); use `--samples 32` |

* **Training data:** all 549,715 ResStock 2024.2 baseline buildings (48 states + DC).
* **Inputs:** the building's location (state, climate zones, census division), so one model serves every state.
* **Evaluation:** `usg models list` shows the headline test results, and `usg models info <model id> --card` shows
  each model's full held-out evaluation.

## Confidence reports

Every `workflow` and `complete` run writes `confidence_report.json` and `confidence_report.html`. Skip them with
`--no-confidence-report`. For each predicted value, the report gives:

* the chance it is right, given what you supplied for that building;
* the expected accuracy for buildings with the same inputs;
* the inputs that drove it;
* the missing GeoJSON properties (e.g. `year_built`, `system_type`) that would raise it most.

The confidence comes from small models precomputed on held-out buildings and shipped inside each model bundle, so it
costs well under a second per 100 buildings. See [docs/confidence_reports.md](docs/confidence_reports.md).

## Model selection

| You pass | Model used |
|---|---|
| nothing | the default model (G-NN) recommended for the project's state |
| `--model tabddpm` | that model type |
| `--model-id tabddpm-national-2024.2-v1` | exactly that model |
| `--model-dir path/to/bundle` | that local bundle |

The selected model and the reason are always printed and recorded in `run_manifest.json`. Details:
[docs/models.md](docs/models.md).

## Location

The models take five location inputs. Every building of a GeoJSON project gets the same location, read only from the
weather filename and the *Site Origin* point:

| Input | Source |
|---|---|
| `State` | `--state XX` > weather filename (`USA_MD_…` → `MD`) > the Site Origin's nearest ResStock weather station |
| `ASHRAE IECC Climate Zone 2004` | `--climate-zone` > the Site Origin's nearest station > the state's zone when ≥ 90% of its homes share it |
| `Building America Climate Zone` | implied by the ASHRAE zone when the data allows > the Site Origin's nearest station > the state's zone |
| `Census Division` | the state |
| `PUMA Metro Status` | only when you provide it (CSV column or `-a`) |

**County and PUMA** is used by post-processing and is not a model input. It comes from the 2010 Census tract nearest
to the Site Origin, within the project's state; the tract table is packaged, so no network access is needed. County,
PUMA, City, AHS Region, ISO/RTO and the other location columns are then derived from it.

For CSV input, `latitude`/`longitude` columns locate each row when the CSV has no `State` values. Details:
[docs/location.md](docs/location.md).

## Python API

```python
from usg import ModelRequest, resolve, run_workflow
from usg.pipeline import ConfidenceOptions, complete_dataframe, load_model
import pandas as pd

result = run_workflow("buildings.json", "uo_buildstock_mapping.csv", ModelRequest(), confidence_report=True)

imputer = load_model(resolve(ModelRequest(state="CO", model_type="gnn")))
df = pd.DataFrame([{"Building": "b1", "State": "CO", "Vintage": "1990s"}])
completion = complete_dataframe(df, imputer, confidence=ConfidenceOptions())
print(completion.data.T, completion.records[0]["confidence"]["attributes"]["Heating Fuel"])
print(imputer.impute(df).top_k("Heating Fuel", 3))
```

## URBANopt integration

URBANopt can call `usg workflow` from Ruby:
* inference is single-threaded by default to avoid thread-pool hangs in subprocesses (change it with `--threads` or
  `USG_NUM_THREADS`);
* exit codes distinguish model selection (3), bundle (4) and input (5) errors;
* model downloads use file locks, so parallel runs share the cache safely.

## Documentation

* [CLI reference](docs/cli_usage.md)
* [Models, registry, selection, cache](docs/models.md)
* [Location inputs: precedence, sources, County and PUMA](docs/location.md)
* [Confidence reports and calibration formats](docs/confidence_reports.md)
* [Bundle format `usg-bundle/v1`](docs/bundle_format.md)
* [Upgrading from 0.1.x](docs/upgrading_to_1.0.md)
* [Changelog](CHANGELOG.md)

## Development

```bash
pip install -e ".[dev]"
pytest                                   # unit, CLI and end-to-end tests (tiny bundles are built at test time)
ruff check src tests
python scripts/fetch_embedded_bundles.py # copy the registry's embedded model bundles before building a wheel
```

Models are trained, calibrated and exported with NLR's model training pipeline.

## Authors & Contributors

**Lead Developer:** Rawad El Kontar

**Organization:** National Laboratory of the Rockies (NLR)

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, add tests for your changes and open a
pull request.

## License

Copyright (c) 2025-2026, Alliance for Energy Innovation, LLC

This project is licensed under the BSD 3-Clause License; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights are
trained on NREL ResStock data (CC-BY-4.0) and distributed under the license stated in each bundle's manifest and model
card.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{urban_system_generator,
  title={Urban System Generator},
  author={El Kontar, Rawad},
  organization={National Laboratory of the Rockies},
  version={1.0.0},
  year={2026},
  url={https://github.com/NatLabRockies/urban-system-generator}
}
```

## Acknowledgments

This work was authored by the National Laboratory of the Rockies (NLR) for the U.S. Department of Energy (DOE),
operated under Contract No. DE-AC36-08GO28308. Funding provided by NLR's Laboratory Directed Research and Development
program, and the DOE Office of Science. The views expressed in the article do not necessarily represent the views of
the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for
publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to
publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

## Contact

For questions or support, please contact Rawad El Kontar (rawad.elkontar@nlr.gov) or open an issue on GitHub.

---

**NLR Software Record:** SWR 25-36
