# Confidence and validation reports

urban-system-generator fills in missing building characteristics. Confidence reports tell you **how much to trust each filled
value for your run**, given the inputs you actually provided for each building. They never change the values.

## Quick start

```bash
# End-to-end: the report matches the FINAL post-processed file
usg workflow -i buildings.json -o uo_buildstock_mapping.csv --model gnn

# Complete a CSV and write reports next to the output
usg complete -i incomplete.csv -o completed.csv

# Complete with an explicit calibration table or excluded attributes
usg confidence -i incomplete.csv -o completed.csv --calibration table.json --report-format json,html
```

`workflow`, `complete` and `confidence` always write reports; `predict-single` prints confidence and writes reports
with `-o`. Pass `--no-confidence-report` to skip them.

Files written (in `--report-dir`, default the output directory):

* `confidence_report.json` — machine-readable, per building and attribute
* `confidence_report.html` — self-contained visual report
* `confidence_report.md` — Markdown summary (`--report-format md`)

## How confidence is computed (default)

Every national bundle ships **precomputed confidence models** (`confidence_calibration.json`, format
`usg-confidence-calibration/v2`) fit by the model training pipeline on held-out ResStock buildings that the model never trained on.
Building a report costs a few matrix multiplications — no extra model passes — so it adds well under a second to a
run.

For each predicted characteristic there are two small logistic models over the building's **input pattern**:

| Feature | Meaning |
|---|---|
| one bit per GeoJSON fact | building type, vintage, floor area, bedrooms, stories, foundation, heating fuel, … (a fact counts as known when any of its columns is known, e.g. *Vintage* = Vintage or Vintage ACS) |
| `location` | the state is known (from the weather file, coordinates or `--state`) |
| `log1p_other_known` | how many other characteristics you supplied |
| pairwise fact products | lets facts reinforce each other (e.g. building type × floor area) |

1. **Expected accuracy** — `P(correct | input pattern)`: how often this characteristic is right for buildings where
   the user knew the same facts.
2. **Confidence** — `P(correct | model probability of the delivered value, input pattern)`: the calibrated chance
   that *this building's* value is right. The model probability is the G-NN softmax (temperature-scaled) or, for
   TabDDPM, how often the delivered value appeared among the samples.

So confidence changes from run to run and from building to building: it rises when you supply more (and more
informative) facts, and when the model strongly favours the value it filled.

Toggling one fact bit in the expected-accuracy model gives:

* **drivers** — known facts that raise this attribute's expected accuracy the most (`effect`, in accuracy units);
* **suggestions** — missing facts that would raise it the most (`gain`).

Per building, `suggested_facts` ranks missing facts by their mean gain over all predicted attributes; the run summary
aggregates these into the facts that would help the project most.

**Reliability.** Each table records how well it is calibrated on held-out buildings (`validation`): expected
calibration error (ECE) of the confidence versus the raw model probability, and Brier scores. The HTML report quotes
it, e.g. *"stated confidence differed from observed accuracy by 0.8 points on average"*.

Bands: `high` ≥ 0.80, `medium` ≥ 0.50, otherwise `low` (tables may override them with `bands`).

## Methods (`--confidence-method`)

| Method | Signal | Cost | Notes |
|---|---|---|---|
| `native` (default) | precomputed confidence models (above); without a v2 table: the model probability blended with coverage and a v1 calibration table | free | TabDDPM: use `--samples 16` or more (a warning is printed otherwise). |
| `ensemble` | **evidence-masking stability**: re-impute `--passes` times, each time hiding every known input with probability `--drop-prob`; `agreement` = share of passes that reproduce the reported value | `passes ×` imputation | For research; conditioning inputs are never hidden. |
| `both` | mean of probability and agreement | as `ensemble` | |

Numeric targets report `ensemble_mean`, `ensemble_std` (native: from the 5–95 % interval) and `stability`
(`1 − std/|mean|`).

## Per-attribute record

```json
{
  "value": "Natural Gas", "type": "categorical", "method": "native",
  "confidence": 0.842, "band": "high",
  "expected_accuracy": 0.78, "baseline_accuracy": 0.61, "probability": 0.91,
  "margin": 0.84, "entropy": 0.21,
  "drivers": [{"fact": "Heating Fuel", "effect": 0.21}],
  "suggestions": [{"fact": "Vintage", "gain": 0.03}],
  "ensemble_top": "Natural Gas",
  "ensemble_alternatives": [{"value": "Electricity", "share": 0.07}],
  "coverage": null, "calibrated_accuracy": 0.78
}
```

`baseline_accuracy` is how often the most common value is right on the held-out buildings. `calibrated_accuracy`
repeats `expected_accuracy` for readers of older reports. Characteristics without a v2 model (rare) fall back to the
model probability.

Each building record also lists `known_inputs` (your inputs), `context_inputs` (filled from the GeoJSON project or
location), `n_known`, `method`, `model_id`, `known_facts`, `missing_facts`, `suggested_facts`
(`{fact, mean_gain, n_attributes_improved}`) and a `summary` (including `mean_expected_accuracy`).

The report JSON has `meta` (model, inputs, run settings), `validation` (the table's held-out reliability),
`run_summary` (`mean_building_confidence`, `mean_expected_accuracy`, band counts, weakest/strongest attributes,
`facts` and `suggested_facts`) and `buildings`.

## Reports aligned to the final output

The post-processor adds, drops and rewrites columns. `usg workflow` reconciles the inference-stage records with the
final file: only final columns appear, values are the final values, and each cell is labeled by source:

* **predicted by model** (`model_inferred`) — carries the confidence score
* **predicted, then adjusted by postprocessing** (`model_inferred_adjusted`) — a schema or consistency rule
  overrode the model's value; the report shows the final value and `model_value`
* **provided by you** (`user_provided`) — a known input
* **taken from the GeoJSON project** (`project_context`) — e.g. `State` from the weather file, climate zone
* **filled by postprocessing** (`postprocess_default`) — a deterministic default, no confidence score

Run-level confidence averages model-predicted cells only.

## Calibration table format (v2)

```json
{
  "schema": "usg-confidence-calibration/v2",
  "model_type": "gnn",
  "facts": [{"name": "Vintage", "columns": ["Vintage", "Vintage ACS"]}, "..."],
  "location_column": "State",
  "feature_names": ["fact:Vintage", "...", "location", "log1p_other_known", "pair:Vintage|Geometry Floor Area", "..."],
  "bands": {"high": 0.8, "medium": 0.5},
  "sampling": {"n_pairs": 60000, "n_buildings": 54000, "fact_known_p": [0.5, 0.85], "location_p": 0.85},
  "validation": {"n_validation_cells": 1100000, "ece_model_probability": 0.012, "ece_confidence": 0.008,
                 "ece_expected_accuracy": 0.011, "brier_model_probability": 0.14, "brier_confidence": 0.13},
  "by_target": {
    "Heating Fuel": {
      "type": "categorical", "baseline": 0.61, "accuracy": 0.83, "n": 52000,
      "expected": {"intercept": 0.4, "coef": [0.1, "..."]},
      "confidence": {"intercept": -0.2, "prob_coef": 1.1, "coef": [0.05, "..."]}
    },
    "Rare Attribute": {"type": "categorical", "baseline": 0.9, "expected": {"constant": 0.97}, "confidence": {"constant": 0.97}}
  }
}
```

* `coef` follows `feature_names` exactly; the feature construction lives in `usg.confidence.model` and is
  shared with the model training pipeline, so fitting and inference cannot drift.
* A `{"constant": p}` model is used when a characteristic had too few (or single-outcome) held-out cells.
* Legacy `usg-confidence-calibration/v1` tables (`by_n_known`, `key_drivers`, …) are still accepted; they use the
  blended score: `confidence = 0.7 × calibrated + 0.3 × probability` (or `0.8 × probability + 0.2 × coverage`).

## Sensitivity table format

`sensitivity.json` (or `--sensitivity`), derived from the expected-accuracy models: the share of a characteristic's
expected accuracy lost when each fact is removed from a fully specified GeoJSON building.

```json
{"schema": "usg-sensitivity/v1",
 "inputs": ["Geometry Building Type RECS", "Vintage", "Geometry Floor Area"],
 "by_target": {"Heating Fuel": {"Vintage": 0.4, "Geometry Building Type RECS": 0.6}}}
```

It feeds the model card's recommended inputs and the legacy coverage signal.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--confidence-method` | `native` | `native`, `ensemble` or `both` |
| `--report-format` | `json,html` | any of `json`, `html`, `md` |
| `--passes` | `24` | ensemble passes (higher = steadier, slower) |
| `--drop-prob` | `0.35` | probability of hiding each known input per pass |
| `--samples` | bundle default | samples per building (TabDDPM confidence wants ≥ 16) |
| `--sensitivity` | bundle's | sensitivity table (`usg confidence` only) |
| `--calibration` | bundle's | calibration table (`usg confidence` only) |
| `--exclude` | defaults | attribute(s) to omit (repeatable; replaces the default exclusions) |
| `--report-dir` | output dir | where to write reports (`complete`/`confidence`) |

## Excluded targets

By default reports omit simulation outputs produced downstream by URBANopt rather than inferred building
characteristics (`site_energy.total.energy_consumption.kwh`, see
`usg.confidence.DEFAULT_EXCLUDED_TARGETS`). Conditioning columns are inputs and are never scored.

## Static model card

Each bundle's `MODEL_CARD.md` and `metrics.json` hold the global, held-out evaluation (per-attribute accuracy, skill
over baselines, the known-inputs curve). Per-run reports answer "should I trust *this* fill?"; the model card answers
"how good is the model overall?". See `usg models info <id> --card`.
