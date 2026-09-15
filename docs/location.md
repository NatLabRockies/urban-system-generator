# Location inputs

Every model conditions on five location columns. They are inputs only: the model never imputes them, and an unknown
one is simply masked. A correct value sharpens the predictions, and a wrong one biases them.

| Column | Example |
|---|---|
| `State` | `MD` |
| `ASHRAE IECC Climate Zone 2004` | `4A` |
| `Building America Climate Zone` | `Mixed-Humid` |
| `Census Division` | `South Atlantic` |
| `PUMA Metro Status` | `In metro area, principal city` |

`usg.location.resolve_location` fills them. It only ever produces ResStock labels, and it leaves a column
unknown rather than guess.

## GeoJSON projects

`usg workflow`, `usg geojson2csv` and `usg process -g` give **every building of a GeoJSON file the same project
location**. Only two things in the file are read:

1. **The weather filename** `USA_<ST>_<city>…epw`, from `project.weather_filename` or else the *Site Origin*
   feature's `weather_filename`. It gives the state: `USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw` → `MD`.
2. **The Site Origin point**: the Point geometry of the feature whose `type` (or `name`) is `Site Origin`. Its nearest
   ResStock weather station gives the climate zones, and also the state when there is no weather filename. Its nearest
   2010 Census tract gives the project's `County and PUMA` (see [County and PUMA](#county-and-puma)).

Nothing else in the file locates the project. `project.climate_zone`, other project coordinates and the building
footprints are not used.

## Precedence

For each column, the first source that gives a value wins:

| Column | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `State` | `--state XX` | weather filename | Site Origin's nearest station, share ≥ 0.6 | — |
| `ASHRAE IECC Climate Zone 2004` | `--climate-zone` | Site Origin's nearest station, share ≥ 0.6 | from state, share ≥ 0.9 | — |
| `Building America Climate Zone` | explicit (CSV, API) | from the ASHRAE zone, share ≥ 0.99 | Site Origin's nearest station, share ≥ 0.6 | from state, share ≥ 0.9 |
| `Census Division` | from state | — | — | — |
| `PUMA Metro Status` | explicit only (CSV, `-a`, API) | — | — | — |

The explicit options (`--state`, `--climate-zone`, `--latitude/--longitude`) are opt-in overrides;
`--latitude/--longitude` replace the Site Origin point.

What each source means:

* **Nearest station**: the ResStock weather station closest to the coordinates (see below). Its value is used only
  when it covers at least 60% of the homes simulated with that station's weather.
* **From state**:
  * Census Division always follows the state.
  * A climate zone follows the state only when at least 90% of the state's homes share it. KY, for example, is 100%
    `4A` and MO is 93% `4A`; TX has no zone above 55%, so it stays unknown.
* **From the ASHRAE zone**: an ASHRAE zone whose homes fall in one Building America zone at least 99% of the time.
  That holds for 12 of the 15 zones (see [What the data shows](#what-the-data-shows)).

### Consistency guards

* A climate zone from the station must be one of the state's zones. Take a `USA_MD_…` weather filename with a Site
  Origin in Charlotte, NC: the station's `3A` never occurs in MD, so MD's own `4A` is used.
* A derived Building America zone must occur together with the ASHRAE zone in the data. A Denver station (`Cold`) is
  rejected for an explicit `3A`, so the column stays unknown with a warning. An explicit Building America zone that
  never occurs with the ASHRAE zone is kept, with a warning.

### Sources in the output

Each run prints one line. It lists the known values with the label of their source:

```
Location: MD (weather file) · 4A (nearest station) · Mixed-Humid (from 4A) · South Atlantic (from state) · nearest station Baltimore Blt Washngtn IntL (16 km) · County and PUMA G2405100, G24000804 (Site Origin, nearest Census tract 0.2 km)
Location: KY (weather file) · 4A (from state) · Mixed-Humid (from 4A) · East South Central (from state)
```

`run_manifest.json` stores the full record under `location`:
* `summary`, `values`, `sources` and `warnings`;
* `latitude`, `longitude` and `coordinates_source` (`Site Origin` or `explicit`);
* `nearest_station` and `nearest_station_km`;
* `county_and_puma` and `county_and_puma_source`;
* `weather_filename`.

A source is written as `label` or `label (detail)`:

| Source | Meaning |
|---|---|
| `explicit` | CLI option or Python argument |
| `input` | a value already in the CSV row / `-a` attribute |
| `weather file (USA_MD_….epw)`, `Site Origin weather file (…)` | state from the weather filename |
| `nearest station (NAME, 16 km)` | the Site Origin's (or given coordinates') nearest ResStock weather station |
| `from state (KY 100%)` | the state's value, shared by that share of its homes |
| `from 4A` | implied by the ASHRAE zone |
| `from state` | Census Division of the state |

## Coordinates and the station lookup

Distances to the ResStock weather stations are great-circle (haversine). GeoJSON positions are
`[longitude, latitude]`.

A few stations appear twice in ResStock with the same airport and different county assignments: 31 record pairs lie
within 1 km of each other. Records within 2 km of the nearest one are therefore pooled, weighted by their building
counts. Without this, a Site Origin in Manhattan would flip between the NY and NJ records of Central Park (0.1 km
apart).

Coordinates are **not used**, and a warning explains why, when:

* they fall outside the contiguous United States (latitude 24.3–49.5, longitude −125.0 to −66.8). Alaska and Hawaii
  get a hint; pass `--state AK` instead.
* the nearest station is more than **150 km** away. That covers the ocean, the Gulf of Mexico and most of Mexico and
  Canada. For scale, the median distance between neighbouring stations is 43 km and the largest gap is 217 km.
* the Site Origin position is not in degrees, for example a projected coordinate system.

A known limitation: a point just across the Canadian or Mexican border (Toronto, Tijuana) is within 150 km of a US
station and resolves to it.

## County and PUMA

Post-processing needs ResStock's `County and PUMA` (e.g. `G2405100, G24000804`). It is not a model input, but
post-processing derives `County`, `PUMA`, `City`, `AHS Region`, `ISO RTO Region`, `REEDS Balancing Area` and the other
location columns from it, so every location attribute describes the same place.

* **With coordinates** (the Site Origin, or `--latitude/--longitude`), it comes from the nearest **2010 Census tract**.
  * The packaged table `resources/location/county_puma_tracts.npz` (0.5 MB) holds the population centers of all
    74,001 tracts; ResStock uses the same 2010 geography.
  * Tracts are small (about 4,000 people), so the nearest center almost always lies in the site's own county and
    PUMA.
  * No network access is needed.
* **Only tracts of the project's state** are considered, so the result never contradicts the weather file. If no such
  tract is within 25 km, a warning is printed and post-processing falls back as below.
* **Without coordinates**, post-processing uses the most common `County and PUMA` for the state and weather station,
  from the bundle's derivation table.

Example (Baltimore): the Site Origin's nearest tract is 0.2 km away. It gives `G2405100, G24000804`, from which
post-processing derives `MD, Baltimore city` and PUMA `MD, 00804`. The table is rebuilt from two public-domain Census
files with `python scripts/build_county_puma_lookup.py`.

## Post-processing

The workflow's post-processing and `usg process -g` use the same resolved location for their state, climate-zone and
`County and PUMA` defaults, not `project.climate_zone`. Location columns missing from the input CSV, such as `Building
America Climate Zone`, take the resolved values.

## Warnings

* The Site Origin's station state disagrees with the weather filename. The weather filename wins.
* The nearest station is ambiguous and the column had no other source, so the state or zone stays unknown. Downtown
  Washington, DC is an example: Reagan National's homes are MD 44%, DC 35%, VA 22%.
* The coordinates are outside the contiguous US, more than 150 km from every station, or not in degrees.
* The state has no ResStock 2024.2 data (AK, HI). `State` and `Census Division` are kept for post-processing, but the
  model inputs leave any label outside the model's vocabulary empty.
* CSV only:
  * cells that differ from the value the location gives (the cell is kept)
  * invalid latitude/longitude values
  * coordinate columns ignored because the CSV already has `State` values

## `PUMA Metro Status` is never derived

PUMA Metro Status describes a Public Use Microdata Area (about 100,000 people): whether it lies in a metro area and in
its principal city. It changes over a few kilometres, and a weather station's homes mix city cores, suburbs and rural
areas:

* Only 35% of homes belong to stations whose homes all share one status.
* Only 44% belong to stations where one status covers 90%.

Resolving it from coordinates would need PUMA boundaries, which the package does not ship. At state level, only DC, DE
and WY have a single status. A wrong status would push predictions such as building type or vintage confidently in the
wrong direction, while an unknown one lets the model average over the three statuses. So it is filled only when you
provide it: a CSV column, `-a "PUMA Metro Status" …`, or `resolve_location(puma_metro_status=…)`. The per-station and
per-state shares are kept in the lookup for reference.

## CSV and single-building input

For `usg complete`, `usg confidence` and `usg predict-single`:

* Rows are located one by one from `latitude`/`longitude` columns (case-insensitive) when the input has **no `State`
  values**.
* Explicit `--state`, `--climate-zone`, `--latitude` and `--longitude` apply to **every row**. A row's own coordinates
  win over `--latitude/--longitude`.
* **A value already in the CSV is never overwritten.** If it differs from the value the location gives, it is kept and
  reported, for example: `State: 3 row(s) keep their input value although the location gives another one (e.g. row 1:
  'MD' vs 'VA')`. Location values derived for the rest of that row stay consistent with the kept value; a row's own
  `5A`, for instance, gives `Cold`.
* The coordinate columns are consumed: they are not passed to the model and not reported as unknown columns.
* Model selection then follows the filled `State` column. With `--state auto`, rows in different states use their own
  models.
* `run_manifest.json` records under `location`: the mode (`coordinates` or `explicit`), the explicit values, the
  number of cells filled per column, conflicts, the most common location summaries, and warnings.

## Python API

```python
from usg.location import LocationLookup, LocationOverrides, fill_location_columns, resolve_location
from usg.geojson import read_project_context, site_origin_point
from usg.pipeline import run_workflow

info = resolve_location(weather_filename="USA_MD_x.epw", latitude=39.302, longitude=-76.606)
info.values()      # {'State': 'MD', 'ASHRAE IECC Climate Zone 2004': '4A', 'Building America Climate Zone': ..., ...}
info.sources()     # {'State': 'weather file (USA_MD_x.epw)', 'ASHRAE IECC Climate Zone 2004': 'nearest station (...)', ...}
info.summary(), info.warnings, info.nearest_station, info.station_distance_km

LocationLookup.load().nearest_stations(39.29, -76.61, k=3)   # [(Station, km), ...]
read_project_context("buildings.json").conditioning_values()  # the location every building of the project gets
read_project_context("buildings.json", LocationOverrides(climate_zone="5A")).climate_zone
fill_location_columns(df, climate_zone="5A").data             # a building DataFrame, cells never overwritten
run_workflow("buildings.json", "out.csv", location=LocationOverrides(latitude=39.29, longitude=-76.61))
```

`resolve_location` raises `InputValidationError` (CLI exit code 5) for an invalid explicit label or coordinates.

## Data provenance

`src/usg/resources/location/location_lookup.json` has the format `usg-location-lookup/v1` and is about
190 KB.

* **Source.** NREL ResStock 2024 release 2 (`resstock_tmy3_release_2`), national `baseline_metadata_only.parquet`:
  549,715 sample homes. It was downloaded from the OEDI data lake; the URL and SHA-256 are recorded in the file.
* **Build.** The file is built from ResStock metadata by NLR's model training pipeline.
  * It aggregates per weather station (`in.weather_file_city`, 858 stations) and per state (`in.state`, 48 states +
    DC), using the ResStock `weight` column.
  * Coordinates are rounded to 3 decimals.
  * Each share map keeps the most common labels until they cover at least 99% of the weight, with shares rounded to 3
    decimals.
  * Labels are checked against the national bundle vocabulary and `options_lookup.tsv`. At load time,
    `LocationLookup` checks again that every label belongs to the file's own vocabulary.
* **License.** ResStock data is CC-BY-4.0: *Derived from NREL ResStock (End-Use Load Profiles for the U.S. Building
  Stock, 2024 release 2), National Renewable Energy Laboratory, licensed under CC-BY-4.0. Building-weighted aggregates
  per weather station and per state; not an official NREL product.*

| Key | Content |
|---|---|
| `format`, `data_release`, `release_name`, `generated_at`, `generated_by` | identification |
| `license`, `attribution`, `source`, `method` | provenance and aggregation settings |
| `columns` | short key → column (`state`, `ashrae`, `ba`, `division`, `metro`) |
| `summary` | data facts quoted below (not used at inference time) |
| `vocab` | labels per column |
| `ashrae_to_ba` | Building America zone shares per ASHRAE zone |
| `states` | `{XX: {division, n, ashrae, ba, metro}}` |
| `stations` | `[{name, lat, lon, n, state, ashrae, ba, metro}]` |

### What the data shows

**The Building America zone is not a function of the ASHRAE zone overall.** A one-to-one mapping explains 95.2% of
homes. Per zone, though, it holds at 99% or more for 12 of the 15 zones:

| ASHRAE zones | Building America zone |
|---|---|
| 1A, 2A | Hot-Humid |
| 3B | Hot-Dry |
| 3C, 4C | Marine |
| 4A | Mixed-Humid |
| 5A, 5B, 6A, 6B | Cold |
| 7A, 7B | Very Cold |

Only those 12 zones imply a Building America zone. The other three split:

| ASHRAE zone | Split |
|---|---|
| 3A | Mixed-Humid 65.5%, Hot-Humid 34.5% (the Gulf coast) |
| 2B | Hot-Dry 93%, Hot-Humid 7% |
| 4B | Mixed-Dry 98.9%, Cold 1.1% |

**Station purity** is the building-weighted mean share of a station's dominant label:

| Column | Mean dominant share | Homes at stations with a dominant label ≥ 0.6 |
|---|---|---|
| State | 0.953 | 95.9% |
| ASHRAE zone | 0.988 | 99.5% |
| Building America zone | 0.990 | 99.8% |
| PUMA Metro Status | 0.819 | 83.7% |

169 stations serve homes in more than one state.

**States with a single ASHRAE zone:** CT, DC, DE, KY, MA, MT, NE, RI, SC, VA, VT. States where one zone covers at
least 90% of homes are those plus KS, MD, ME, MO, OK and WI.
