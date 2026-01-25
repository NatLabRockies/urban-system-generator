# Urban System Generator CLI Documentation

The Urban System Generator provides a comprehensive command-line interface (CLI) for all its functionality, including ML inference and post-processing for URBANopt-BuildStock compatibility.

## Installation

After installing the package, the `usg` command will be available:

```bash
pip install -e .
```

## Available Commands

### 1. Convert GeoJSON to CSV (`geojson2csv`)

Convert building data from GeoJSON format to CSV with all model attribute columns:

```bash
# Basic usage - creates CSV with all model attributes as columns
usg geojson2csv -i buildings.json -o buildings.csv

# With verbose output
usg geojson2csv -i buildings.json -o buildings.csv --verbose
```

### 2. Predict Single Building (`predict-single`)

Predict missing attributes for a single building:

```bash
# Using command-line attributes
usg predict-single \
  -a "Vintage" "2000s" \
  -a "Bedrooms" "3" \
  -a "Geometry Stories" "2"

# Using JSON input file
usg predict-single -j building_attrs.json -o completed.json
```

### 3. Complete Batch of Buildings (`complete`)

Process multiple buildings to complete missing attributes using ML inference:

```bash
# Basic usage
usg complete -i incomplete.csv -o completed.csv

# With verbose output
usg complete -i incomplete.csv -o completed.csv --verbose
```

### 4. Post-Process for URBANopt (`process`) ⭐ NEW

Post-process ML inference output for URBANopt-BuildStock compatibility. This runs a 3-step pipeline:
1. **Missing Column Filler** - Adds missing columns required by URBANopt-BuildStock
2. **Schema Validator** - Validates and fixes values against `options_lookup.tsv`
3. **Consistency Processor** - Enforces cross-field constraints (HVAC, PV, etc.)

```bash
# Basic usage
usg process -i completed.csv -o uo_buildstock_mapping.csv

# With GeoJSON for climate zone extraction
usg process -i completed.csv -o uo_buildstock_mapping.csv -g buildings.json

# With custom resource files
usg process -i completed.csv -o uo_buildstock_mapping.csv \
  --options-lookup custom_options.tsv \
  --consistency-rules custom_rules.json

# Disable validation reports
usg process -i completed.csv -o uo_buildstock_mapping.csv --no-reports
```

**Generated Reports:**
- `schema_validation_report.csv` - Issues found and fixed during schema validation
- `consistency_audit_report.csv` - Cross-field constraint violations and fixes

### 5. Complete Workflow (`workflow`)

Run the entire workflow from GeoJSON to URBANopt-ready CSV in one command:

```bash
# Full workflow (GeoJSON → Inference → Post-processing)
usg workflow -i buildings.json -o uo_buildstock_mapping.csv

# With verbose output
usg workflow -i buildings.json -o uo_buildstock_mapping.csv --verbose

# Keep intermediate files for debugging
usg workflow -i buildings.json -o uo_buildstock_mapping.csv --keep-intermediate

# Skip post-processing (inference output only)
usg workflow -i buildings.json -o inference_output.csv --skip-postprocess
```

**Workflow Steps:**
1. Convert GeoJSON to CSV with model attributes
2. Run ML inference to complete missing attributes  
3. Post-process for URBANopt-BuildStock compatibility

### 6. Model Information (`info`)

Display information about the model and resources:

```bash
# Show model and resource info
usg info

# Export attribute list to JSON
usg info --export attributes.json
```

## Default File Locations

The CLI looks for files in these default locations relative to the package:

**Model Files (`resources/pretrained_model/`):**
- `adaptive_model_1.keras` - Neural network model
- `cat_scaler.pkl` - Categorical feature scaler
- `num_scaler.pkl` - Numerical feature scaler  
- `encoding_mapper.json` - Encoding dictionary

**Post-Processor Resources (`resources/postprocessor/`):**
- `options_lookup.tsv` - URBANopt-BuildStock parameter schema
- `consistency_rules.json` - Cross-field constraint rules

## Examples

### Example 1: Complete URBANopt Workflow
```bash
# Single command to go from GeoJSON to URBANopt-ready CSV
usg workflow -i urban_district.json -o uo_buildstock_mapping.csv --verbose
```

### Example 2: Step-by-Step Processing
```bash
# Step 1: Convert GeoJSON
usg geojson2csv -i district.json -o district_raw.csv

# Step 2: Complete attributes with ML
usg complete -i district_raw.csv -o district_completed.csv

# Step 3: Post-process for URBANopt
usg process -i district_completed.csv -o uo_buildstock_mapping.csv -g district.json
```

### Example 3: Single Building Analysis
```bash
# Create known attributes JSON
echo '{
  "Vintage": "1990s",
  "Bedrooms": "3",
  "Geometry Stories": "2",
  "Geometry Building Type RECS": "Single-Family Detached",
  "Heating Fuel": "Natural Gas"
}' > building.json

# Predict missing attributes
usg predict-single -j building.json -o completed_building.json
```

## Exit Codes

- `0`: Success
- `1`: Error occurred (details in error message)

## Tips

1. **Check Resources**: Use `usg info` to verify model files and post-processor resources are available
2. **Verbose Mode**: Use `-v` for detailed progress during processing
3. **Keep Intermediates**: Use `--keep-intermediate` when debugging the workflow
4. **Review Reports**: Check validation reports to understand what was automatically fixed
5. **Climate Zone**: Provide GeoJSON with `--geojson` for accurate climate zone assignment
