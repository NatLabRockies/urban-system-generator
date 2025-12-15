# Urban System Generator CLI Documentation

The Urban System Generator provides a comprehensive command-line interface (CLI) for all its functionality.

## Installation

After installing the package, the `usg` command will be available:

```bash
pip install -e .
```

## Available Commands

### 1. Convert GeoJSON to CSV

Convert building data from GeoJSON format to CSV:

```bash
usg geojson2csv -i buildings.json -o buildings.csv

# With verbose output
usg geojson2csv -i buildings.json -o buildings.csv --verbose
```

### 2. Predict Single Building

Predict missing attributes for a single building:

```bash
# Using command-line attributes
usg predict-single \
  -a "Vintage" "2000s" \
  -a "Bedrooms" "3" \
  -a "Geometry Stories" "2"

# Using JSON input file
usg predict-single -j building_attrs.json -o completed.json

# With custom model location
usg predict-single \
  -m path/to/model.keras \
  -c path/to/cat_scaler.pkl \
  -n path/to/num_scaler.pkl \
  -e path/to/encoding.json \
  -j building.json
```

### 3. Batch Prediction

Process multiple buildings to complete missing attributes:

```bash
# Basic usage
usg batch-predict -i incomplete.csv -o completed.csv

# With progress display
usg batch-predict -i incomplete.csv -o completed.csv --progress

# Custom model location
usg batch-predict \
  -i incomplete.csv \
  -o completed.csv \
  -m path/to/model.keras \
  --progress
```

### 4. Complete Workflow

Run the entire workflow from GeoJSON to simulation-ready CSV:

```bash
# Basic workflow
usg workflow -i buildings.json -o simulation_ready.csv

# With verbose output and keep intermediate files
usg workflow -i buildings.json -o simulation_ready.csv --verbose --keep-intermediate

# Custom model directory
usg workflow -i buildings.json -o simulation_ready.csv -m path/to/model/dir
```

### 5. Model Information

Display information about the model and its attributes:

```bash
# Show model info
usg info

# Export attribute list to JSON
usg info --export attributes.json

# Custom model directory
usg info -m path/to/model/dir
```

## Command Options

### Global Options
- `-h, --help`: Show help message
- `--version`: Show version information

### Common Options
- `-m, --model`: Path to model file
- `-c, --cat-scaler`: Path to categorical scaler
- `-n, --num-scaler`: Path to numerical scaler
- `-e, --encoding`: Path to encoding dictionary
- `-i, --input`: Input file path
- `-o, --output`: Output file path
- `-v, --verbose`: Enable verbose output

## Examples

### Example 1: Complete Workflow
```bash
# Convert GeoJSON and complete all missing attributes
usg workflow -i urban_area.json -o simulation_data.csv --verbose
```

### Example 2: Batch Processing with Custom Model
```bash
usg batch-predict \
  -i survey_data.csv \
  -o completed_survey.csv \
  -m models/custom_model.keras \
  -c models/cat_scaler.pkl \
  -n models/num_scaler.pkl \
  -e models/encoding.json \
  --progress
```

### Example 3: Single Building Analysis
```bash
# Create a JSON file with known attributes
echo '{
  "Vintage": "1990s",
  "Bedrooms": "3",
  "Geometry Stories": "2",
  "Geometry Building Type RECS": "Single-Family Detached"
}' > building.json

# Predict missing attributes
usg predict-single -j building.json -o completed_building.json
```

### Example 4: Pipeline Processing
```bash
# Step 1: Convert GeoJSON
usg geojson2csv -i district.json -o district.csv

# Step 2: Complete attributes
usg batch-predict -i district.csv -o district_completed.csv --progress

# Step 3: Check results
usg info --export district_attributes.json
```

## Default File Locations

The CLI looks for model files in these default locations:
- Model: `resources/pretrained_model/adaptive_model_1.keras`
- Cat Scaler: `resources/pretrained_model/cat_scaler.pkl`
- Num Scaler: `resources/pretrained_model/num_scaler.pkl`
- Encoding: `resources/pretrained_model/encoding_mapper.json`

## Exit Codes

- `0`: Success
- `1`: Error occurred (details in error message)

## Environment Variables

You can set default paths using environment variables:

```bash
export USG_MODEL_DIR=/path/to/model/directory
export USG_OUTPUT_DIR=/path/to/output/directory
```

## Tips

1. **Check Model Compatibility**: Use `usg info` to verify model files are properly loaded
2. **Verbose Mode**: Use `-v` or `--verbose` for detailed output during processing
3. **Progress Tracking**: Use `--progress` for batch operations to track completion
4. **Keep Intermediates**: Use `--keep-intermediate` when debugging workflows
