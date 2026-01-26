# Urban System Generator

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**NLR Software Record:** SWR 25-36

Urban System Generator (USG) is a machine learning-based tool for completing missing building attributes in urban energy modeling workflows. Given partial building information from GeoJSON files, USG predicts missing characteristics like HVAC systems, insulation levels, and other energy-relevant properties. The tool includes post-processing for URBANopt-BuildStock compatibility.

## Features

- **GeoJSON Processing**: Extract and convert building data from GeoJSON files
- **Attribute Prediction**: Use trained neural networks to predict missing building attributes
- **Batch Processing**: Process multiple buildings efficiently
- **Post-Processing Pipeline**: Validate and fix outputs for URBANopt-BuildStock compatibility
  - Missing column filler for required URBANopt parameters
  - Schema validation against options_lookup.tsv
  - Cross-field consistency enforcement (HVAC, PV, foundations, etc.)
- **Simulation Ready**: Output formatted for URBANopt and other urban energy simulation tools

## Installation

```bash
# Clone the repository
git clone https://github.com/NatLabRockies/urban-system-generator.git
cd urban-system-generator

# Create a virtual environment
python3 -m venv .venv  # Or try `python -m venv .venv` if `python3` doesn't work

# Activate the virtual environment
source .venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install the package and dependencies
pip install -e .

```

## Command Line Interface (CLI)

After installation, the `usg` command will be available. The CLI provides easy access to all functionality:

```bash
# Convert GeoJSON to CSV
usg geojson2csv -i buildings.json -o buildings.csv

# Predict missing attributes for a single building
usg predict-single -a "Vintage" "2000s" -a "Bedrooms" "3"

# Process batch of buildings (ML inference)
usg complete -i incomplete.csv -o completed.csv

# Post-process for URBANopt-BuildStock compatibility
usg process -i completed.csv -o uo_buildstock_mapping.csv -g buildings.json

# Run complete workflow (GeoJSON → Inference → Post-processing)
usg workflow -i buildings.json -o uo_buildstock_mapping.csv --verbose

# Get help
usg --help
```

For detailed CLI documentation, see [docs/cli_usage.md](docs/cli_usage.md).

## Quick Start (Python API)

### 1. Convert GeoJSON to CSV

```python
from usg.geojson_processor import GeoJSONProcessor
from usg.resources.model_attributes import all_model_attributes

processor = GeoJSONProcessor()
processor.geojson_to_csv(
    geojson_path="buildings.json",
    output_csv_path="buildings_incomplete.csv",
    all_model_attributes=all_model_attributes,
)
```

### 2. Predict Missing Attributes

```python
from usg.inference import USGInference
from pathlib import Path

# Get the path to bundled resources
usg_dir = Path(__file__).parent / "usg" / "resources" / "pretrained_model"

# Initialize inference engine
inference = USGInference(
    model_path=usg_dir / "adaptive_model_1.keras",
    cat_scaler_path=usg_dir / "cat_scaler.pkl",
    num_scaler_path=usg_dir / "num_scaler.pkl",
    encoding_dict_path=usg_dir / "encoding_mapper.json",
)

# Process buildings
inference.process_buildings_batch(
    input_csv_path="buildings_incomplete.csv",
    output_csv_path="buildings_complete.csv",
)
```

**Note:** When using the CLI (`usg` command), resource paths are automatically resolved. The manual paths above are only needed for direct Python API usage.

### 3. Single Building Prediction

```python
known_attrs = {
    'Geometry Building Type RECS': 'Single-Family Detached',
    'Vintage': '2000s',
    'Geometry Floor Area': '1500-1999',
    'Geometry Stories': '2',
    'Bedrooms': '3',
}

completed = inference.predict_missing_single(known_attrs)
print(completed)
```

### 4. Post-Process for URBANopt-BuildStock

```python
from usg.postprocessor import USGPostProcessor, get_default_resource_paths

# Get bundled resource paths (recommended)
options_lookup, consistency_rules = get_default_resource_paths()

# Initialize post-processor
postprocessor = USGPostProcessor(
    options_lookup_path=options_lookup,
    consistency_rules_path=consistency_rules,
)

# Process inference output for URBANopt compatibility
postprocessor.process(
    input_csv_path="buildings_complete.csv",
    output_csv_path="uo_buildstock_mapping.csv",
    geojson_path="buildings.json",  # Optional: for climate zone extraction
    generate_reports=True,  # Generate validation reports
)
```

The post-processor performs three steps:
1. **Missing Column Filler**: Adds required URBANopt-BuildStock columns with appropriate defaults
2. **Schema Validator**: Validates all values against options_lookup.tsv and fixes invalid entries
3. **Consistency Processor**: Enforces cross-field constraints (HVAC configurations, PV systems, etc.)

## Repository Structure

```
urban-system-generator/
├── usg/                          # Main package
│   ├── __init__.py               # Package initialization
│   ├── inference.py              # Inference engine
│   ├── geojson_processor.py      # GeoJSON processing
│   ├── postprocessor.py          # URBANopt post-processing
│   ├── model.py                  # Neural network model
│   ├── cli.py                    # Command-line interface
│   └── resources/                # Bundled model and post-processor resources
│       ├── pretrained_model/     # Trained model files
│       │   ├── adaptive_model_1.keras
│       │   ├── cat_scaler.pkl
│       │   ├── num_scaler.pkl
│       │   └── encoding_mapper.json
│       ├── postprocessor/        # Post-processor resources
│       │   ├── options_lookup.tsv     # URBANopt-BuildStock schema
│       │   └── consistency_rules.json # Cross-field constraints
│       └── model_attributes.py   # Model schema
├── examples/                     # Usage examples
├── docs/                         # Documentation
└── tests/                        # Unit tests
```

## Input Data Format

### GeoJSON Requirements

Buildings in your GeoJSON should have the following properties:

```json
{
  "type": "Feature",
  "properties": {
    "id": "building_001",
    "type": "Building",
    "building_type": "Single-Family Detached",
    "year_built": 2005,
    "floor_area": 2000,
    "number_of_stories": 2,
    "number_of_bedrooms": 3,
    "number_of_residential_units": 1
  },
  "geometry": {...}
}
```

### Supported Building Types

- Single-Family Detached
- Single-Family Attached
- Multifamily (2-4 units)
- Multifamily (5+ units)
- Mobile Home

## Model Attributes

The model can predict the following building attributes:

- HVAC Cooling Type
- HVAC Heating Type
- Heating Fuel
- Foundation Type
- Attic Type
- Insulation levels
- Window characteristics
- And more...

See `usg/resources/model_attributes.py` for the complete list.

## Examples

See the `examples/` directory for complete usage examples:

- `example_usage.py` - Complete workflow demonstrations
- `example_geojson.json` - Sample GeoJSON file

## Testing

The repository includes comprehensive unit and integration tests.

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run specific test module
python run_tests.py inference
python run_tests.py geojson

# Using pytest (alternative)
pytest tests/

# Run with coverage
pytest --cov=usg tests/
```

### Test Structure

- `tests/test_inference.py` - Unit tests for the inference engine
- `tests/test_geojson_processor.py` - Unit tests for GeoJSON processing
- `tests/test_integration.py` - End-to-end integration tests

## Authors & Contributors

**Lead Developer:** Rawad El Kontar

**Organization:** National Laboratory of the Rockies (NLR)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Copyright (c) 2025-2026, Alliance for Energy Innovation, LLC


This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{urban_system_generator,
  title={Urban System Generator},
  author={El Kontar, Rawad},
  organization={National Laboratory of the Rockies},
  year={2025},
  url={https://github.com/NatLabRockies/urban-system-generator}
}
```

## Acknowledgments

This work was authored by the National Laboratory of the Rockies(NLR) for the U.S. Department of Energy (DOE), operated under Contract No. DE-AC36-08GO28308. Funding provided by NLR’s Laboratory Directed Research and Development program, and the DOE Office of Science. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

## Contact

For questions or support, please contact Rawad El Kontar (rawad.elkontar@nlr.gov) or open an issue on GitHub.

---

**NLR Software Record:** SWR 25-36