# Urban System Generator

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**NREL Software Record:** SWR 25-36

Urban System Generator (USG) is a machine learning-based tool for completing missing building attributes in urban energy modeling workflows. Given partial building information from GeoJSON files, USG predicts missing characteristics like HVAC systems, insulation levels, and other energy-relevant properties.

## Features

- **GeoJSON Processing**: Extract and convert building data from GeoJSON files
- **Attribute Prediction**: Use trained neural networks to predict missing building attributes
- **Batch Processing**: Process multiple buildings efficiently
- **Simulation Ready**: Output formatted for urban energy simulation tools

## Installation

```bash
# Clone the repository
git clone https://github.com/NREL/urban-system-generator.git
cd urban-system-generator

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Convert GeoJSON to CSV

```python
from usg.geojson_processor import GeoJSONProcessor
from resources.model_attributes import all_model_attributes

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

# Initialize inference engine
inference = USGInference(
    model_path="resources/pretrained_model/adaptive_model_1.keras",
    cat_scaler_path="resources/pretrained_model/cat_scaler.pkl",
    num_scaler_path="resources/pretrained_model/num_scaler.pkl",
    encoding_dict_path="resources/pretrained_model/encoding_mapper.json",
    all_model_attributes=all_model_attributes,
)

# Process buildings
inference.process_buildings_batch(
    input_csv_path="buildings_incomplete.csv",
    output_csv_path="buildings_complete.csv",
)
```

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

## Repository Structure

```
urban-system-generator/
├── usg/                      # Main package
│   ├── inference.py          # Inference engine
│   ├── geojson_processor.py  # GeoJSON processing
│   ├── model.py              # Neural network model
│   └── utils.py              # Utility functions
├── resources/                # Model resources
│   ├── pretrained_model/     # Trained model files
│   └── model_attributes.py   # Model schema
├── examples/                 # Usage examples
└── tests/                    # Unit tests
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

See `resources/model_attributes.py` for the complete list.

## Examples

See the `examples/` directory for complete usage examples:

- `example_usage.py` - Complete workflow demonstrations
- `example_geojson.json` - Sample GeoJSON file

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=usg tests/
```

## Authors & Contributors

**Lead Developer:** Rawad El Kontar (NREL)

**Organization:** National Renewable Energy Laboratory (NREL)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Copyright (c) 2025 Alliance for Sustainable Energy, LLC

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{urban_system_generator,
  title={Urban System Generator},
  author={El Kontar, Rawad},
  organization={National Renewable Energy Laboratory},
  year={2025},
  url={https://github.com/NREL/urban-system-generator}
}
```

## Acknowledgments

This work was supported by the U.S. Department of Energy's Building Technologies Office.

## Contact

For questions or support, please contact Rawad El Kontar (rawad.elkontar@nrel.gov) or open an issue on GitHub.

---

**NREL Software Record:** SWR 25-36