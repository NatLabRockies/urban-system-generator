## Key Files Description:

### `usg/inference.py`
Main class with methods:
- `predict_missing_single()` - Predict for one building
- `process_buildings_batch()` - Predict for multiple buildings
- `process_for_simulation()` - Prepare completed data for simulation

### `usg/geojson_processor.py`
- `extract_buildings_from_geojson()` - Read GeoJSON and extract building info
- `geojson_to_csv()` - Convert GeoJSON to CSV format
- `map_geojson_properties()` - Map GeoJSON properties to model attributes

### `usg/model.py`
- `ScaledInputMaskedNN` - Neural network model definition

### `usg/utils.py`
- All mapping functions (vintage, floor area, building type, etc.)
- Helper functions

### `resources/model_attributes.py`
- Contains the `all_model_attributes` list used by the model

## Usage Flow:

1. **GeoJSON → CSV**: `geojson_processor.py` extracts building data
2. **CSV → Predictions**: `inference.py` completes missing attributes
3. **Output**: Completed CSV ready for simulation