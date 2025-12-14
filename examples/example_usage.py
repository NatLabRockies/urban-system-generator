"""
Example Usage of Urban System Generator

This script demonstrates how to use the USG package to:
1. Convert GeoJSON to CSV
2. Predict missing building attributes
3. Process buildings for simulation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from usg.geojson_processor import GeoJSONProcessor
from usg.inference import USGInference
from resources.model_attributes import all_model_attributes

import logging
logging.basicConfig(level=logging.ERROR)
#logging.basicConfig(level=logging.WARNING)


def example_geojson_to_csv():
    """Example: Convert GeoJSON to CSV"""
    print("\n=== Example 1: GeoJSON to CSV ===")
    
    processor = GeoJSONProcessor()
    
    # Convert GeoJSON to CSV
    status = processor.geojson_to_csv(
        geojson_path="examples/example_geojson.json",
        output_csv_path="examples/buildings_incomplete.csv",
        all_model_attributes=all_model_attributes,
    )
    
    print(status)


def example_single_building_prediction():
    """Example: Predict missing attributes for a single building"""
    print("\n=== Example 2: Single Building Prediction ===")
    
    # Initialize inference engine
    inference = USGInference(
        model_path="resources/pretrained_model/adaptive_model_1.keras",
        cat_scaler_path="resources/pretrained_model/cat_scaler.pkl",
        num_scaler_path="resources/pretrained_model/num_scaler.pkl",
        encoding_dict_path="resources/pretrained_model/encoding_mapper.json",
        all_model_attributes=all_model_attributes,
    )
    
    # Known attributes for a building
    known_attrs = {
        'Geometry Building Type RECS': 'Single-Family Attached',
        'Vintage': '2010s',
        'Geometry Floor Area': '750-999',
        'Geometry Floor Area Bin': '0-1499',
        'Geometry Stories': '2',
        'Bedrooms': '1',
        'Geometry Building Number Units SFA': '1',
        'Geometry Foundation Type': 'Heated Basement',
        'Geometry Attic Type': 'Vented Attic',
        'HVAC Cooling Type': 'Central AC',
        'Heating Fuel': 'Natural Gas',
        'Vintage ACS': '2010s'
    }
    
    # Predict missing attributes
    completed_attrs = inference.predict_missing_single(known_attrs)
    
    print("\nCompleted Building Attributes:")
    for key, value in completed_attrs.items():
        print(f"  {key}: {value}")


def example_batch_prediction():
    """Example: Predict missing attributes for multiple buildings"""
    print("\n=== Example 3: Batch Prediction ===")
    
    # Initialize inference engine
    inference = USGInference(
        model_path="resources/pretrained_model/adaptive_model_1.keras",
        cat_scaler_path="resources/pretrained_model/cat_scaler.pkl",
        num_scaler_path="resources/pretrained_model/num_scaler.pkl",
        encoding_dict_path="resources/pretrained_model/encoding_mapper.json",
        all_model_attributes=all_model_attributes,
    )
    
    # Process batch of buildings
    status = inference.process_buildings_batch(
        input_csv_path="examples/buildings_incomplete.csv",
        output_csv_path="examples/buildings_completed.csv",
        id_col="Building",
    )
    
    print(status)


def example_end_to_end():
    """Example: Complete workflow from GeoJSON to completed CSV"""
    print("\n=== Example 4: End-to-End Workflow ===")
    
    # Step 1: Convert GeoJSON to CSV
    print("\nStep 1: Converting GeoJSON to CSV...")
    processor = GeoJSONProcessor()
    status = processor.geojson_to_csv(
        geojson_path="examples/example_geojson.json",
        output_csv_path="examples/temp_incomplete.csv",
        all_model_attributes=all_model_attributes,
    )
    print(status)
    
    # Step 2: Complete missing attributes
    print("\nStep 2: Predicting missing attributes...")
    inference = USGInference(
        model_path="resources/pretrained_model/adaptive_model_1.keras",
        cat_scaler_path="resources/pretrained_model/cat_scaler.pkl",
        num_scaler_path="resources/pretrained_model/num_scaler.pkl",
        encoding_dict_path="resources/pretrained_model/encoding_mapper.json",
        all_model_attributes=all_model_attributes,
    )
    
    status = inference.process_for_simulation(
        input_csv_path="examples/temp_incomplete.csv",
        output_csv_path="examples/buildings_simulation_ready.csv",
        id_col="Building",
    )
    
    print(f"\n{status}")
    print("\nWorkflow complete! Buildings are ready for simulation.")


if __name__ == "__main__":
    # Run all examples
    print("Urban System Generator - Usage Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run:
    
    # example_geojson_to_csv()
    # example_single_building_prediction()
    # example_batch_prediction()
    example_end_to_end()