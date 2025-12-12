"""
GeoJSON Processing Module

This module handles reading GeoJSON files and converting building data
into the format required for the inference model.

Author: Rawad El Kontar (NREL)
"""

import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional


class GeoJSONProcessor:
    """
    Processes GeoJSON files containing building information and converts
    them to CSV format for inference.
    """
    
    # Property mapping from GeoJSON to model attributes
    PROPERTY_MAPPER = {
        'building_type': 'Geometry Building Type RECS',
        'year_built': 'Vintage',
        'floor_area': 'Geometry Floor Area',
        'number_of_stories': 'Geometry Stories',
        'number_of_bedrooms': 'Bedrooms',
        'number_of_residential_units': 'Geometry Building Number Units SFA',
        'foundation_type': 'Geometry Foundation Type',
        'attic_type': 'Geometry Attic Type',
        'system_type': 'HVAC Cooling Type',
        'heating_system_fuel_type': 'Heating Fuel',
    }
    
    def __init__(self):
        """Initialize the GeoJSON processor."""
        pass
    
    # ========== MAPPING FUNCTIONS ==========
    
    @staticmethod
    def map_vintage(year_built: int) -> str:
        """Map year_built to vintage category."""
        if year_built < 1940:
            return "<1940"
        elif year_built < 1950:
            return "1940s"
        elif year_built < 1960:
            return "1950s"
        elif year_built < 1970:
            return "1960s"
        elif year_built < 1980:
            return "1970s"
        elif year_built < 1990:
            return "1980s"
        elif year_built < 2000:
            return "1990s"
        elif year_built < 2010:
            return "2000s"
        else:
            return "2010s"
    
    @staticmethod
    def map_vintage_acs(year_built: int) -> str:
        """Map year_built to ACS vintage category."""
        if year_built < 1940:
            return "<1940"
        elif year_built < 1960:
            return "1940-59"
        elif year_built < 1980:
            return "1960-79"
        elif year_built < 2000:
            return "1980-99"
        elif year_built < 2010:
            return "2000-09"
        else:
            return "2010s"
    
    @staticmethod
    def map_floor_area_to_category(floor_area: float) -> str:
        """Map floor area to category."""
        if floor_area < 500:
            return "0-499"
        elif floor_area < 750:
            return "500-749"
        elif floor_area < 1000:
            return "750-999"
        elif floor_area < 1500:
            return "1000-1499"
        elif floor_area < 2000:
            return "1500-1999"
        elif floor_area < 2500:
            return "2000-2499"
        elif floor_area < 3000:
            return "2500-2999"
        elif floor_area < 4000:
            return "3000-3999"
        else:
            return "4000+"
    
    @staticmethod
    def map_floor_area_to_bin(floor_area: float) -> str:
        """Map floor area to bin category."""
        if floor_area < 1500:
            return "0-1499"
        elif floor_area < 2500:
            return "1500-2499"
        elif floor_area < 4000:
            return "2500-3999"
        else:
            return "4000+"
    
    @staticmethod
    def map_building_type(building_type: str, properties: dict) -> str:
        """Map building type to RECS category."""
        if building_type == 'Multifamily':
            units = properties.get('number_of_residential_units', 0)
            if isinstance(units, str):
                units = int(units) if units.isdigit() else 0
            
            if units >= 5:
                return 'Multi-Family with 5+ Units'
            elif 2 <= units <= 4:
                return 'Multi-Family with 2 - 4 Units'
            else:
                return 'Multi-Family with 5+ Units'
        
        elif building_type in ['Single-Family Attached', 'Single-Family']:
            return 'Single-Family Attached'
        
        elif building_type == 'Single-Family Detached':
            return 'Single-Family Detached'
        
        elif building_type == 'Mobile Home':
            return 'Mobile Home'
        
        else:
            return 'Other Category'
    
    @staticmethod
    def map_foundation_type(foundation_type: str) -> str:
        """Map foundation type from GeoJSON to model format."""
        mapping = {
            'slab': 'Slab',
            'crawlspace - vented': 'Vented Crawlspace',
            'crawlspace - unvented': 'Unvented Crawlspace',
            'crawlspace - conditioned': 'Conditioned Crawlspace',
            'basement - unconditioned': 'Unheated Basement',
            'basement - conditioned': 'Heated Basement',
            'ambient': 'Ambient',
        }
        return mapping.get(foundation_type, 'Other Category')
    
    @staticmethod
    def map_attic_type(attic_type: str) -> str:
        """Map attic type from GeoJSON to model format."""
        mapping = {
            'attic - vented': 'Vented Attic',
            'attic - unvented': 'Unvented Attic',
            'attic - conditioned': 'Finished Attic or Cathedral Ceilings',
        }
        return mapping.get(attic_type, 'None')
    
    @staticmethod
    def map_heating_fuel(heating_fuel: str) -> str:
        """Map heating fuel type from GeoJSON to model format."""
        mapping = {
            'electricity': 'Electricity',
            'natural gas': 'Natural Gas',
            'fuel oil': 'Fuel Oil',
            'propane': 'Propane',
            'wood': 'Wood',
        }
        return mapping.get(heating_fuel, 'Other Fuel')
    
    @staticmethod
    def map_hvac_cooling_type(system_type: str) -> str:
        """Map HVAC system type from GeoJSON to model format."""
        mapping = {
            'Residential - electric resistance and central air conditioner': 'Central AC',
            'Residential - electric resistance and room air conditioner': 'Room AC',
            'Residential - furnace and central air conditioner': 'Central AC',
            'Residential - electric resistance and evaporative cooler': 'Evaporative or Swamp Cooler',
            'Residential - air-to-air heat pump': 'Ducted Heat Pump',
        }
        return mapping.get(system_type, 'None')
    
    # ========== PROCESSING METHODS ==========
    
    def read_geojson(self, geojson_path: str) -> dict:
        """
        Read a GeoJSON file.
        
        Args:
            geojson_path: Path to the GeoJSON file
        
        Returns:
            Parsed GeoJSON dictionary
        """
        with open(geojson_path, 'r') as f:
            return json.load(f)
    
    def extract_residential_buildings(self, geojson: dict) -> List[dict]:
        """
        Extract residential buildings from GeoJSON.
        
        Args:
            geojson: Parsed GeoJSON dictionary
        
        Returns:
            List of residential building features
        """
        residential_types = [
            'Multifamily',
            'Single-Family Attached',
            'Single-Family Detached',
            'Single-Family'
        ]
        
        residential_buildings = []
        
        for feature in geojson.get('features', []):
            props = feature.get('properties', {})
            
            # Skip non-building features
            if props.get('type') != 'Building':
                continue
            
            # Check if residential
            building_type = props.get('building_type', '')
            if building_type in residential_types:
                residential_buildings.append(feature)
        
        return residential_buildings
    
    def map_properties(self, feature: dict) -> dict:
        """
        Map GeoJSON properties to model attributes.
        
        Args:
            feature: GeoJSON feature dictionary
        
        Returns:
            Dictionary of mapped attributes
        """
        props = feature.get('properties', {})
        mapped = {}
        
        # Extract building ID
        building_id = props.get('id', '')
        # Remove "way/" prefix if present
        if isinstance(building_id, str) and building_id.startswith('way/'):
            building_id = building_id.replace('way/', '')
        mapped['Building'] = building_id
        
        # Map and transform properties
        for geojson_key, model_key in self.PROPERTY_MAPPER.items():
            value = props.get(geojson_key)
            
            if value is None:
                continue
            
            # Apply transformations based on property type
            if geojson_key == 'year_built':
                mapped['Vintage'] = self.map_vintage(int(value))
                mapped['Vintage ACS'] = self.map_vintage_acs(int(value))
            
            elif geojson_key == 'floor_area':
                mapped['Geometry Floor Area'] = self.map_floor_area_to_category(float(value))
                mapped['Geometry Floor Area Bin'] = self.map_floor_area_to_bin(float(value))
            
            elif geojson_key == 'building_type':
                mapped['Geometry Building Type RECS'] = self.map_building_type(value, props)
            
            elif geojson_key == 'foundation_type':
                mapped['Geometry Foundation Type'] = self.map_foundation_type(value)
            
            elif geojson_key == 'attic_type':
                mapped['Geometry Attic Type'] = self.map_attic_type(value)
            
            elif geojson_key == 'heating_system_fuel_type':
                mapped['Heating Fuel'] = self.map_heating_fuel(value)
            
            elif geojson_key == 'system_type':
                mapped['HVAC Cooling Type'] = self.map_hvac_cooling_type(value)
            
            elif geojson_key in ['number_of_stories', 'number_of_bedrooms']:
                mapped[model_key] = str(int(value))
            
            elif geojson_key == 'number_of_residential_units':
                mapped[model_key] = str(int(value)) if value else 'None'
        
        return mapped
    
    def geojson_to_csv(
        self,
        geojson_path: str,
        output_csv_path: str,
        all_model_attributes: Optional[List[str]] = None,
    ) -> str:
        """
        Convert GeoJSON to CSV format for inference.
        
        Args:
            geojson_path: Path to input GeoJSON file
            output_csv_path: Path to save output CSV
            all_model_attributes: List of all model attributes (optional)
        
        Returns:
            Status message with path to output CSV
        """
        # Read GeoJSON
        geojson = self.read_geojson(geojson_path)
        
        # Extract residential buildings
        buildings = self.extract_residential_buildings(geojson)
        
        if not buildings:
            raise ValueError("No residential buildings found in GeoJSON")
        
        # Map properties for each building
        mapped_buildings = [self.map_properties(b) for b in buildings]
        
        # Create DataFrame
        df = pd.DataFrame(mapped_buildings)
        
        # If all_model_attributes provided, ensure all columns exist
        if all_model_attributes:
            # Find missing columns
            missing_cols = [attr for attr in all_model_attributes if attr not in df.columns]
            
            # Add all missing columns at once (more efficient)
            if missing_cols:
                missing_df = pd.DataFrame({col: [None] * len(df) for col in missing_cols})
                df = pd.concat([df, missing_df], axis=1)
            
            # Reorder columns: Building ID first, then model attributes
            cols = ['Building'] + [c for c in all_model_attributes if c in df.columns]
            df = df[cols]
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        
        return f"Successfully converted {len(buildings)} buildings to CSV: {output_csv_path}"
    
    def geojson_to_dataframe(
        self,
        geojson_path: str,
        all_model_attributes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert GeoJSON directly to DataFrame.
        
        Args:
            geojson_path: Path to GeoJSON file
            all_model_attributes: List of all model attributes (optional)
        
        Returns:
            DataFrame with building data
        """
        geojson = self.read_geojson(geojson_path)
        buildings = self.extract_residential_buildings(geojson)
        mapped_buildings = [self.map_properties(b) for b in buildings]
        df = pd.DataFrame(mapped_buildings)
        
        if all_model_attributes:
            for attr in all_model_attributes:
                if attr not in df.columns:
                    df[attr] = None
            
            cols = ['Building'] + [c for c in all_model_attributes if c in df.columns]
            df = df[cols]
        
        return df


# ========== TEST CODE ==========
if __name__ == "__main__":
    """
    Test the GeoJSON processor with sample data.
    
    To run this test:
    1. Make sure you have a GeoJSON file (e.g., 'test_buildings.json')
    2. Run: python -m usg.geojson_processor
    """
    
    print("=" * 60)
    print("Testing GeoJSON Processor")
    print("=" * 60)
    
    # Initialize processor
    processor = GeoJSONProcessor()
    
    # Example 1: Test mapping functions
    print("\n--- Example 1: Testing Mapping Functions ---")
    print(f"Year 1995 -> Vintage: {processor.map_vintage(1995)}")
    print(f"Year 1995 -> Vintage ACS: {processor.map_vintage_acs(1995)}")
    print(f"Floor Area 1200 -> Category: {processor.map_floor_area_to_category(1200)}")
    print(f"Floor Area 1200 -> Bin: {processor.map_floor_area_to_bin(1200)}")
    
    # Example 2: Convert GeoJSON to CSV
    print("\n--- Example 2: Convert GeoJSON to CSV ---")
    
    # Update these paths to your actual files
    geojson_file = "test_buildings.json"  # Your GeoJSON file
    output_csv = "buildings_output.csv"
    
    try:
        result = processor.geojson_to_csv(
            geojson_path=geojson_file,
            output_csv_path=output_csv,
        )
        print(result)
        
        # Display first few rows
        df = pd.read_csv(output_csv)
        print("\nFirst 5 rows of output CSV:")
        print(df.head())
        print(f"\nTotal columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        
    except FileNotFoundError:
        print(f"ERROR: Could not find '{geojson_file}'")
        print("\nTo test this module:")
        print("1. Create a GeoJSON file with building data")
        print("2. Update the 'geojson_file' variable above")
        print("3. Run: python -m usg.geojson_processor")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Example 3: Convert to DataFrame directly
    print("\n--- Example 3: Convert GeoJSON to DataFrame ---")
    try:
        df = processor.geojson_to_dataframe(geojson_path=geojson_file)
        print(f"Created DataFrame with {len(df)} buildings")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{geojson_file}'")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)