#!/usr/bin/env python
"""
Urban System Generator CLI

Command-line interface for the Urban System Generator package.
Provides commands for GeoJSON processing, building attribute prediction,
and batch processing of incomplete building data.

"""

import json
import sys
from pathlib import Path
import click
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from usg.inference import USGInference
from usg.geojson_processor import GeoJSONProcessor

# Context settings for help options
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="0.1.0", prog_name="Urban System Generator")
def cli():
    """
    Urban System Generator - ML-based building attribute completion tool.
    
    \b
    This tool provides three main functions:
    1. Convert GeoJSON files to CSV format
    2. Predict missing attributes for single buildings
    3. Process batches of buildings to complete missing data
    
    For more information, visit: https://github.com/NREL/urban-system-generator
    """
    pass


@cli.command(short_help="Convert GeoJSON file to CSV format")
@click.option(
    "-i", "--input",
    "geojson_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to input GeoJSON file containing building data"
)
@click.option(
    "-o", "--output",
    "csv_file",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to output CSV file (default: input_name.csv)"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
def geojson2csv(geojson_file, csv_file, verbose):
    """
    Convert a GeoJSON file containing building data to CSV format.
    
    \b
    This command extracts building properties from a GeoJSON file and
    converts them to a CSV format suitable for the inference model.
    
    \b
    Example:
        usg geojson2csv -i buildings.json -o buildings.csv
    """
    try:
        if verbose:
            click.echo(f"Reading GeoJSON file: {geojson_file}")
        
        # Set default output path if not provided
        if csv_file is None:
            csv_file = Path(geojson_file).with_suffix('.csv')
        
        # Initialize processor
        processor = GeoJSONProcessor()
        
        # Convert GeoJSON to CSV
        status = processor.geojson_to_csv(
            geojson_path=geojson_file,
            output_csv_path=str(csv_file),
        )
        
        click.echo(click.style(f"✓ {status}", fg="green"))
        
        if verbose:
            # Show sample of the output
            df = pd.read_csv(csv_file)
            click.echo(f"\nOutput contains {len(df)} buildings with {len(df.columns)} attributes")
            click.echo(f"First 5 building IDs: {df['Building'].head().tolist()}")
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Predict missing attributes for a single building")
@click.option(
    "-m", "--model",
    "model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/adaptive_model_1.keras",
    help="Path to trained model file"
)
@click.option(
    "-c", "--cat-scaler",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/cat_scaler.pkl",
    help="Path to categorical scaler file"
)
@click.option(
    "-n", "--num-scaler",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/num_scaler.pkl",
    help="Path to numerical scaler file"
)
@click.option(
    "-e", "--encoding",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/encoding_mapper.json",
    help="Path to encoding dictionary"
)
@click.option(
    "-a", "--attributes",
    multiple=True,
    type=(str, str),
    help="Known attributes as key-value pairs (can be used multiple times)"
)
@click.option(
    "-j", "--json-input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Path to JSON file containing known attributes"
)
@click.option(
    "-o", "--output",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
    help="Path to save predicted attributes as JSON"
)
def predict_single(model_path, cat_scaler, num_scaler, encoding, attributes, json_input, output):
    """
    Predict missing attributes for a single building.
    
    \b
    Provide known attributes either as command-line arguments or via JSON file.
    The model will predict all missing attributes.
    
    \b
    Examples:
        # Using command-line attributes
        usg predict-single -a "Vintage" "2000s" -a "Bedrooms" "3"
        
        # Using JSON file input
        usg predict-single -j building_attrs.json -o completed.json
    """
    try:
        # Initialize inference engine
        click.echo("Loading model...")
        inference = USGInference(
            model_path=model_path,
            cat_scaler_path=cat_scaler,
            num_scaler_path=num_scaler,
            encoding_dict_path=encoding,
        )
        
        # Build known attributes dictionary
        known_attrs = {}
        
        # Add attributes from command line
        if attributes:
            for key, value in attributes:
                known_attrs[key] = value
        
        # Add/override with JSON file if provided
        if json_input:
            with open(json_input, 'r') as f:
                json_attrs = json.load(f)
                known_attrs.update(json_attrs)
        
        if not known_attrs:
            click.echo(click.style("✗ Error: No attributes provided", fg="red"), err=True)
            click.echo("Use -a to provide attributes or -j for JSON file input")
            sys.exit(1)
        
        click.echo(f"Known attributes: {len(known_attrs)}")
        for key, value in known_attrs.items():
            click.echo(f"  {key}: {value}")
        
        # Predict missing attributes
        click.echo("\nPredicting missing attributes...")
        completed_attrs = inference.predict_missing_single(known_attrs)
        
        click.echo(click.style(f"✓ Completed {len(completed_attrs)} attributes", fg="green"))
        
        # Save to file if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(completed_attrs, f, indent=2)
            click.echo(f"✓ Results saved to {output}")
        else:
            # Display some key predicted attributes
            click.echo("\nSample predicted attributes:")
            sample_attrs = ['HVAC Cooling Type', 'Heating Fuel', 'Insulation Wall', 
                          'Water Heater Fuel', 'HVAC Heating Type']
            for attr in sample_attrs:
                if attr in completed_attrs and attr not in known_attrs:
                    click.echo(f"  {attr}: {completed_attrs[attr]}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Process batch of buildings to complete missing attributes")
@click.option(
    "-i", "--input",
    "input_csv",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to input CSV with incomplete building data"
)
@click.option(
    "-o", "--output",
    "output_csv",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to output CSV with completed data (default: input_completed.csv)"
)
@click.option(
    "-m", "--model",
    "model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/adaptive_model_1.keras",
    help="Path to trained model file"
)
@click.option(
    "-c", "--cat-scaler",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/cat_scaler.pkl",
    help="Path to categorical scaler file"
)
@click.option(
    "-n", "--num-scaler",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/num_scaler.pkl",
    help="Path to numerical scaler file"
)
@click.option(
    "-e", "--encoding",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    default="resources/pretrained_model/encoding_mapper.json",
    help="Path to encoding dictionary"
)
@click.option(
    "--id-column",
    default="Building",
    help="Name of the building ID column"
)
@click.option(
    "-p", "--progress",
    is_flag=True,
    help="Show progress during processing"
)
def batch_predict(input_csv, output_csv, model_path, cat_scaler, num_scaler, 
                 encoding, id_column, progress):
    """
    Process a batch of buildings to complete missing attributes.
    
    \b
    This command takes a CSV file with incomplete building data and
    uses the ML model to predict all missing attributes for each building.
    
    \b
    Example:
        usg batch-predict -i incomplete.csv -o completed.csv --progress
    """
    try:
        # Set default output path if not provided
        if output_csv is None:
            input_path = Path(input_csv)
            output_csv = input_path.parent / f"{input_path.stem}_completed.csv"
        
        click.echo(f"Loading input data from: {input_csv}")
        df_input = pd.read_csv(input_csv)
        click.echo(f"Found {len(df_input)} buildings to process")
        
        # Initialize inference engine
        click.echo("Loading model...")
        inference = USGInference(
            model_path=model_path,
            cat_scaler_path=cat_scaler,
            num_scaler_path=num_scaler,
            encoding_dict_path=encoding,
        )
        
        click.echo(f"Model loaded with {len(inference.all_model_attributes)} attributes")
        
        # Process buildings
        if progress:
            click.echo("\nProcessing buildings...")
            
        status = inference.process_buildings_batch(
            input_csv_path=input_csv,
            output_csv_path=str(output_csv),
            id_col=id_column,
        )
        
        click.echo(click.style(f"\n✓ {status}", fg="green"))
        
        # Show summary statistics
        df_output = pd.read_csv(output_csv)
        click.echo(f"\nOutput summary:")
        click.echo(f"  Buildings processed: {len(df_output)}")
        click.echo(f"  Attributes per building: {len(df_output.columns) - 1}")
        click.echo(f"  Output saved to: {output_csv}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Complete end-to-end workflow from GeoJSON to completed CSV")
@click.option(
    "-i", "--input",
    "geojson_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to input GeoJSON file"
)
@click.option(
    "-o", "--output",
    "output_csv",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to final output CSV (default: input_simulation_ready.csv)"
)
@click.option(
    "-m", "--model-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    default="resources/pretrained_model",
    help="Directory containing model files"
)
@click.option(
    "--keep-intermediate",
    is_flag=True,
    help="Keep intermediate CSV file"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
def workflow(geojson_file, output_csv, model_dir, keep_intermediate, verbose):
    """
    Run complete workflow from GeoJSON to simulation-ready CSV.
    
    \b
    This command combines all steps:
    1. Convert GeoJSON to CSV
    2. Predict missing attributes for all buildings
    3. Output simulation-ready data
    
    \b
    Example:
        usg workflow -i buildings.json -o simulation_ready.csv --verbose
    """
    try:
        # Set paths
        geojson_path = Path(geojson_file)
        if output_csv is None:
            output_csv = geojson_path.parent / f"{geojson_path.stem}_simulation_ready.csv"
        
        intermediate_csv = geojson_path.parent / f"{geojson_path.stem}_intermediate.csv"
        
        # Step 1: Convert GeoJSON to CSV
        click.echo("Step 1: Converting GeoJSON to CSV...")
        processor = GeoJSONProcessor()
        status = processor.geojson_to_csv(
            geojson_path=str(geojson_file),
            output_csv_path=str(intermediate_csv),
        )
        
        if verbose:
            click.echo(f"  {status}")
        
        # Step 2: Complete missing attributes
        click.echo("\nStep 2: Predicting missing attributes...")
        
        # Build model paths
        model_path = Path(model_dir) / "adaptive_model_1.keras"
        cat_scaler = Path(model_dir) / "cat_scaler.pkl"
        num_scaler = Path(model_dir) / "num_scaler.pkl"
        encoding = Path(model_dir) / "encoding_mapper.json"
        
        # Check model files exist
        for file_path, name in [(model_path, "model"), (cat_scaler, "cat_scaler"),
                                (num_scaler, "num_scaler"), (encoding, "encoding")]:
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        inference = USGInference(
            model_path=str(model_path),
            cat_scaler_path=str(cat_scaler),
            num_scaler_path=str(num_scaler),
            encoding_dict_path=str(encoding),
        )
        
        status = inference.process_buildings_batch(
            input_csv_path=str(intermediate_csv),
            output_csv_path=str(output_csv),
            id_col="Building",
        )
        
        if verbose:
            click.echo(f"  {status}")
        
        # Cleanup intermediate file if requested
        if not keep_intermediate and intermediate_csv.exists():
            intermediate_csv.unlink()
            if verbose:
                click.echo(f"  Removed intermediate file: {intermediate_csv}")
        
        click.echo(click.style(f"\n✓ Workflow complete!", fg="green"))
        click.echo(f"  Simulation-ready data saved to: {output_csv}")
        
        # Show summary
        df_final = pd.read_csv(output_csv)
        click.echo(f"\nFinal output summary:")
        click.echo(f"  Total buildings: {len(df_final)}")
        click.echo(f"  Total attributes: {len(df_final.columns) - 1}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Show information about model attributes")
@click.option(
    "-m", "--model-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    default="resources/pretrained_model",
    help="Directory containing model files"
)
@click.option(
    "--export",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
    help="Export attribute list to JSON file"
)
def info(model_dir, export):
    """
    Display information about the model and its expected attributes.
    
    \b
    This command shows:
    - Number of attributes the model expects
    - List of categorical and numerical attributes
    - Model file locations
    """
    try:
        # Build model paths
        model_path = Path(model_dir) / "adaptive_model_1.keras"
        cat_scaler = Path(model_dir) / "cat_scaler.pkl"
        num_scaler = Path(model_dir) / "num_scaler.pkl"
        encoding = Path(model_dir) / "encoding_mapper.json"
        
        click.echo("Urban System Generator Model Information")
        click.echo("=" * 50)
        
        # Check which files exist
        click.echo("\nModel files:")
        for file_path, name in [(model_path, "Neural Network Model"),
                                (cat_scaler, "Categorical Scaler"),
                                (num_scaler, "Numerical Scaler"),
                                (encoding, "Encoding Dictionary")]:
            if file_path.exists():
                size = file_path.stat().st_size / 1024  # KB
                click.echo(f"  ✓ {name}: {file_path.name} ({size:.1f} KB)")
            else:
                click.echo(f"  ✗ {name}: NOT FOUND")
        
        # Load model to get attributes
        if all(p.exists() for p in [model_path, cat_scaler, num_scaler, encoding]):
            click.echo("\nLoading model to extract attribute information...")
            inference = USGInference(
                model_path=str(model_path),
                cat_scaler_path=str(cat_scaler),
                num_scaler_path=str(num_scaler),
                encoding_dict_path=str(encoding),
            )
            
            click.echo(f"\nModel Attributes:")
            click.echo(f"  Total attributes: {len(inference.all_model_attributes)}")
            click.echo(f"  Categorical: {len(inference.cat_cols)}")
            click.echo(f"  Numerical: {len(inference.num_cols)}")
            
            # Show sample attributes
            click.echo("\nSample categorical attributes:")
            for attr in inference.cat_cols[:10]:
                click.echo(f"  - {attr}")
            if len(inference.cat_cols) > 10:
                click.echo(f"  ... and {len(inference.cat_cols) - 10} more")
            
            click.echo("\nNumerical attributes:")
            for attr in inference.num_cols:
                click.echo(f"  - {attr}")
            
            # Export if requested
            if export:
                export_data = {
                    "total_attributes": len(inference.all_model_attributes),
                    "categorical_attributes": inference.cat_cols,
                    "numerical_attributes": inference.num_cols,
                    "all_attributes": inference.all_model_attributes
                }
                with open(export, 'w') as f:
                    json.dump(export_data, f, indent=2)
                click.echo(f"\n✓ Attribute list exported to: {export}")
        else:
            click.echo("\n✗ Cannot load model - missing required files")
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
