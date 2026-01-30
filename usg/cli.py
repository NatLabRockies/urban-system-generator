#!/usr/bin/env python
"""
Urban System Generator CLI

Command-line interface for the Urban System Generator package.
Provides commands for GeoJSON processing, building attribute prediction,
batch processing of incomplete building data, and post-processing for URBANopt.

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

# CLI default: suppress library warnings to keep command-line output clean.
import logging
logging.basicConfig(level=logging.ERROR)

# Context settings for help options
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def get_default_paths():
    """Get default paths relative to the package installation."""
    module_dir = Path(__file__).parent  # usg/ directory
    return {
        "model_dir": module_dir / "resources" / "pretrained_model",
        "postprocessor_dir": module_dir / "resources" / "postprocessor",
    }


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="0.1.1", prog_name="Urban System Generator")
def cli():
    """
    Urban System Generator - ML-based building attribute completion tool.
    
    \b
    This tool provides the following main functions:
    1. Convert GeoJSON files to CSV format
    2. Predict missing attributes for single buildings
    3. Process batches of buildings to complete missing data
    4. Post-process completed data for URBANopt-BuildStock compatibility
    5. Run complete workflow from GeoJSON to URBANopt-ready CSV
    
    For more information, visit: https://github.com/NREL/urban-system-generator
    """
    pass


@cli.command(short_help="Convert GeoJSON file to CSV format")
@click.option("-i", "--input", "geojson_file", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True), required=True, help="Path to input GeoJSON file containing building data")
@click.option("-o", "--output", "csv_file", type=click.Path(file_okay=True, dir_okay=False, resolve_path=True), default=None, help="Path to output CSV file (default: input_name.csv)")
@click.option("-d", "--model-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True), default=None, help="Directory containing model files")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def geojson2csv(geojson_file, csv_file, model_dir, verbose):
    """Convert a GeoJSON file containing building data to CSV format."""
    try:
        if verbose:
            click.echo(f"Reading GeoJSON file: {geojson_file}")
        
        if csv_file is None:
            csv_file = Path(geojson_file).with_suffix('.csv')
        
        if model_dir is None:
            model_dir = get_default_paths()["model_dir"]
        model_dir = Path(model_dir)
        
        model_path = model_dir / "adaptive_model_1.keras"
        cat_scaler = model_dir / "cat_scaler.pkl"
        num_scaler = model_dir / "num_scaler.pkl"
        encoding = model_dir / "encoding_mapper.json"
        
        if verbose:
            click.echo("Loading model to get all attribute columns...")
        
        inference = USGInference(
            model_path=str(model_path),
            cat_scaler_path=str(cat_scaler),
            num_scaler_path=str(num_scaler),
            encoding_dict_path=str(encoding),
        )
        
        all_model_attributes = inference.all_model_attributes
        
        if verbose:
            click.echo(f"Model expects {len(all_model_attributes)} attributes")
        
        processor = GeoJSONProcessor()
        status = processor.geojson_to_csv(
            geojson_path=geojson_file,
            output_csv_path=str(csv_file),
            all_model_attributes=all_model_attributes,
        )
        
        click.echo(click.style(f"✓ {status}", fg="green"))
        
        if verbose:
            df = pd.read_csv(csv_file)
            click.echo(f"\nOutput contains {len(df)} buildings with {len(df.columns)} columns")
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Predict missing attributes for a single building")
@click.option("-m", "--model", "model_path", type=click.Path(exists=True), default=None, help="Path to trained model file")
@click.option("-c", "--cat-scaler", type=click.Path(exists=True), default=None, help="Path to categorical scaler file")
@click.option("-n", "--num-scaler", type=click.Path(exists=True), default=None, help="Path to numerical scaler file")
@click.option("-e", "--encoding", type=click.Path(exists=True), default=None, help="Path to encoding dictionary")
@click.option("-a", "--attributes", multiple=True, type=(str, str), help="Known attributes as key-value pairs")
@click.option("-j", "--json-input", type=click.Path(exists=True), help="Path to JSON file containing known attributes")
@click.option("-o", "--output", type=click.Path(), help="Path to save predicted attributes as JSON")
def predict_single(model_path, cat_scaler, num_scaler, encoding, attributes, json_input, output):
    """Predict missing attributes for a single building."""
    try:
        defaults = get_default_paths()
        model_dir = defaults["model_dir"]
        
        if model_path is None:
            model_path = str(model_dir / "adaptive_model_1.keras")
        if cat_scaler is None:
            cat_scaler = str(model_dir / "cat_scaler.pkl")
        if num_scaler is None:
            num_scaler = str(model_dir / "num_scaler.pkl")
        if encoding is None:
            encoding = str(model_dir / "encoding_mapper.json")
        
        click.echo("Loading model...")
        inference = USGInference(
            model_path=model_path,
            cat_scaler_path=cat_scaler,
            num_scaler_path=num_scaler,
            encoding_dict_path=encoding,
        )
        
        known_attrs = {}
        if attributes:
            for key, value in attributes:
                known_attrs[key] = value
        
        if json_input:
            with open(json_input, 'r') as f:
                json_attrs = json.load(f)
                known_attrs.update(json_attrs)
        
        if not known_attrs:
            click.echo(click.style("Warning: No known attributes provided.", fg="yellow"))
        
        click.echo(f"Predicting attributes for building with {len(known_attrs)} known attributes...")
        result = inference.predict_missing_single(known_attrs)
        result.pop("_decode_warnings", None)
        
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(click.style(f"✓ Predictions saved to: {output}", fg="green"))
        else:
            click.echo("\nPredicted attributes:")
            for key, value in sorted(result.items()):
                known = "✓" if key in known_attrs else " "
                click.echo(f"  [{known}] {key}: {value}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Process batch of buildings to complete missing attributes")
@click.option("-i", "--input", "input_csv", type=click.Path(exists=True), required=True, help="Path to input CSV file")
@click.option("-o", "--output", "output_csv", type=click.Path(), required=True, help="Path to output CSV file")
@click.option("-d", "--model-dir", type=click.Path(exists=True), default=None, help="Directory containing model files")
@click.option("--id-col", default="Building", help="Column name for building IDs")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def complete(input_csv, output_csv, model_dir, id_col, verbose):
    """Process a batch of buildings to complete missing attributes."""
    try:
        if model_dir is None:
            model_dir = get_default_paths()["model_dir"]
        model_dir = Path(model_dir)
        
        model_path = model_dir / "adaptive_model_1.keras"
        cat_scaler = model_dir / "cat_scaler.pkl"
        num_scaler = model_dir / "num_scaler.pkl"
        encoding = model_dir / "encoding_mapper.json"
        
        for file_path, name in [(model_path, "model"), (cat_scaler, "cat_scaler"), (num_scaler, "num_scaler"), (encoding, "encoding")]:
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        click.echo("Loading model...")
        inference = USGInference(
            model_path=str(model_path),
            cat_scaler_path=str(cat_scaler),
            num_scaler_path=str(num_scaler),
            encoding_dict_path=str(encoding),
        )
        
        click.echo(f"Processing buildings from: {input_csv}")
        status = inference.process_buildings_batch(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            id_col=id_col,
        )
        
        click.echo(click.style(f"\n✓ {status}", fg="green"))
        
        if verbose:
            df = pd.read_csv(output_csv)
            click.echo(f"\nOutput summary:")
            click.echo(f"  Total buildings: {len(df)}")
            click.echo(f"  Total attributes: {len(df.columns) - 1}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Post-process completed CSV for URBANopt-BuildStock")
@click.option("-i", "--input", "input_csv", type=click.Path(exists=True), required=True, help="Path to input CSV (inference output)")
@click.option("-o", "--output", "output_csv", type=click.Path(), required=True, help="Path to output CSV (URBANopt-ready)")
@click.option("-g", "--geojson", "geojson_file", type=click.Path(exists=True), default=None, help="Path to GeoJSON file (for climate zone)")
@click.option("--options-lookup", type=click.Path(exists=True), default=None, help="Path to options_lookup.tsv")
@click.option("--consistency-rules", type=click.Path(exists=True), default=None, help="Path to consistency_rules.json")
@click.option("--no-reports", is_flag=True, help="Disable generation of validation reports")
@click.option("--reports-dir", type=click.Path(), default=None, help="Directory for validation reports")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def process(input_csv, output_csv, geojson_file, options_lookup, consistency_rules, no_reports, reports_dir, verbose):
    """
    Post-process inference output for URBANopt-BuildStock compatibility.
    
    \b
    This command runs a 3-step post-processing pipeline:
    1. Add missing columns required by URBANopt-BuildStock
    2. Validate and fix values against options_lookup.tsv schema
    3. Enforce cross-field consistency rules
    
    \b
    Example:
        usg process -i completed.csv -o urbanopt_ready.csv -g buildings.json
    """
    try:
        from usg.postprocessor import USGPostProcessor, get_default_resource_paths
        
        if options_lookup is None or consistency_rules is None:
            default_options, default_rules = get_default_resource_paths()
            if options_lookup is None:
                options_lookup = str(default_options)
            if consistency_rules is None:
                consistency_rules = str(default_rules)
        
        if not Path(options_lookup).exists():
            raise FileNotFoundError(f"Options lookup file not found: {options_lookup}")
        if not Path(consistency_rules).exists():
            raise FileNotFoundError(f"Consistency rules file not found: {consistency_rules}")
        
        click.echo("=" * 60)
        click.echo("USG Post-Processor")
        click.echo("=" * 60)
        click.echo(f"Input:  {input_csv}")
        click.echo(f"Output: {output_csv}")
        if geojson_file:
            click.echo(f"GeoJSON: {geojson_file}")
        click.echo()
        
        processor = USGPostProcessor(
            options_lookup_path=options_lookup,
            consistency_rules_path=consistency_rules,
        )
        
        status = processor.process(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            geojson_path=geojson_file,
            generate_reports=not no_reports,
            reports_dir=reports_dir,
        )
        
        click.echo()
        click.echo(click.style(f"✓ {status}", fg="green"))
        
        if verbose:
            df = pd.read_csv(output_csv)
            click.echo(f"\nOutput summary:")
            click.echo(f"  Total buildings: {len(df)}")
            click.echo(f"  Total columns: {len(df.columns)}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Complete end-to-end workflow from GeoJSON to URBANopt-ready CSV")
@click.option("-i", "--input", "geojson_file", type=click.Path(exists=True), required=True, help="Path to input GeoJSON file")
@click.option("-o", "--output", "output_csv", type=click.Path(), default=None, help="Path to final output CSV")
@click.option("-d", "--model-dir", type=click.Path(exists=True), default=None, help="Directory containing model files")
@click.option("--skip-postprocess", is_flag=True, help="Skip post-processing step")
@click.option("--keep-intermediate", is_flag=True, help="Keep intermediate CSV files")
@click.option("--no-reports", is_flag=True, help="Disable generation of validation reports")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def workflow(geojson_file, output_csv, model_dir, skip_postprocess, keep_intermediate, no_reports, verbose):
    """
    Run complete workflow from GeoJSON to URBANopt-ready CSV.
    
    \b
    This command combines all steps:
    1. Convert GeoJSON to CSV
    2. Predict missing attributes for all buildings
    3. Post-process for URBANopt-BuildStock compatibility
    
    \b
    Example:
        usg workflow -i buildings.json -o uo_buildstock_mapping.csv --verbose
    """
    try:
        from usg.postprocessor import USGPostProcessor, get_default_resource_paths
        
        geojson_path = Path(geojson_file)
        if output_csv is None:
            output_csv = geojson_path.parent / f"{geojson_path.stem}_uo_buildstock_mapping.csv"
        output_csv = Path(output_csv)
        
        intermediate_csv = geojson_path.parent / f"{geojson_path.stem}_intermediate.csv"
        inference_csv = geojson_path.parent / f"{geojson_path.stem}_inference_output.csv"
        
        defaults = get_default_paths()
        if model_dir is None:
            model_dir = defaults["model_dir"]
        model_dir = Path(model_dir)
        
        model_path = model_dir / "adaptive_model_1.keras"
        cat_scaler = model_dir / "cat_scaler.pkl"
        num_scaler = model_dir / "num_scaler.pkl"
        encoding = model_dir / "encoding_mapper.json"
        
        for file_path, name in [(model_path, "model"), (cat_scaler, "cat_scaler"), (num_scaler, "num_scaler"), (encoding, "encoding")]:
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        click.echo("=" * 60)
        click.echo("USG Complete Workflow")
        click.echo("=" * 60)
        click.echo(f"Input GeoJSON: {geojson_file}")
        click.echo(f"Output CSV:    {output_csv}")
        click.echo()
        
        # Step 1: Load model
        click.echo("Step 1: Loading model...")
        inference = USGInference(
            model_path=str(model_path),
            cat_scaler_path=str(cat_scaler),
            num_scaler_path=str(num_scaler),
            encoding_dict_path=str(encoding),
        )
        all_model_attributes = inference.all_model_attributes
        
        if verbose:
            click.echo(f"  Model expects {len(all_model_attributes)} attributes")
        
        # Step 2: Convert GeoJSON to CSV
        click.echo("\nStep 2: Converting GeoJSON to CSV...")
        processor = GeoJSONProcessor()
        status = processor.geojson_to_csv(
            geojson_path=str(geojson_file),
            output_csv_path=str(intermediate_csv),
            all_model_attributes=all_model_attributes,
        )
        
        if verbose:
            df_temp = pd.read_csv(intermediate_csv)
            click.echo(f"  Created CSV with {len(df_temp)} buildings, {len(df_temp.columns)} columns")
        
        # Step 3: Complete missing attributes
        click.echo("\nStep 3: Predicting missing attributes...")
        
        inference_output = output_csv if skip_postprocess else inference_csv
        
        status = inference.process_buildings_batch(
            input_csv_path=str(intermediate_csv),
            output_csv_path=str(inference_output),
            id_col="Building",
        )
        
        if verbose:
            click.echo(f"  {status}")
        
        # Step 4: Post-process (unless skipped)
        if not skip_postprocess:
            click.echo("\nStep 4: Post-processing for URBANopt-BuildStock...")
            
            default_options, default_rules = get_default_resource_paths()
            
            postprocessor = USGPostProcessor(
                options_lookup_path=default_options,
                consistency_rules_path=default_rules,
            )
            
            status = postprocessor.process(
                input_csv_path=inference_output,
                output_csv_path=output_csv,
                geojson_path=geojson_file,
                generate_reports=not no_reports,
                reports_dir=output_csv.parent,
            )
            
            if verbose:
                click.echo(f"  {status}")
        else:
            click.echo("\nStep 4: Skipped post-processing (--skip-postprocess)")
        
        # Cleanup intermediate files
        if not keep_intermediate:
            for f in [intermediate_csv, inference_csv]:
                if f.exists() and f != output_csv:
                    f.unlink()
                    if verbose:
                        click.echo(f"  Removed intermediate file: {f}")
        
        click.echo()
        click.echo(click.style("✓ Workflow complete!", fg="green"))
        click.echo(f"  Output saved to: {output_csv}")
        
        df_final = pd.read_csv(output_csv)
        click.echo(f"\nFinal output summary:")
        click.echo(f"  Total buildings: {len(df_final)}")
        click.echo(f"  Total columns: {len(df_final.columns)}")
        
        if not skip_postprocess:
            click.echo(f"  URBANopt-BuildStock compatible: Yes")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@cli.command(short_help="Show information about model and resources")
@click.option("-d", "--model-dir", type=click.Path(exists=True), default=None, help="Directory containing model files")
@click.option("--export", type=click.Path(), help="Export attribute list to JSON file")
def info(model_dir, export):
    """Display information about the model and its expected attributes."""
    try:
        from usg.postprocessor import get_default_resource_paths
        
        defaults = get_default_paths()
        if model_dir is None:
            model_dir = defaults["model_dir"]
        model_dir = Path(model_dir)
        
        model_path = model_dir / "adaptive_model_1.keras"
        cat_scaler = model_dir / "cat_scaler.pkl"
        num_scaler = model_dir / "num_scaler.pkl"
        encoding = model_dir / "encoding_mapper.json"
        
        click.echo("Urban System Generator - Resource Information")
        click.echo("=" * 55)
        
        click.echo("\nModel files:")
        for file_path, name in [(model_path, "Neural Network Model"), (cat_scaler, "Categorical Scaler"), (num_scaler, "Numerical Scaler"), (encoding, "Encoding Dictionary")]:
            if file_path.exists():
                size = file_path.stat().st_size / 1024
                click.echo(f"  ✓ {name}: {file_path.name} ({size:.1f} KB)")
            else:
                click.echo(f"  ✗ {name}: NOT FOUND")
        
        click.echo("\nPost-processor resources:")
        default_options, default_rules = get_default_resource_paths()
        
        for file_path, name in [(default_options, "Options Lookup"), (default_rules, "Consistency Rules")]:
            if file_path.exists():
                size = file_path.stat().st_size / 1024
                click.echo(f"  ✓ {name} ({size:.1f} KB)")
            else:
                click.echo(f"  ✗ {name}: NOT FOUND")
        
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
            
            click.echo("\nSample categorical attributes:")
            for attr in inference.cat_cols[:10]:
                click.echo(f"  - {attr}")
            if len(inference.cat_cols) > 10:
                click.echo(f"  ... and {len(inference.cat_cols) - 10} more")
            
            click.echo("\nNumerical attributes:")
            for attr in inference.num_cols:
                click.echo(f"  - {attr}")
            
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
