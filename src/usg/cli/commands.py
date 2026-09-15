"""Building-level commands: geojson2csv, predict-single, complete, confidence, process, workflow, info."""

from __future__ import annotations

import json
from pathlib import Path

import click

from usg.cli._common import handle_errors, location_options, model_options, ok, parse_formats, warn
from usg.constants import ID_COLUMN

_CONFIDENCE_METHODS = click.Choice(["native", "ensemble", "both"])


def _echo_resolution(resolved) -> None:
    items = resolved.values() if isinstance(resolved, dict) else [resolved]
    for r in {id(v): v for v in items}.values():
        click.echo(f"Model: {r.model_id} — {r.reason}")


def _fact_tips(suggested: list) -> str:
    from usg.report import GEOJSON_FIELDS

    names = [s["fact"] if isinstance(s, dict) else s for s in suggested]
    return ", ".join(f"{name} ({GEOJSON_FIELDS[name]})" if name in GEOJSON_FIELDS else name for name in names)


def _echo_confidence_summary(summary: dict | None) -> None:
    if not summary or summary.get("mean_building_confidence") is None:
        return
    bands = summary["cell_band_counts"]
    line = f"Confidence: mean {summary['mean_building_confidence']:.2f}"
    if summary.get("mean_expected_accuracy") is not None:
        line += f", expected accuracy {summary['mean_expected_accuracy']:.2f}"
    click.echo(f"{line} — {bands['high']} high / {bands['medium']} medium / {bands['low']} low predicted values")
    if summary.get("suggested_facts"):
        click.echo(f"  Most helpful inputs to add: {_fact_tips(summary['suggested_facts'])}")


def _derivations_for(model_dir: Path | None) -> dict | None:
    """The ResStock derivation table of ``model_dir``, else of the newest embedded national G-NN bundle."""
    from usg.bundle import Bundle
    from usg.bundle.manifest import DERIVATIONS
    from usg.resources import embedded_bundles_dir

    if model_dir is None:
        found = sorted(p for p in embedded_bundles_dir().glob("gnn-national-*") if (p / DERIVATIONS).is_file())
        if not found:
            return None
        model_dir = found[-1]
    table = Bundle.open(model_dir, verify="none").derivations()
    if table is None:
        warn(f"{model_dir} has no {DERIVATIONS}; static defaults fill the columns the model does not predict")
    return table


def _echo_postprocess(summary: dict | None, paths: list[str]) -> None:
    if not summary:
        return
    (warn if summary["verdict"] == "review" else click.echo)(summary["headline"])
    for path in paths:
        if path.endswith(".html"):
            ok(f"Post-processing report: {path}")


def _echo_run(result, verbose: bool) -> None:
    for w in result.warnings:
        warn(w)
    if result.issues_csv:
        warn(f"{len(result.completion.issues)} input issue(s) written to {result.issues_csv}")
    _echo_confidence_summary(result.confidence_summary)
    for path in result.reports:
        ok(f"Confidence report: {path}")
    _echo_postprocess(result.postprocess_summary, result.postprocess_reports)
    if verbose and result.manifest_path:
        click.echo(f"  Run manifest: {result.manifest_path}")


@click.command(short_help="Convert an URBANopt GeoJSON file to a building CSV.")
@click.option("-i", "--input", "geojson_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="URBANopt GeoJSON file.")
@click.option("-o", "--output", "csv_file", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output CSV (default: <input>.csv).")
@click.option("-d", "--model-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Also add every column of this bundle's schema (empty cells to fill).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
@location_options(with_state=True)
@handle_errors
def geojson2csv(location, geojson_file, csv_file, model_dir, verbose):
    """Map residential buildings of a GeoJSON file to ResStock parameter columns (no model needed)."""
    from usg.geojson import geojson_to_dataframe, read_project_context

    schema = None
    if model_dir is not None:
        from usg.bundle import Bundle

        schema = Bundle.open(model_dir, verify="none").schema
    context = read_project_context(geojson_file, overrides=location.overrides())
    click.echo(context.summary())
    for w in context.warnings:
        warn(w)
    df = geojson_to_dataframe(geojson_file, schema=schema, context=context)
    csv_file = csv_file or geojson_file.with_suffix(".csv")
    df.to_csv(csv_file, index=False)
    if verbose:
        click.echo(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
    ok(f"Successfully converted {len(df)} buildings to CSV: {csv_file}")


@click.command("predict-single", short_help="Predict missing attributes for a single building.")
@click.option("-a", "--attributes", multiple=True, type=(str, str), metavar="KEY VALUE",
              help="Known attribute (repeatable), e.g. -a Vintage 1990s.")
@click.option("-j", "--json-input", type=click.Path(exists=True, dir_okay=False), help="JSON file of known attributes.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Save the completed building as JSON.")
@click.option("--top-k", type=click.IntRange(min=0), default=3, show_default=True,
              help="Most likely alternatives to show per filled column (0 = none).")
@click.option("--confidence-report/--no-confidence-report", default=True, show_default=True,
              help="Show the confidence of each predicted value; with -o also write <output>_confidence_report.{json,html}.")
@location_options
@model_options
@handle_errors
def predict_single(model, location, attributes, json_input, output, top_k, confidence_report):
    """Predict missing attributes for one building given KEY VALUE pairs and/or a JSON file.

    Location options (and latitude/longitude attributes without a State) fill the building's location columns;
    attributes you give are never overwritten.
    """
    import pandas as pd

    from usg.constants import STATE_COLUMN
    from usg.location import fill_location_columns
    from usg.pipeline import load_model
    from usg.selection import resolve, state_key

    known: dict = dict(attributes)
    if json_input:
        with open(json_input, encoding="utf-8") as f:
            known.update(json.load(f))
    overrides = location.overrides(model.state)
    if not known and not overrides.given:
        warn("no known attributes provided")
    fill = fill_location_columns(pd.DataFrame([known], dtype=object), **overrides.kwargs())
    df = fill.data
    for line in fill.describe():
        click.echo(line)
    for w in fill.warnings:
        warn(w)
    request = model.request()
    if request.state == "auto" and STATE_COLUMN in df.columns and state_key(df.at[0, STATE_COLUMN]):
        request.state = state_key(df.at[0, STATE_COLUMN])
    resolved = resolve(request, model.load_registry())
    _echo_resolution(resolved)
    for w in resolved.warnings:
        warn(w)
    imputer = load_model(resolved, device=model.device, threads=model.threads)
    schema = imputer.schema
    result = imputer.impute(
        df, n_samples=model.samples, seed=model.seed, unknown=model.unknown_labels,
        ignore_columns=[ID_COLUMN, *[c for c in df.columns if c not in schema]],
    )
    for issue in result.issues:
        warn(f"{issue.column}={issue.value!r}: {issue.kind.replace('_', ' ')} ({issue.action.replace('_', ' ')})")
    record = estimator = None
    if confidence_report:
        from usg.confidence import ConfidenceEstimator

        estimator = ConfidenceEstimator.for_imputer(imputer, seed=model.seed)
        record = estimator.estimate(df, result)[0]
        for w in estimator.warnings:
            warn(w)
    scores = record["attributes"] if record else {}

    row = result.values.iloc[0]
    imputed = result.imputed.iloc[0]
    values = {c: (None if pd.isna(row[c]) else (float(row[c]) if schema.column(c).type == "numeric" else row[c]))
              for c in schema.names}
    alternatives = {
        c: [[label, round(p, 4)] for label, p in result.top_k(c, top_k)[0]]
        for c in result.probabilities
        if top_k and bool(imputed[c])
    }
    if output:
        payload = {k: v for k, v in values.items() if v is not None}
        payload["_meta"] = {"model_id": imputer.model_id, "imputed": [c for c in schema.names if imputed[c]],
                            "top_k": alternatives, "seed": model.seed, "n_samples": result.n_samples,
                            "confidence": record}
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ok(f"Predictions saved to: {output}")
        if record is not None:
            from usg.report import write_reports

            out = Path(output)
            for path in write_reports([{"building": out.stem, "confidence": record}], out.parent,
                                      basename=f"{out.stem}_confidence_report", meta={"model": imputer.model_id},
                                      validation=estimator.validation):
                ok(f"Confidence report: {path}")
        return
    click.echo("\nPredicted attributes ([✓] = provided by you; conf = chance the predicted value is right):")
    provided = {c for c in df.columns if c in schema and pd.notna(df.at[0, c])}
    for name in sorted(values):
        if values[name] is None or (name not in provided and not bool(imputed[name])):
            continue
        mark = "✓" if name in provided else " "
        score = scores.get(name) or {}
        conf = f"   conf {score['confidence']:.2f} {score['band']}" if score.get("confidence") is not None else ""
        extra = ""
        if name in alternatives:
            extra = "   (" + ", ".join(f"{label} {p:.2f}" for label, p in alternatives[name]) + ")"
        click.echo(f"  [{mark}] {name}: {values[name]}{conf}{extra}")
    if record is not None and record["summary"].get("mean_confidence") is not None:
        click.echo(f"\nMean confidence {record['summary']['mean_confidence']:.2f} over "
                   f"{record['summary']['n_targets']} predicted values")
        if record.get("suggested_facts"):
            click.echo(f"Most helpful inputs to add: {_fact_tips(record['suggested_facts'])}")


def _confidence_options(enabled, method, passes, drop_prob, seed, sensitivity=None, calibration=None, exclude=None):
    if not enabled:
        return None
    from usg.pipeline import ConfidenceOptions

    return ConfidenceOptions(
        method=method, n_passes=passes, drop_prob=drop_prob, seed=seed,
        sensitivity=sensitivity, calibration=calibration, exclude_targets=list(exclude) if exclude else None,
    )


def _run_complete(model, input_csv, output_csv, id_col, confidence, report_dir, report_format, verbose, location=None):
    from usg.cli._common import LocationArgs
    from usg.pipeline import run_complete

    click.echo(f"Processing buildings from: {input_csv}")
    result = run_complete(
        input_csv, output_csv, model.request(), id_col=id_col, registry=model.load_registry(), confidence=confidence,
        report_dir=report_dir, report_format=report_format, n_samples=model.samples, seed=model.seed,
        unknown=model.unknown_labels, device=model.device, threads=model.threads, log=click.echo,
        location=(location or LocationArgs()).overrides(model.state),
    )
    _echo_run(result, verbose)
    ok(f"Successfully processed {len(result.completion.data)} buildings. Output saved to {output_csv}")
    if verbose:
        click.echo(f"  Total attributes: {len(result.completion.data.columns) - 1}")


_COMPLETE_OPTIONS = [
    click.option("-i", "--input", "input_csv", type=click.Path(exists=True, dir_okay=False), required=True,
                 help="Input CSV with missing attribute cells."),
    click.option("-o", "--output", "output_csv", type=click.Path(dir_okay=False), required=True,
                 help="Completed output CSV."),
    click.option("--id-col", default=ID_COLUMN, show_default=True, help="Building ID column."),
    click.option("--report-dir", type=click.Path(file_okay=False), default=None,
                 help="Directory for confidence reports (default: next to the output)."),
    click.option("--report-format", default="json,html", show_default=True, callback=parse_formats,
                 help="Comma-separated: json, html, md."),
    click.option("--confidence-method", type=_CONFIDENCE_METHODS, default="native", show_default=True,
                 help="native = model probabilities; ensemble = evidence-masking stability; both."),
    click.option("--passes", "n_passes", type=click.IntRange(min=1), default=24, show_default=True,
                 help="Evidence-masking ensemble passes."),
    click.option("--drop-prob", type=click.FloatRange(0.0, 1.0), default=0.35, show_default=True,
                 help="Probability of hiding each known input per ensemble pass."),
    click.option("-v", "--verbose", is_flag=True, help="Verbose output."),
]  # fmt: skip


def _complete_options(fn):
    for option in reversed(_COMPLETE_OPTIONS):
        fn = option(fn)
    return fn


@click.command(short_help="Complete missing attributes for a CSV of buildings.")
@_complete_options
@click.option("--confidence-report/--no-confidence-report", default=True, show_default=True,
              help="Write per-building confidence reports (JSON + HTML) next to the output.")
@location_options
@model_options
@handle_errors
def complete(model, location, input_csv, output_csv, id_col, report_dir, report_format, confidence_method, n_passes,
             drop_prob, verbose, confidence_report):
    """Fill missing cells of a building CSV. With --state auto, a multi-state `State` column uses per-state models.

    Location options apply to every row; rows are also located from latitude/longitude columns when the CSV has no
    State values. Location values already in the CSV are never overwritten.
    """
    confidence = _confidence_options(confidence_report, confidence_method, n_passes, drop_prob, model.seed)
    _run_complete(model, input_csv, output_csv, id_col, confidence, report_dir, report_format, verbose, location)


@click.command(short_help="Complete a CSV and write per-building confidence reports.")
@_complete_options
@click.option("--sensitivity", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Sensitivity table JSON (default: the bundle's sensitivity.json).")
@click.option("--calibration", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Calibration table JSON (default: the bundle's confidence_calibration.json).")
@click.option("--exclude", "exclude_targets", multiple=True,
              help="Attribute to omit from the report (repeatable; replaces the default exclusions).")
@location_options
@model_options
@handle_errors
def confidence(model, location, input_csv, output_csv, id_col, report_dir, report_format, confidence_method, n_passes,
               drop_prob, verbose, sensitivity, calibration, exclude_targets):
    """Complete missing attributes AND report how much to trust each filled value given the inputs provided."""
    options = _confidence_options(
        True, confidence_method, n_passes, drop_prob, model.seed, sensitivity, calibration, exclude_targets
    )
    _run_complete(model, input_csv, output_csv, id_col, options, report_dir, report_format, verbose, location)


@click.command(short_help="Post-process a completed CSV for URBANopt-BuildStock.")
@click.option("-i", "--input", "input_csv", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Completed CSV (inference output).")
@click.option("-o", "--output", "output_csv", type=click.Path(dir_okay=False), required=True,
              help="URBANopt-ready output CSV.")
@click.option("-g", "--geojson", "geojson_file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="GeoJSON project (state and climate zone for defaults).")
@click.option("--options-lookup", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Custom options_lookup.tsv.")
@click.option("--consistency-rules", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Custom consistency_rules.json.")
@click.option("--no-reports", is_flag=True, help="Do not write validation reports.")
@click.option("--reports-dir", type=click.Path(file_okay=False), default=None, help="Directory for validation reports.")
@click.option("-d", "--model-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Bundle whose ResStock derivation table fills the columns the model does not predict "
                   "(default: the embedded national G-NN when installed).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
@location_options(with_state=True)
@handle_errors
def process(location, input_csv, output_csv, geojson_file, options_lookup, consistency_rules, no_reports, reports_dir,
            model_dir, verbose):
    """Fill required columns, validate against options_lookup.tsv and enforce consistency rules.

    Location defaults come from -g and the location options (the options win over the GeoJSON).

    \b
    Example:
        usg process -i completed.csv -o uo_buildstock_mapping.csv -g buildings.json
    """
    from usg.postprocessor import USGPostProcessor

    overrides = location.overrides()
    context = None
    if geojson_file or overrides.given:
        from usg.geojson import project_context_from_geojson, read_project_context

        context = (read_project_context(geojson_file, overrides) if geojson_file
                   else project_context_from_geojson({}, overrides))
        click.echo(context.summary())
        for w in context.warnings:
            warn(w)
    processor = USGPostProcessor(options_lookup, consistency_rules, verbose=True, derivations=_derivations_for(model_dir))
    status = processor.process(input_csv, output_csv, geojson_path=geojson_file, generate_reports=not no_reports,
                               reports_dir=reports_dir, context=context)
    ok(status)
    _echo_postprocess(processor.last_report.summary, processor.last_report_paths)
    if verbose:
        from usg.postprocessor import read_csv_strings

        df = read_csv_strings(output_csv)
        click.echo(f"  Total buildings: {len(df)}\n  Total columns: {len(df.columns)}")


@click.command(short_help="GeoJSON -> completed, URBANopt-ready CSV in one step.")
@click.option("-i", "--input", "geojson_file", type=click.Path(exists=True, dir_okay=False), required=True,
              help="URBANopt GeoJSON file.")
@click.option("-o", "--output", "output_csv", type=click.Path(dir_okay=False), default=None,
              help="Final CSV (default: <input>_uo_buildstock_mapping.csv).")
@click.option("--skip-postprocess", is_flag=True, help="Write the inference output without post-processing.")
@click.option("--keep-intermediate", is_flag=True, help="Keep intermediate CSV files.")
@click.option("--no-reports", is_flag=True, help="Do not write validation reports.")
@click.option("--confidence-report/--no-confidence-report", default=True, show_default=True,
              help="Write a confidence report (JSON + HTML) aligned to the final output.")
@click.option("--report-format", default="json,html", show_default=True, callback=parse_formats,
              help="Comma-separated: json, html, md.")
@click.option("--confidence-method", type=_CONFIDENCE_METHODS, default="native", show_default=True,
              help="native, ensemble or both.")
@click.option("--passes", "n_passes", type=click.IntRange(min=1), default=24, show_default=True,
              help="Evidence-masking ensemble passes.")
@click.option("--drop-prob", type=click.FloatRange(0.0, 1.0), default=0.35, show_default=True,
              help="Probability of hiding each known input per ensemble pass.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
@location_options
@model_options
@handle_errors
def workflow(model, location, geojson_file, output_csv, skip_postprocess, keep_intermediate, no_reports,
             confidence_report, report_format, confidence_method, n_passes, drop_prob, verbose):
    """Convert GeoJSON, pick a model for the project's state, impute, and post-process for URBANopt-BuildStock.

    Writes confidence_report.json/.html next to the output unless --no-confidence-report.

    \b
    Examples:
        usg workflow -i buildings.json -o uo_buildstock_mapping.csv
        usg workflow -i buildings.json -o uo_buildstock_mapping.csv --model tabddpm --samples 32
    """
    from usg.pipeline import run_workflow

    click.echo(f"Input GeoJSON: {geojson_file}")
    result = run_workflow(
        geojson_file, output_csv, model.request(), registry=model.load_registry(), skip_postprocess=skip_postprocess,
        keep_intermediate=keep_intermediate, generate_reports=not no_reports, confidence_report=confidence_report,
        report_format=report_format,
        confidence=_confidence_options(confidence_report, confidence_method, n_passes, drop_prob, model.seed),
        n_samples=model.samples, seed=model.seed, unknown=model.unknown_labels, device=model.device,
        threads=model.threads, verbose=verbose, log=click.echo, location=location.overrides(model.state),
    )
    _echo_run(result, verbose)
    ok("Workflow complete!")
    click.echo(f"  Output saved to: {result.output_csv}")
    click.echo(f"  Total buildings: {len(result.completion.data)}")
    click.echo(f"  URBANopt-BuildStock compatible: {'No (post-processing skipped)' if skip_postprocess else 'Yes'}")


@click.command(short_help="Show the selected model and packaged resources.")
@click.option("--export", type=click.Path(dir_okay=False), help="Export the model's attribute list to JSON.")
@model_options
@handle_errors
def info(model, export):
    """Show which model would be used, its schema and confidence artefacts, and post-processor resources."""
    from usg.bundle import Bundle
    from usg.postprocessor import get_default_resource_paths, load_required_columns
    from usg.selection import resolve

    click.echo("Post-processor resources:")
    for path in get_default_resource_paths():
        click.echo(f"  {'✓' if path.exists() else '✗'} {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    click.echo(f"  ✓ required_columns.json ({len(load_required_columns())} columns)")

    resolved = resolve(model.request(), model.load_registry())
    bundle = Bundle.open(resolved.bundle_path, verify="fast")
    manifest, schema = bundle.manifest, bundle.schema
    click.echo(f"\nModel: {bundle.model_id}  ({manifest.model_type}, arch v{manifest.arch_version})")
    click.echo(f"  Selected because: {resolved.reason}")
    for w in resolved.warnings:
        warn(w)
    click.echo(f"  Path: {bundle.path}")
    click.echo(f"  Region: {manifest.scope} {manifest.target_state or ''}".rstrip())
    click.echo(f"  Data release: {schema.data_release}")
    click.echo(f"  Imputable columns: {len(schema.imputable_names)}  Conditioning: {', '.join(schema.conditioning_names)}")
    click.echo(f"  Calibration table: {'yes' if bundle.calibration() else 'no'}  "
               f"Sensitivity table: {'yes' if bundle.sensitivity() else 'no'}")
    summary = manifest.raw.get("metrics_summary")
    if summary:
        click.echo("  Metrics: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    if export:
        payload = {
            "model_id": bundle.model_id,
            "total_attributes": len(schema.names),
            "categorical_attributes": [c.name for c in schema.cat_columns],
            "numerical_attributes": [c.name for c in schema.num_columns],
            "imputable_attributes": schema.imputable_names,
            "conditioning_attributes": schema.conditioning_names,
            "all_attributes": schema.names,
        }
        Path(export).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ok(f"Attribute list exported to: {export}")
