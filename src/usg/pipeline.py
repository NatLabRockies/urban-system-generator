"""CLI-independent orchestration: resolve -> impute (+ confidence) -> post-process -> reports -> run manifest."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from usg.__about__ import __version__
from usg.bundle import Bundle
from usg.bundle.manifest import MANIFEST, sha256_file, write_json
from usg.constants import ID_COLUMN, STATE_COLUMN
from usg.errors import InputValidationError
from usg.geojson import ProjectContext, geojson_to_dataframe, read_project_context
from usg.location import LocationOverrides, fill_location_columns
from usg.schema import InputIssue
from usg.selection import ModelRequest, ResolvedModel, resolve, resolve_partitioned, state_key

RUN_MANIFEST = "run_manifest.json"
Log = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


@dataclass
class ConfidenceOptions:
    method: str = "native"
    n_passes: int = 24
    drop_prob: float = 0.35
    seed: int = 0
    sensitivity: dict | str | Path | None = None
    calibration: dict | str | Path | None = None
    exclude_targets: list[str] | None = None


@dataclass
class Completion:
    data: pd.DataFrame  # id column + output columns
    imputed: pd.DataFrame  # bool, same columns without the id; True where the model filled the cell
    issues: list[InputIssue]
    records: list[dict] | None  # [{"building": id, "confidence": record}] when confidence was requested
    models: list[dict]
    warnings: list[str] = field(default_factory=list)
    n_samples: int | None = None
    confidence_validation: dict[str, dict] = field(default_factory=dict)  # model_id -> held-out reliability

    @property
    def inference_columns(self) -> list[str]:
        return list(self.data.columns)


@dataclass
class RunResult:
    output_csv: Path
    completion: Completion
    resolved: ResolvedModel | dict[str | None, ResolvedModel]
    context: ProjectContext | None = None
    reports: list[str] = field(default_factory=list)
    issues_csv: Path | None = None
    manifest_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    confidence_summary: dict | None = None  # report run_summary when confidence reports were written
    postprocess_summary: dict | None = None  # usg.postprocess_report summary when post-processing ran
    postprocess_reports: list[str] = field(default_factory=list)


# ==================================================================================================
# Completion
# ==================================================================================================


def read_input_csv(path: str | Path) -> pd.DataFrame:
    """Read a building CSV as text; only empty cells are missing (``"None"`` stays a value)."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except FileNotFoundError:
        raise InputValidationError(f"input CSV not found: {path}") from None
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"cannot read CSV {path}: {exc}") from None


def load_model(resolved: ResolvedModel, *, device: str = "cpu", threads: int | None = 1, verify: str = "fast"):
    return Bundle.open(resolved.bundle_path, verify=verify).load_imputer(device=device, threads=threads)


def resolve_for_dataframe(
    df: pd.DataFrame, request: ModelRequest, registry=None, cache=None, *, embedded_dir=None
) -> ResolvedModel | dict[str | None, ResolvedModel]:
    """Partition by the ``State`` column when ``state="auto"`` and rows span several states; else one model."""
    auto = request.state is None or str(request.state).strip().lower() == "auto"
    if auto and request.bundle_dir is None and not request.model_id and STATE_COLUMN in df.columns:
        states = {state_key(v) for v in df[STATE_COLUMN]}
        if len(states) > 1:
            return resolve_partitioned(request, df[STATE_COLUMN], registry, cache, embedded_dir=embedded_dir)
        (only,) = states
        if only is not None:
            request = replace(request, state=only)
    return resolve(request, registry, cache, embedded_dir=embedded_dir)


def _parameter_names() -> set[str]:
    from usg.geojson import _allowed_options

    return set(_allowed_options())


def _output_columns(schema, values: pd.DataFrame, required: set[str]) -> list[str]:
    """Imputable columns, plus conditioning inputs that are required downstream and known for some row."""
    return [
        c.name
        for c in schema.columns
        if c.role == "imputable" or (c.name in required and values[c.name].notna().any())
    ]


def complete_dataframe(
    df: pd.DataFrame,
    model,
    *,
    id_col: str = ID_COLUMN,
    n_samples: int | None = None,
    seed: int | None = 0,
    batch_size: int = 4096,
    unknown: str = "missing",
    decode: str | None = None,
    confidence: ConfidenceOptions | None = None,
    context_columns: Iterable[str] = (),
    device: str = "cpu",
    threads: int | None = 1,
    verify: str = "fast",
    log: Log = _quiet,
) -> Completion:
    """Fill missing cells of ``df`` with ``model``: an imputer, a :class:`ResolvedModel`, or a per-state mapping.

    Output columns: ``id_col``, every schema imputable column, conditioning inputs that post-processing requires
    (e.g. ``State``), and any other required ResStock parameter present in ``df`` (passed through unchanged).
    """
    from usg.postprocessor import load_required_columns

    if id_col not in df.columns:
        raise InputValidationError(f"ID column {id_col!r} not found in input (columns: {list(df.columns)[:10]}...)")
    df = df.reset_index(drop=True)
    if isinstance(model, Mapping):
        keys = df[STATE_COLUMN].map(state_key) if STATE_COLUMN in df.columns else pd.Series([None] * len(df))
        by_bundle: dict[Path, tuple[ResolvedModel, list[np.ndarray]]] = {}
        for key, resolved in model.items():
            mask = keys.isna().to_numpy() if key is None else (keys == key).to_numpy()
            if mask.any():
                by_bundle.setdefault(Path(resolved.bundle_path), (resolved, []))[1].append(np.flatnonzero(mask))
        groups = [(r, np.sort(np.concatenate(p))) for r, p in by_bundle.values()]
        covered = sum(len(p) for _, p in groups)
        if covered != len(df):
            raise InputValidationError(f"{len(df) - covered} row(s) have a state with no resolved model")
    else:
        groups = [(model, np.arange(len(df)))]

    required = set(load_required_columns())
    parameters = _parameter_names()
    parts, issues, records, models, warnings = [], [], [], [], []
    validations: dict[str, dict] = {}
    k_used = None
    for spec, positions in groups:
        if isinstance(spec, ResolvedModel):
            log(f"Loading {spec.model_id} for {len(positions)} building(s)")
            imputer = load_model(spec, device=device, threads=threads, verify=verify)
        else:
            imputer = spec
        schema = imputer.schema
        sub = df.iloc[positions].reset_index(drop=True)
        ignore = [id_col] + [c for c in sub.columns if c not in schema and c in parameters]
        result = imputer.impute(
            sub, n_samples=n_samples, seed=seed, batch_size=batch_size, unknown=unknown, decode=decode,
            ignore_columns=ignore,
        )
        k_used = result.n_samples
        for issue in result.issues:
            row = int(positions[issue.row]) if issue.row >= 0 else -1
            if row >= 0 or issue not in issues:
                issues.append(replace(issue, row=row))

        columns = _output_columns(schema, result.values, required)
        out = result.values[columns].copy()
        imputed = result.imputed[columns].copy()
        for col in sub.columns:
            if col != id_col and col not in schema and col in required:
                out[col] = sub[col].to_numpy()
                imputed[col] = False
        out.insert(0, id_col, sub[id_col].to_numpy())
        out.index = imputed.index = positions
        parts.append((out, imputed))

        if confidence is not None:
            from usg.confidence import ConfidenceEstimator

            estimator = ConfidenceEstimator.for_imputer(
                imputer,
                method=confidence.method,
                n_passes=confidence.n_passes,
                drop_prob=confidence.drop_prob,
                seed=confidence.seed,
                sensitivity=confidence.sensitivity,
                calibration=confidence.calibration,
                exclude_targets=confidence.exclude_targets,
            )
            recs = estimator.estimate(sub, result, context_columns=context_columns)
            records += [(int(p), {"building": b, "confidence": r}) for p, b, r in zip(positions, sub[id_col], recs)]
            warnings += [w for w in estimator.warnings if w not in warnings]
            if estimator.validation:
                validations[imputer.model_id] = estimator.validation
        bundle = imputer.bundle
        models.append(
            {
                "model_id": imputer.model_id,
                "model_type": bundle.model_type,
                "bundle_path": str(bundle.path),
                "manifest_sha256": sha256_file(bundle.path / MANIFEST),
                "schema_hash": bundle.manifest.schema_hash,
                "n_rows": len(positions),
                **(
                    {"region": spec.region, "source": spec.source, "reason": spec.reason,
                     "archive_sha256": spec.archive_sha256}
                    if isinstance(spec, ResolvedModel) else {}
                ),
            }
        )
        if isinstance(spec, ResolvedModel):
            warnings += [w for w in spec.warnings if w not in warnings]

    data = pd.concat([p[0] for p in parts]).sort_index()
    imputed = pd.concat([p[1] for p in parts]).sort_index()
    data.index = imputed.index = pd.RangeIndex(len(data))
    imputed = imputed.fillna(False).astype(bool)
    if confidence is not None:
        records = [r for _, r in sorted(records, key=lambda x: x[0])]
    return Completion(
        data=data,
        imputed=imputed,
        issues=sorted(issues, key=lambda i: (i.row, i.column)),
        records=records if confidence is not None else None,
        models=models,
        warnings=warnings,
        n_samples=k_used,
        confidence_validation=validations,
    )


# ==================================================================================================
# Outputs
# ==================================================================================================


def write_issues_csv(issues: list[InputIssue], data: pd.DataFrame, path: Path, id_col: str = ID_COLUMN) -> Path | None:
    if not issues:
        return None
    ids = data[id_col].tolist() if id_col in data.columns else []
    rows = [
        {
            "row": i.row,
            id_col: ids[i.row] if 0 <= i.row < len(ids) else "",
            "column": i.column,
            "value": i.value,
            "kind": i.kind,
            "action": i.action,
        }
        for i in issues
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_manifest(
    *,
    command: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    completion: Completion,
    seed: int | None,
    batch_size: int,
    decode: str | None,
    context: ProjectContext | None = None,
    confidence: ConfidenceOptions | None = None,
    warnings: Iterable[str] = (),
    location: dict | None = None,  # location resolution record (usg.location)
    confidence_summary: dict | None = None,
) -> dict:
    torch = sys.modules.get("torch")
    return {
        "run_manifest_format": "usg-run-manifest/v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "inputs": inputs,
        "outputs": outputs,
        "models": completion.models,
        "seed": seed,
        "n_samples": completion.n_samples,
        "batch_size": batch_size,
        "decode": decode,
        "confidence": None if confidence is None else {
            "method": confidence.method, "n_passes": confidence.n_passes, "drop_prob": confidence.drop_prob,
            "validation": completion.confidence_validation or None, "summary": confidence_summary,
        },
        "project_context": None if context is None else {
            "state": context.state, "state_source": context.state_source, "climate_zone": context.climate_zone,
            "weather_filename": context.weather_filename,
        },
        "location": location,
        "n_buildings": len(completion.data),
        "n_input_issues": len(completion.issues),
        "versions": {
            "urban_system_generator": __version__,
            "torch": getattr(torch, "__version__", None),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


def _report_meta(completion: Completion, **extra) -> dict:
    meta = {"models": ", ".join(m["model_id"] for m in completion.models), **extra}
    return {k: v for k, v in meta.items() if v is not None}


def _write_confidence_reports(records: list[dict], completion: Completion, out_dir, formats, meta: dict,
                              result: RunResult) -> None:
    """Write the confidence reports (with the model's held-out reliability) and keep the run summary on ``result``."""
    from usg.report import run_summary, write_reports

    summary = run_summary(records)
    validations = list(completion.confidence_validation.values())
    result.reports = write_reports(records, out_dir, formats=formats, meta=meta, summary=summary,
                                   validation=validations[0] if len(validations) == 1 else None)
    result.confidence_summary = summary


def _compact_summary(summary: dict | None) -> dict | None:
    if not summary:
        return None
    keys = ("n_buildings", "mean_building_confidence", "mean_expected_accuracy", "cell_band_counts", "suggested_facts")
    return {k: summary.get(k) for k in keys}


def run_complete(
    input_csv: str | Path,
    output_csv: str | Path,
    request: ModelRequest,
    *,
    id_col: str = ID_COLUMN,
    registry=None,
    cache=None,
    confidence: ConfidenceOptions | None = None,
    report_dir: str | Path | None = None,
    report_format: Iterable[str] = ("json", "html"),
    n_samples: int | None = None,
    seed: int | None = 0,
    batch_size: int = 4096,
    unknown: str = "missing",
    decode: str | None = None,
    device: str = "cpu",
    threads: int | None = 1,
    embedded_dir=None,
    log: Log = _quiet,
    location: LocationOverrides | None = None,
) -> RunResult:
    """Complete a building CSV (per-state models for mixed-state files) and write reports + run manifest.

    ``location`` values apply to every row, and rows are located from ``latitude``/``longitude`` columns when the CSV
    has no State values (:func:`usg.location.fill_location_columns`); input cells are never overwritten.
    """
    output_csv = Path(output_csv)
    df = read_input_csv(input_csv)
    if id_col not in df.columns:
        raise InputValidationError(f"ID column {id_col!r} not found in {input_csv}")
    # --- location (usg.location) -----------------------------------------------------------------------
    location_fill = fill_location_columns(df, **(location or LocationOverrides()).kwargs())
    df = location_fill.data
    for line in location_fill.describe():
        log(line)
    # --- end location --------------------------------------------------------------------------------------------
    resolved = resolve_for_dataframe(df, request, registry, cache, embedded_dir=embedded_dir)
    for r in {id(v): v for v in (resolved.values() if isinstance(resolved, dict) else [resolved])}.values():
        log(f"Model: {r.model_id} — {r.reason}")
    completion = complete_dataframe(
        df, resolved, id_col=id_col, n_samples=n_samples, seed=seed, batch_size=batch_size, unknown=unknown,
        decode=decode, confidence=confidence, device=device, threads=threads, log=log,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completion.data.to_csv(output_csv, index=False)
    result = RunResult(output_csv, completion, resolved,
                       warnings=list(dict.fromkeys([*location_fill.warnings, *completion.warnings])))
    result.issues_csv = write_issues_csv(
        completion.issues, completion.data, output_csv.with_name(f"{output_csv.stem}_input_issues.csv"), id_col
    )
    if confidence is not None:
        _write_confidence_reports(
            completion.records, completion, report_dir or output_csv.parent, report_format,
            _report_meta(completion, input_csv=str(input_csv), output_csv=str(output_csv), method=confidence.method,
                         stage="inference"),
            result,
        )
    manifest = run_manifest(
        command="complete",
        inputs={"csv": str(input_csv)},
        outputs={"csv": str(output_csv), "input_issues": str(result.issues_csv) if result.issues_csv else None,
                 "reports": result.reports},
        completion=completion, seed=seed, batch_size=batch_size, decode=decode, confidence=confidence,
        warnings=result.warnings, location=location_fill.to_json(),
        confidence_summary=_compact_summary(result.confidence_summary),
    )
    result.manifest_path = output_csv.parent / RUN_MANIFEST
    write_json(result.manifest_path, manifest)
    return result


def run_workflow(
    geojson_path: str | Path,
    output_csv: str | Path | None = None,
    request: ModelRequest | None = None,
    *,
    registry=None,
    cache=None,
    skip_postprocess: bool = False,
    keep_intermediate: bool = False,
    generate_reports: bool = True,
    confidence_report: bool = False,
    report_format: Iterable[str] = ("json", "html"),
    confidence: ConfidenceOptions | None = None,
    n_samples: int | None = None,
    seed: int | None = 0,
    batch_size: int = 4096,
    unknown: str = "missing",
    decode: str | None = None,
    device: str = "cpu",
    threads: int | None = 1,
    embedded_dir=None,
    verbose: bool = False,
    log: Log = _quiet,
    location: LocationOverrides | None = None,
) -> RunResult:
    """GeoJSON -> model inputs -> imputation -> URBANopt-BuildStock post-processing -> reports + run manifest.

    ``location`` holds explicit values (CLI ``--state``/``--climate-zone``/``--latitude``/``--longitude``) that override
    the location derived from the GeoJSON (:func:`usg.geojson.project_context_from_geojson`).
    """
    from usg.postprocessor import USGPostProcessor, read_csv_strings

    geojson_path = Path(geojson_path)
    output_csv = Path(output_csv) if output_csv else geojson_path.with_name(f"{geojson_path.stem}_uo_buildstock_mapping.csv")
    out_dir = output_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_csv = out_dir / f"{geojson_path.stem}_intermediate.csv"
    inference_csv = output_csv if skip_postprocess else out_dir / f"{geojson_path.stem}_inference_output.csv"
    if confidence_report and confidence is None:
        confidence = ConfidenceOptions(seed=seed or 0)

    # --- location (usg.location): explicit values win over the GeoJSON project and coordinates ----------
    context = read_project_context(geojson_path, overrides=location)
    log(context.summary())
    for w in context.warnings:
        log(f"Warning: {w}")
    # --- end location --------------------------------------------------------------------------------------------
    request = replace(request or ModelRequest(), context=request.context if request and request.context else context)
    resolved = resolve(request, registry, cache, embedded_dir=embedded_dir)
    log(f"Model: {resolved.model_id} — {resolved.reason}")
    for w in resolved.warnings:
        log(f"Warning: {w}")

    bundle = Bundle.open(resolved.bundle_path, verify="fast")
    df = geojson_to_dataframe(geojson_path, schema=bundle.schema, context=context)
    df.to_csv(intermediate_csv, index=False)
    log(f"Converted {len(df)} residential building(s) from {geojson_path.name}")
    context_columns = [c for c in context.conditioning_values() if c in bundle.schema]

    completion = complete_dataframe(
        df, resolved, n_samples=n_samples, seed=seed, batch_size=batch_size, unknown=unknown, decode=decode,
        confidence=confidence, context_columns=context_columns, device=device, threads=threads, log=log,
    )
    completion.data.to_csv(inference_csv, index=False)
    log(f"Imputed {int(completion.imputed.to_numpy().sum())} cell(s) with {resolved.model_id}")
    result = RunResult(output_csv, completion, resolved, context,
                       warnings=list(dict.fromkeys([*context.warnings, *completion.warnings])))
    result.issues_csv = write_issues_csv(
        completion.issues, completion.data, output_csv.with_name(f"{output_csv.stem}_input_issues.csv")
    )

    if not skip_postprocess:
        log("Post-processing for URBANopt-BuildStock...")
        processor = USGPostProcessor(verbose=verbose, derivations=bundle.derivations())
        processor.process(
            inference_csv, output_csv, generate_reports=generate_reports, reports_dir=out_dir, context=context,
            imputed=completion.imputed, context_columns=context_columns, report_formats=tuple(report_format),
        )
        result.postprocess_summary = processor.last_report.summary
        result.postprocess_reports = processor.last_report_paths

    if confidence is not None:
        from usg.report import reconcile_with_final

        records = completion.records
        if not skip_postprocess:
            records = reconcile_with_final(
                records, read_csv_strings(output_csv), id_col=ID_COLUMN, inference_columns=completion.inference_columns
            )
        _write_confidence_reports(
            records, completion, out_dir, report_format,
            _report_meta(completion, geojson=str(geojson_path), output_csv=str(output_csv), location=context.summary(),
                         method=confidence.method, stage="inference" if skip_postprocess else "final (postprocessed)"),
            result,
        )

    manifest = run_manifest(
        command="workflow",
        inputs={"geojson": str(geojson_path)},
        outputs={"csv": str(output_csv), "input_issues": str(result.issues_csv) if result.issues_csv else None,
                 "reports": result.reports, "postprocessed": not skip_postprocess},
        completion=completion, seed=seed, batch_size=batch_size, decode=decode, context=context,
        confidence=confidence, warnings=result.warnings, location=context.location_json(),
        confidence_summary=_compact_summary(result.confidence_summary),
    )
    manifest["selection"] = {"model_id": resolved.model_id, "region": resolved.region, "reason": resolved.reason,
                             "source": resolved.source}
    if result.postprocess_summary is not None:
        keys = ("headline", "verdict", "cells", "n_changed", "changes_by_step", "n_filled_empty", "remaining_violations")
        manifest["postprocess"] = {k: result.postprocess_summary[k] for k in keys}
        manifest["outputs"]["postprocess_reports"] = result.postprocess_reports
    result.manifest_path = out_dir / RUN_MANIFEST
    write_json(result.manifest_path, manifest)

    if not keep_intermediate:
        for f in (intermediate_csv, inference_csv):
            if f != output_csv:
                f.unlink(missing_ok=True)
    return result
