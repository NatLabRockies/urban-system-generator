"""Fill the columns the model does not predict from ResStock itself, consistently with what is already known.

A derivation table (``usg-postprocess-derivations/v1``) is built during model training from the same ResStock release as the
model and shipped in the bundle (``postprocess_derivations.json``). For every column the post-processor must add, it
lists *levels* tried in order. Each level maps the values of some ``from`` columns — known inputs, model predictions,
the project location (incl. the Site Origin's ``Weather Station``) or columns filled earlier — to a value:

* **derived** (the ``from`` columns determine the value for ≥ 99% of ResStock homes), e.g. ``Geometry Story Bin`` from
  ``Geometry Stories`` or ``ISO RTO Region`` from ``County and PUMA``;
* **most common for** the ``from`` columns otherwise, e.g. ``County and PUMA`` for ``State + Weather Station``, then for
  ``State`` alone (the per-state default); finally the national most common value.

Columns are filled in the table's ``order`` so that anchors (``County and PUMA``, ``Income``) come before the columns
derived from them, which keeps all location (or income) attributes mutually consistent. Existing values are never
overwritten, and values that are not valid options are skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

DERIVATIONS_FORMAT = "usg-postprocess-derivations/v1"
WEATHER_STATION = "Weather Station"


def present(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def load_derivations(source: dict | str | Path | None) -> dict | None:
    if source is None:
        return None
    table = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if table.get("format") != DERIVATIONS_FORMAT:
        raise ValueError(f"not a {DERIVATIONS_FORMAT} table: {table.get('format')!r}")
    return table


def describe_level(level: dict, derive_purity: float) -> str:
    names = " + ".join(level["from"])
    return f"derived from {names}" if level.get("purity", 0.0) >= derive_purity else f"most common for {names}"


def _log(log: dict | None, column: str, label: str, n: int) -> None:
    if log is not None and n:
        log.setdefault(column, Counter())[label] += int(n)


def apply_derivations(
    df: pd.DataFrame,
    table: dict | None,
    *,
    context: dict[str, str] | None = None,
    allowed: dict[str, list[str]] | None = None,
    log: dict | None = None,
) -> pd.DataFrame:
    """Fill empty or missing target columns of ``df`` following ``table``; returns a new frame.

    ``context`` holds project-wide values (location columns, ``Weather Station``) used when a row has none.
    ``allowed`` (options_lookup) rejects values that are not valid options. ``log`` collects
    ``{column: Counter(method -> cells)}``.
    """
    if not table:
        return df
    table = load_derivations(table)
    sep = table.get("key_separator", "\t")
    derive_purity = float(table.get("derive_purity", 0.99))
    context = {k: str(v).strip() for k, v in (context or {}).items() if present(v)}
    df = df.copy()
    for column in table.get("order", []):
        spec = table["columns"].get(column)
        if spec is None:
            continue
        created = column not in df.columns
        if created:
            df[column] = pd.Series([None] * len(df), index=df.index, dtype=object)
        df[column] = df[column].astype(object)
        valid = set(allowed[column]) if allowed and column in allowed else None
        empty = ~df[column].map(present)
        if not empty.any():
            continue
        if column in context and (valid is None or context[column] in valid):
            df.loc[empty, column] = context[column]
            _log(log, column, "project location", int(empty.sum()))
            continue
        for level in spec.get("levels", []):
            rows = empty[empty].index
            if len(rows) == 0:
                break
            parts, usable = [], pd.Series(True, index=rows)
            for source in level["from"]:
                if source in df.columns:
                    values = df.loc[rows, source].astype(object)
                    if source in context:
                        values = values.where(values.map(present), context[source])
                elif source in context:
                    values = pd.Series(context[source], index=rows, dtype=object)
                else:
                    usable[:] = False
                    break
                usable &= values.map(present)
                parts.append(values.map(lambda v: str(v).strip() if present(v) else ""))
            if not usable.any():
                continue
            keys = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=sep)
            mapped = keys[usable].map(level["values"])
            if valid is not None:
                mapped = mapped.where(mapped.isin(valid))
            hits = mapped.dropna()
            if len(hits):
                df.loc[hits.index, column] = hits
                empty.loc[hits.index] = False
                _log(log, column, describe_level(level, derive_purity), len(hits))
        rows = empty[empty].index
        mode = spec.get("mode")
        if len(rows) and present(mode) and (valid is None or mode in valid):
            df.loc[rows, column] = mode
            _log(log, column, "most common nationally", len(rows))
        elif created and len(rows) == len(df):
            df = df.drop(columns=[column])  # nothing valid to fill: leave the column to the static defaults
    return df
