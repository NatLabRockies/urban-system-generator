"""Restore columns the model was not trained to predict.

A model trained on the reduced target set never sees the stock's redundant columns: constants, coarser
views of a finer column, relabellings of a column kept under another name, and composites of several
predictions. URBANopt still requires every one of them, so they are put back here by table lookup.

This is not imputation. Each rule was verified single-valued across every building in the release it was
built from, so a lookup either produces the stock's own value or produces nothing and says so. A value the
caller supplied is never overwritten - as elsewhere in usg, the caller's own data wins and a disagreement
is reported rather than silently resolved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

RESOURCE = Path(__file__).parent / "resources" / "postprocessor" / "restoration_table.json"


@dataclass(frozen=True)
class Restored:
    """One cell this step filled, for the audit record."""

    row: int
    column: str
    value: str
    source: str          # the column(s) it was derived from, or "constant"


@dataclass(frozen=True)
class Unmapped:
    """A row whose determining values are a combination the stock never contains.

    The column is left blank rather than guessed, and reported so the caller can fall back to the
    post-processor's default fill. Silence here would ship an incomplete mapping CSV to URBANopt.
    """

    row: int
    column: str
    key: str
    source: str


@dataclass(frozen=True)
class Conflict:
    """A caller-supplied value the stock's own table disagrees with."""

    row: int
    column: str
    given: str
    stock: str
    source: str


@lru_cache(maxsize=4)
def load_table(path: str | None = None) -> dict:
    target = Path(path) if path else RESOURCE
    if not target.is_file():
        return {"constants": {}, "rules": [], "separator": "\u001f"}
    return json.loads(target.read_text())


def restorable_columns(path: str | None = None) -> list[str]:
    table = load_table(path)
    return sorted({*table.get("constants", {}), *(r["column"] for r in table.get("rules", []))})


@dataclass
class Report:
    """What the step did, for the audit trail."""

    filled: list[Restored] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    unmapped: list[Unmapped] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unmapped and not self.missing_sources


def restore(frame: pd.DataFrame, *, path: str | None = None) -> tuple[pd.DataFrame, Report]:
    """Fill every restorable column that ``frame`` is missing or has left blank.

    Returns the frame and a :class:`Report`. Rules are applied in dependency order, so a rule may use a
    column an earlier rule restored. Anything the table cannot resolve is recorded, never guessed.
    """
    out = frame.copy()
    table = load_table(path)
    sep = table.get("separator", "\u001f")
    report = Report()

    for column, value in table.get("constants", {}).items():
        _apply(out, column, pd.Series(value, index=out.index), "constant", report)

    # a rule may depend on a column another rule restores, so keep passing while progress is made
    pending = list(table.get("rules", []))
    while pending:
        progressed = []
        for rule in pending:
            if not all(src in out.columns for src in rule["from"]):
                progressed.append(rule)
                continue
            keys = out[rule["from"][0]].astype(str)
            if len(rule["from"]) > 1:
                keys = out[rule["from"]].astype(str).agg(sep.join, axis=1)
            source = ", ".join(rule["from"])
            values = keys.map(rule["map"])
            _apply(out, rule["column"], values, source, report)
            for pos, (idx, got) in enumerate(values.items()):
                if got is None or pd.isna(got):
                    report.unmapped.append(
                        Unmapped(row=pos, column=rule["column"],
                                 key=str(keys.loc[idx]).replace(sep, " | "), source=source)
                    )
        if len(progressed) == len(pending):
            report.missing_sources = sorted(
                {src for rule in progressed for src in rule["from"] if src not in out.columns}
            )
            break
        pending = progressed
    return out, report


def _apply(
    frame: pd.DataFrame,
    column: str,
    values: pd.Series,
    source: str,
    report: Report,
) -> None:
    if column not in frame.columns:
        frame[column] = pd.NA
    current = frame[column]
    blank = current.isna() | (current.astype(str).str.strip() == "")
    for pos, (idx, want) in enumerate(values.items()):
        if want is None or pd.isna(want):
            continue
        if blank.loc[idx]:
            frame.loc[idx, column] = want
            report.filled.append(Restored(row=pos, column=column, value=str(want), source=source))
        elif str(current.loc[idx]).strip() != str(want):
            report.conflicts.append(
                Conflict(row=pos, column=column, given=str(current.loc[idx]), stock=str(want), source=source)
            )
