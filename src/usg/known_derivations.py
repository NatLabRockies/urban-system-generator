"""Filling the cells a user's own inputs already determine, before the model is asked.

ResStock fixes many characteristics once a few are known: a multifamily building always has
``Geometry Garage = None`` and ``Corridor = Double-Loaded Interior``, in every one of the 101,069
multifamily buildings in the 2024.2 stock. The models are trained on that premise -- their mask sampler
marks a dependent *known* whenever its determinant is -- so asking them for such a cell puts them in a
state they never saw in training, and they answer from the national prior instead. That is how a
20-unit apartment block acquired a three-car garage.

So the rules here are applied *before* inference, not after: every cell they fill is one more thing the
model can condition on, which helps the cells that genuinely are uncertain.

A value the user supplied is never overwritten. Where the user disagrees with the stock, the user wins
and the disagreement is reported -- ResStock is a sample, not a law, and a real building may be a
combination it never drew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

RESOURCE = Path(__file__).with_name("resources") / "postprocessor" / "derivations_from_knowns.json"


@dataclass(frozen=True)
class Conflict:
    """A cell the user supplied that the stock never pairs with their other inputs."""

    row: int
    column: str
    given: str
    stock: str
    because: dict[str, str]

    def describe(self) -> str:
        why = ", ".join(f"{k}={v}" for k, v in self.because.items())
        return (f"{self.column}={self.given!r} does not occur with {why} in ResStock "
                f"(which always has {self.stock!r}); your value was kept")


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> list[dict]:
    source = Path(path) if path else RESOURCE
    if not source.is_file():
        return []
    return json.loads(source.read_text()).get("rules", [])


def derive(df: pd.DataFrame, *, rules: list[dict] | None = None) -> tuple[pd.DataFrame, list[Conflict]]:
    """Fill every cell the known columns determine; return the frame and any user/stock conflicts.

    Rules are applied least-specific first, so a rule keyed on two columns refines one keyed on a single
    column rather than fighting it.
    """
    rules = load_rules() if rules is None else rules
    if not rules or df.empty:
        return df, []
    out = df.copy()
    conflicts: list[Conflict] = []
    for i in range(len(out)):
        known = {c: out.at[out.index[i], c] for c in out.columns if _present(out.at[out.index[i], c])}
        derived: dict[str, tuple[str, dict]] = {}
        for rule in sorted(rules, key=lambda r: len(r["when"])):
            if all(str(known.get(k)) == str(v) for k, v in rule["when"].items()):
                for col, val in rule["then"].items():
                    derived[col] = (val, rule["when"])
        for col, (val, because) in derived.items():
            if col not in out.columns:
                continue
            current = out.at[out.index[i], col]
            if not _present(current):
                out.at[out.index[i], col] = val
            elif str(current) != str(val):
                conflicts.append(Conflict(i, col, str(current), str(val), dict(because)))
    return out, conflicts


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip() != ""
