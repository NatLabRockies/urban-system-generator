"""Tiny synthetic schema, data and model configs for tests (used by the package tests and the model training pipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from usg.schema import ColumnSpec, QuantileNormal, Schema

STATES = ("MD", "VA", "DC")
BUILDING_TYPES = ("Single-Family Detached", "Single-Family Attached", "Multi-Family with 5+ Units")
VINTAGES = ("<1940", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s")
FUELS = ("Electricity", "Natural Gas", "Fuel Oil", "Propane", "None")
HEATING_TYPES = ("Ducted Heat Pump", "Ducted Heating", "Non-Ducted Heating", "None")
GARAGES = ("1 Car", "2 Car", "3 Car", "None")
BEDROOMS = ("1", "2", "3", "4", "5")


def tiny_dataframe(n: int = 512, seed: int = 0, with_numeric: bool = True) -> pd.DataFrame:
    """Synthetic buildings with learnable dependencies between columns."""
    rng = np.random.default_rng(seed)
    state = rng.choice(STATES, size=n, p=[0.5, 0.35, 0.15])
    btype = np.where(
        state == "DC",
        rng.choice(BUILDING_TYPES, size=n, p=[0.1, 0.3, 0.6]),
        rng.choice(BUILDING_TYPES, size=n, p=[0.6, 0.25, 0.15]),
    )
    vintage = rng.choice(VINTAGES, size=n)
    fuel = np.where(
        state == "VA",
        rng.choice(FUELS, size=n, p=[0.6, 0.25, 0.05, 0.08, 0.02]),
        rng.choice(FUELS, size=n, p=[0.3, 0.55, 0.08, 0.05, 0.02]),
    )
    heating = np.select(
        [fuel == "Electricity", fuel == "None"],
        [rng.choice(["Ducted Heat Pump", "Non-Ducted Heating"], size=n, p=[0.7, 0.3]), "None"],
        default="Ducted Heating",
    )
    garage = np.where(
        btype == "Multi-Family with 5+ Units", "None", rng.choice(GARAGES, size=n, p=[0.3, 0.3, 0.1, 0.3])
    )
    bedrooms = np.where(
        btype == "Multi-Family with 5+ Units",
        rng.choice(BEDROOMS[:3], size=n, p=[0.4, 0.4, 0.2]),
        rng.choice(BEDROOMS[1:], size=n, p=[0.2, 0.45, 0.25, 0.1]),
    )
    df = pd.DataFrame(
        {
            "State": state,
            "Geometry Building Type RECS": btype,
            "Vintage": vintage,
            "Heating Fuel": fuel,
            "HVAC Heating Type": heating,
            "Geometry Garage": garage,
            "Bedrooms": bedrooms,
        }
    )
    if with_numeric:
        base = 400.0 * bedrooms.astype(int)
        df["sqft"] = np.round(base * rng.lognormal(0.0, 0.25, size=n) + 200.0)
    return df


def tiny_schema(with_numeric: bool = True, seed: int = 0) -> Schema:
    cols = [
        ColumnSpec("State", "conditioning", "categorical", vocab=STATES),
        ColumnSpec("Geometry Building Type RECS", "imputable", "categorical", vocab=BUILDING_TYPES),
        ColumnSpec("Vintage", "imputable", "categorical", vocab=VINTAGES, ordinal=True),
        ColumnSpec("Heating Fuel", "imputable", "categorical", vocab=FUELS),
        ColumnSpec("HVAC Heating Type", "imputable", "categorical", vocab=HEATING_TYPES),
        ColumnSpec("Geometry Garage", "imputable", "categorical", vocab=GARAGES),
        ColumnSpec("Bedrooms", "imputable", "categorical", vocab=BEDROOMS, ordinal=True),
    ]
    if with_numeric:
        sqft = tiny_dataframe(2048, seed=seed)["sqft"].to_numpy()
        cols.append(
            ColumnSpec(
                "sqft", "imputable", "numeric", transform=QuantileNormal.fit(sqft, 200), clip=(100.0, 15000.0), unit="ft2"
            )
        )
    return Schema(cols, data_release="test", groups={"heating": ["Heating Fuel", "HVAC Heating Type"]})


def tiny_config(model_type: str) -> dict:
    arch = {"d_small": 4, "d_large": 8, "d_numeric": 4, "hidden": 32, "n_blocks": 1, "expansion": 2, "dropout": 0.0}
    if model_type == "gnn":
        return {"model_type": "gnn", "arch_version": 1, "arch": arch, "inference": {"decode": "argmax", "rounds": 4}}
    if model_type == "tabddpm":
        return {
            "model_type": "tabddpm",
            "arch_version": 1,
            "arch": {**arch, "t_embed": 16},
            "diffusion": {"timesteps": 50, "schedule": "cosine"},
            "inference": {"sample_steps": 10, "default_samples": 8, "max_samples": 64, "aggregate": "medoid"},
        }
    raise ValueError(model_type)
