"""Imputer implementations per model type."""

from __future__ import annotations

from usg.errors import CompatibilityError


def get_imputer_cls(model_type: str):
    if model_type == "gnn":
        from usg.backends.gnn import GNNImputer

        return GNNImputer
    if model_type == "tabddpm":
        from usg.backends.tabddpm import TabDDPMImputer

        return TabDDPMImputer
    raise CompatibilityError(f"no imputer for model_type {model_type!r}")
