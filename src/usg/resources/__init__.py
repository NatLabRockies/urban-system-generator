"""Packaged data files: post-processor tables, the location lookup, the default model registry and embedded bundles."""

from __future__ import annotations

import os
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent


def postprocessor_dir() -> Path:
    return RESOURCES_DIR / "postprocessor"


def options_lookup_path() -> Path:
    return postprocessor_dir() / "options_lookup.tsv"


def consistency_rules_path() -> Path:
    return postprocessor_dir() / "consistency_rules.json"


def required_columns_path() -> Path:
    return postprocessor_dir() / "required_columns.json"


def location_lookup_path() -> Path:
    """ResStock weather-station and state location statistics (``usg-location-lookup/v1``)."""
    return RESOURCES_DIR / "location" / "location_lookup.json"


def county_puma_lookup_path() -> Path:
    """2010 Census tract centers with their ResStock ``County and PUMA`` (``usg-county-puma-tracts/v1``)."""
    return RESOURCES_DIR / "location" / "county_puma_tracts.npz"


def registry_path() -> Path:
    return RESOURCES_DIR / "registry.json"


def embedded_bundles_dir() -> Path:
    """Bundles shipped inside the wheel (``resources/bundles/<model_id>/``); ``$USG_EMBEDDED_DIR`` overrides it."""
    override = os.environ.get("USG_EMBEDDED_DIR")
    return Path(override).expanduser() if override else RESOURCES_DIR / "bundles"
