"""Shared constants: format identifiers, states, environment variables."""

from __future__ import annotations

import re

BUNDLE_FORMAT = "usg-bundle/v1"
SCHEMA_FORMAT = "usg-schema/v1"
REGISTRY_FORMAT = "usg-registry/v1"
METRICS_FORMAT = "usg-metrics/v1"
CALIBRATION_FORMAT = "usg-confidence-calibration/v1"
CALIBRATION_V2_FORMAT = "usg-confidence-calibration/v2"
SENSITIVITY_FORMAT = "usg-sensitivity/v1"

MODEL_TYPES = ("gnn", "tabddpm")
NATIONAL = "national"
MODEL_ID_RE = re.compile(r"^(gnn|tabddpm)-([A-Z]{2}|national)-(\d{4}\.\d+)-v(\d+)$")

# 50 states + DC (ResStock 2024.2 covers 48 contiguous states + DC; AK/HI fall back to national).
STATES = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
)

# Column names used as conditioning inputs (ResStock parameter names).
STATE_COLUMN = "State"
CLIMATE_ZONE_COLUMN = "ASHRAE IECC Climate Zone 2004"

ID_COLUMN = "Building"

ENV_CACHE_DIR = "USG_CACHE_DIR"
ENV_REGISTRY = "USG_REGISTRY"
ENV_OFFLINE = "USG_OFFLINE"
ENV_NUM_THREADS = "USG_NUM_THREADS"
