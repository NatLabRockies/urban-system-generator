"""
Urban System Generator (USG)

A machine learning-based tool for completing missing building attributes
in urban energy modeling workflows.

NREL Software Record: SWR 25-36
"""

from .inference import USGInference
from .geojson_processor import GeoJSONProcessor
from .model import ScaledInputMaskedNN

__version__ = "0.1.0"
__author__ = "NREL"

__all__ = [
    "USGInference",
    "GeoJSONProcessor",
    "ScaledInputMaskedNN",
]