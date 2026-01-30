"""
Urban System Generator (USG)

A machine learning-based tool for completing missing building attributes
in urban energy modeling workflows.

NREL Software Record: SWR 25-36
"""

from .inference import USGInference
from .geojson_processor import GeoJSONProcessor
from .model import ScaledInputMaskedNN
from .postprocessor import USGPostProcessor

__version__ = "0.1.1"
__author__ = "Rawad El Kontar (NREL)"
__copyright__ = "Copyright (c) 2025 Alliance for Sustainable Energy, LLC"

__all__ = [
    "USGInference",
    "GeoJSONProcessor",
    "ScaledInputMaskedNN",
    "USGPostProcessor",
]