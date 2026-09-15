"""urban-system-generator: infer missing residential building characteristics with pretrained ResStock models.

Importing this package must stay cheap: torch is only imported when a model is actually loaded
(see ``usg._torch``). Public names are resolved lazily.
"""

from usg.__about__ import __version__

__all__ = [
    "Bundle",
    "ConfidenceEstimator",
    "ModelRequest",
    "Registry",
    "Schema",
    "__version__",
    "complete_dataframe",
    "load_imputer",
    "resolve",
    "run_workflow",
]

_LAZY = {
    "ConfidenceEstimator": "usg.confidence",
    "ModelRequest": "usg.selection",
    "Registry": "usg.registry",
    "complete_dataframe": "usg.pipeline",
    "resolve": "usg.selection",
    "run_workflow": "usg.pipeline",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    if name == "Bundle":
        from usg.bundle import Bundle

        return Bundle
    if name == "Schema":
        from usg.schema import Schema

        return Schema
    if name == "load_imputer":
        from usg.bundle import load_imputer

        return load_imputer
    raise AttributeError(f"module 'usg' has no attribute {name!r}")
