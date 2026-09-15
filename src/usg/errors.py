"""Exception hierarchy. Each error carries the CLI exit code it maps to."""

from __future__ import annotations


class USGError(Exception):
    exit_code = 1


class SchemaError(USGError):
    """Malformed schema.json or data that cannot be encoded against it."""

    exit_code = 5


class InputValidationError(USGError):
    """User input rejected (e.g. unknown labels with ``--unknown-labels error``)."""

    exit_code = 5


class BundleError(USGError):
    """A model bundle is missing files, corrupted, or its weights don't match its schema."""

    exit_code = 4


class CompatibilityError(BundleError):
    """The bundle requires a bundle format, architecture or package version we don't support."""


class RegistryError(USGError):
    exit_code = 3


class DownloadError(USGError):
    exit_code = 3


class SelectionError(USGError):
    """No model could be resolved for the request."""

    exit_code = 3
