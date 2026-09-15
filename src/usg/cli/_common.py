"""Shared CLI plumbing: error handling, model-selection options, output helpers. No torch imports."""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from usg.constants import MODEL_TYPES
from usg.errors import USGError

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 110}
REPORT_FORMATS = ("json", "html", "md")


def ok(message: str) -> None:
    click.echo(click.style(f"✓ {message}", fg="green"))


def warn(message: str) -> None:
    click.echo(click.style(f"! {message}", fg="yellow"), err=True)


def fail(message: str, exit_code: int) -> None:
    click.echo(click.style(f"✗ Error: {message}", fg="red"), err=True)
    sys.exit(exit_code)


def handle_errors(fn):
    """Map :class:`USGError` subclasses to their exit codes (1 generic, 3 selection, 4 bundle, 5 input)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (click.ClickException, click.exceptions.Exit, click.Abort, KeyboardInterrupt):
            raise
        except USGError as exc:
            fail(str(exc), exc.exit_code)
        except Exception as exc:
            if os.environ.get("USG_DEBUG"):
                raise
            fail(f"{type(exc).__name__}: {exc}", 1)

    return wrapper


def parse_formats(ctx, param, value: str) -> list[str]:
    """Click callback for ``--report-format json,html,md``."""
    formats = [f.strip().lower() for f in value.split(",") if f.strip()]
    bad = [f for f in formats if f not in REPORT_FORMATS]
    if bad or not formats:
        raise click.BadParameter(f"expected a comma-separated subset of {', '.join(REPORT_FORMATS)}, got {value!r}")
    return formats


@dataclass
class ModelArgs:
    model_type: str | None
    state: str
    model_id: str | None
    model_dir: Path | None
    samples: int | None
    seed: int
    device: str
    threads: int
    offline: bool
    no_fallback: bool
    registry: str | None
    unknown_labels: str

    FIELDS = (
        "model_type", "state", "model_id", "model_dir", "samples", "seed", "device", "threads", "offline",
        "no_fallback", "registry", "unknown_labels",
    )  # fmt: skip

    def request(self, context=None):
        from usg.selection import ModelRequest

        return ModelRequest(
            model_type=self.model_type,
            state=self.state,
            model_id=self.model_id,
            bundle_dir=self.model_dir,
            context=context,
            offline=self.offline,
            allow_fallback=not self.no_fallback,
        )

    def load_registry(self):
        """The registry to resolve against (not needed with ``--model-dir``)."""
        if self.model_dir is not None:
            return None
        from usg.registry import Registry

        return Registry.load(self.registry)


_MODEL_OPTIONS = [
    click.option("--model", "model_type", type=click.Choice(MODEL_TYPES), default=None,
                 help="Model type (default: the registry default for the region)."),
    click.option("--state", default="auto", show_default=True,
                 help="Two-letter state, 'national', or 'auto' (GeoJSON project state / CSV State column). "
                      "A two-letter state is also used as the State input."),
    click.option("--model-id", default=None, help="Exact registry model id, e.g. gnn-MD-2024.2-v1."),
    click.option("-d", "--model-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
                 help="Use this bundle directory (bypasses the registry)."),
    click.option("--samples", type=click.IntRange(min=1), default=None,
                 help="Samples per building (default from the bundle; TabDDPM confidence wants >= 16)."),
    click.option("--seed", type=int, default=0, show_default=True, help="Random seed."),
    click.option("--device", default="cpu", show_default=True, help="cpu, cuda, mps or auto."),
    click.option("--threads", type=click.IntRange(min=1), default=1, show_default=True,
                 help="CPU threads for torch ($USG_NUM_THREADS overrides)."),
    click.option("--offline", is_flag=True, help="Never download; embedded or cached models only ($USG_OFFLINE=1)."),
    click.option("--no-fallback", is_flag=True, help="Fail instead of falling back to the national/embedded model."),
    click.option("--registry", default=None, help="Registry file or URL ($USG_REGISTRY; default: packaged)."),
    click.option("--unknown-labels", type=click.Choice(["missing", "error"]), default="missing", show_default=True,
                 help="Treat values that are not valid options as missing, or fail."),
]  # fmt: skip


def model_options(fn):
    """Add the shared model-selection options; the command receives them as one ``model: ModelArgs`` argument."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        margs = ModelArgs(**{name: kwargs.pop(name) for name in ModelArgs.FIELDS})
        return fn(*args, model=margs, **kwargs)

    for option in reversed(_MODEL_OPTIONS):
        wrapper = option(wrapper)
    return wrapper


@dataclass
class LocationArgs:
    """Explicit location options; they override locations derived from the GeoJSON or CSV (never CSV cell values)."""

    climate_zone: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    state: str | None = None

    def overrides(self, model_state: str | None = None):
        """:class:`LocationOverrides`; a two-letter model ``--state`` (not auto/national) also sets the State input."""
        from usg.location import LocationOverrides
        from usg.selection import state_key

        state = self.state or (state_key(model_state) if model_state else None)
        return LocationOverrides(state=state, climate_zone=self.climate_zone, latitude=self.latitude,
                                 longitude=self.longitude)


def _climate_zone_callback(ctx, param, value):
    if value is None:
        return None
    from usg.constants import CLIMATE_ZONE_COLUMN
    from usg.location import canonical_location_label

    label = canonical_location_label(CLIMATE_ZONE_COLUMN, value)
    if label is None:
        raise click.BadParameter(f"{value!r} is not an ASHRAE IECC 2004 climate zone (e.g. 4A, 5B)")
    return label


def _state_callback(ctx, param, value):
    if value is None:
        return None
    from usg.constants import STATES

    if value.strip().upper() not in STATES:
        raise click.BadParameter(f"{value!r} is not a two-letter US state code")
    return value.strip().upper()


_LOCATION_OPTIONS = [
    click.option("--climate-zone", default=None, callback=_climate_zone_callback,
                 help="ASHRAE IECC 2004 climate zone, e.g. 4A (overrides the nearest weather station's zone)."),
    click.option("--latitude", type=click.FloatRange(-90, 90), default=None,
                 help="Site latitude in degrees, with --longitude (overrides the GeoJSON Site Origin point)."),
    click.option("--longitude", type=click.FloatRange(-180, 180), default=None,
                 help="Site longitude in degrees, with --latitude."),
]  # fmt: skip
_LOCATION_STATE_OPTION = click.option("--state", default=None, callback=_state_callback,
                                      help="Two-letter state (overrides the GeoJSON weather-file state).")


def location_options(fn=None, *, with_state: bool = False):
    """Add ``--climate-zone/--latitude/--longitude`` (and ``--state`` with ``with_state``, for commands without the
    model options); the command receives them as one ``location: LocationArgs`` argument."""

    def decorate(fn):
        names = ("climate_zone", "latitude", "longitude", *(("state",) if with_state else ()))

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            values = {name: kwargs.pop(name) for name in names}
            if (values["latitude"] is None) != (values["longitude"] is None):
                raise click.UsageError("--latitude and --longitude must be given together")
            return fn(*args, location=LocationArgs(**values), **kwargs)

        for option in reversed([*_LOCATION_OPTIONS, *([_LOCATION_STATE_OPTION] if with_state else [])]):
            wrapper = option(wrapper)
        return wrapper

    return decorate if fn is None else decorate(fn)
