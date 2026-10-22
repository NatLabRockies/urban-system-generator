"""Model architectures (pure torch). Shared with the model training pipeline, so changes here are part of the bundle contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from usg.errors import CompatibilityError

if TYPE_CHECKING:
    from torch import nn

    from usg.schema import Schema

# model_type -> architecture versions this package can load
SUPPORTED_ARCH: dict[str, tuple[int, ...]] = {"gnn": (1, 2), "tabddpm": (1,)}


def build_model(schema: Schema, config: dict) -> nn.Module:
    """Instantiate an untrained network for ``schema`` from a bundle ``config.json`` dict."""
    model_type = config.get("model_type")
    arch_version = int(config.get("arch_version", 1))
    if model_type not in SUPPORTED_ARCH:
        raise CompatibilityError(f"unknown model_type {model_type!r}; supported: {sorted(SUPPORTED_ARCH)}")
    if arch_version not in SUPPORTED_ARCH[model_type]:
        raise CompatibilityError(
            f"{model_type} arch_version {arch_version} is not supported by this urban-system-generator "
            f"(supported: {SUPPORTED_ARCH[model_type]}); upgrade urban-system-generator"
        )
    args = (
        schema.cardinalities,
        len(schema.num_columns),
        schema.cat_imputable.tolist(),
        schema.num_imputable.tolist(),
        config.get("arch", {}),
    )
    if model_type == "gnn":
        from usg.models.gnn import GNNImputerNet

        return GNNImputerNet(*args, film_cols=_film_cols(schema, config, arch_version))
    from usg.models.tabddpm import TabDDPMDenoiser

    return TabDDPMDenoiser(*args)


def _film_cols(schema: Schema, config: dict, arch_version: int) -> list[int]:
    """Resolve ``arch.film_columns`` (column names) to positions among the categorical columns.

    Names rather than indices, because a position depends on the column order of whichever schema the
    bundle was built with. An unknown name is an error: silently dropping it would train or serve a model
    that is quietly missing the conditioning it was configured for.
    """
    names = list(config.get("arch", {}).get("film_columns", ()))
    if not names:
        return []
    if arch_version < 2:
        raise CompatibilityError(
            f"arch.film_columns needs gnn arch_version >= 2, got {arch_version}; "
            "bump the bundle's arch_version or drop film_columns"
        )
    missing = [n for n in names if n not in schema]
    if missing:
        raise CompatibilityError(f"arch.film_columns names columns absent from the schema: {missing}")
    not_cat = [n for n in names if schema.column(n).type != "categorical"]
    if not_cat:
        raise CompatibilityError(f"arch.film_columns must be categorical columns, got {not_cat}")
    return [schema.cat_index(n) for n in names]


def build_schedule(config: dict):
    from usg.models.diffusion import DiffusionSchedule
    from usg.models.tabddpm import DEFAULT_DIFFUSION

    d = {**DEFAULT_DIFFUSION, **config.get("diffusion", {})}
    return DiffusionSchedule(timesteps=d["timesteps"], schedule=d["schedule"])
