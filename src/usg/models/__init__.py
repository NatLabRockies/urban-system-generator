"""Model architectures (pure torch). Shared with the model training pipeline, so changes here are part of the bundle contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from usg.errors import CompatibilityError

if TYPE_CHECKING:
    from torch import nn

    from usg.schema import Schema

# model_type -> architecture versions this package can load
SUPPORTED_ARCH: dict[str, tuple[int, ...]] = {"gnn": (1,), "tabddpm": (1,)}


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

        return GNNImputerNet(*args)
    from usg.models.tabddpm import TabDDPMDenoiser

    return TabDDPMDenoiser(*args)


def build_schedule(config: dict):
    from usg.models.diffusion import DiffusionSchedule
    from usg.models.tabddpm import DEFAULT_DIFFUSION

    d = {**DEFAULT_DIFFUSION, **config.get("diffusion", {})}
    return DiffusionSchedule(timesteps=d["timesteps"], schedule=d["schedule"])
