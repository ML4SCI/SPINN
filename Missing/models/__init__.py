"""Model registry for potential PINN architectures."""

from __future__ import annotations

from .pig import PigPotentialPINN
from .pixel import PixelPotentialPINN
from .vanilla import VanillaPotentialPINN

MODEL_KINDS = ("vanilla", "pixel", "pig")


def build_model(config):
    kind = config.model.kind.lower()
    bounds = config.domain_bounds()
    if kind == "vanilla":
        return VanillaPotentialPINN(
            width=config.model.vanilla_width,
            depth=config.model.vanilla_depth,
            activation=config.model.activation,
        )
    if kind == "pixel":
        return PixelPotentialPINN(
            domain_bounds=bounds,
            grid_resolution=config.model.pixel_grid_resolution,
            feature_dim=config.model.pixel_feature_dim,
            hidden_width=config.model.pixel_hidden_width,
            num_grids=config.model.pixel_num_grids,
        )
    if kind == "pig":
        return PigPotentialPINN(
            domain_bounds=bounds,
            num_gaussians=config.model.pig_num_gaussians,
            feature_dim=config.model.pig_feature_dim,
            hidden_width=config.model.pig_hidden_width,
            init_sigma=config.model.pig_init_sigma,
        )
    raise ValueError(f"unknown potential model kind: {kind}")

