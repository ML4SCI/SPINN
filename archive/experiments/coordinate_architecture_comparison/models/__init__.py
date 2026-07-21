"""Model registry for the three coordinate-projection architectures."""

from __future__ import annotations

from ..config import ExperimentConfig
from .base import CoordinateProjectionModel
from .pig_model import PigCoordinateProjection
from .pixel_model import PixelCoordinateProjection
from .vanilla_wrapper import VanillaCoordinateProjection

MODEL_KINDS = ("vanilla", "pixel", "pig")


def build_model(config: ExperimentConfig) -> CoordinateProjectionModel:
    m = config.model
    if m.kind == "vanilla":
        return VanillaCoordinateProjection(
            width=m.mlp_width,
            depth=m.mlp_depth,
            zero_last_layer=m.zero_last_layer,
        )
    if m.kind == "pixel":
        return PixelCoordinateProjection(
            domain_bounds=config.domain_bounds(),
            grid_resolution=m.pixel_grid_resolution,
            feature_dim=m.pixel_feature_dim,
            hidden_width=m.pixel_hidden_width,
        )
    if m.kind == "pig":
        return PigCoordinateProjection(
            domain_bounds=config.domain_bounds(),
            num_gaussians=m.pig_num_gaussians,
            feature_dim=m.pig_feature_dim,
            hidden_width=m.pig_hidden_width,
            init_sigma=m.pig_init_sigma,
        )
    raise ValueError(f"unknown model kind: {m.kind!r}, expected one of {MODEL_KINDS}")


__all__ = [
    "CoordinateProjectionModel",
    "VanillaCoordinateProjection",
    "PixelCoordinateProjection",
    "PigCoordinateProjection",
    "build_model",
    "MODEL_KINDS",
]
