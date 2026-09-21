"""Physics backends (NN_theta): MLP baseline, PIXEL, PIG."""

from __future__ import annotations

from torch import nn

from .mlp import MLPBackend
from .pig import PigBackend
from .pixel import PixelBackend

__all__ = ["MLPBackend", "PixelBackend", "PigBackend", "build_backend"]


def build_backend(config) -> nn.Module:
    pb = config.physics_backend
    kind = pb.type.lower()
    if kind == "mlp":
        return MLPBackend(
            hidden_layers=pb.mlp_hidden_layers, width=pb.mlp_width, activation=pb.activation
        )
    if kind == "pixel":
        return PixelBackend(
            outer_radius=config.experiment.outer_radius,
            grid_resolution=pb.pixel_grid_resolution,
            feature_dim=pb.pixel_feature_dim,
            num_grids=pb.pixel_num_grids,
            decoder_layers=pb.pixel_decoder_layers,
            decoder_width=pb.pixel_decoder_width,
        )
    if kind == "pig":
        return PigBackend(
            outer_radius=config.experiment.outer_radius,
            num_gaussians=pb.K,
            feature_dim=pb.feature_dim,
            init_sigma=pb.init_sigma,
            decoder_layers=pb.decoder_layers,
            decoder_width=pb.decoder_width,
            symmetry=pb.symmetry,
        )
    raise ValueError(f"unknown physics backend {pb.type!r}")
