"""Coordinate Projection Network, reused from the instructor module.

The CPN and the one-argument-to-two-argument adapter are imported directly; the
adapter lets the CPN be passed to the repo's pullback operators, which expect a
``coordinate_map(points, parameter)`` signature.
"""

from __future__ import annotations

from benchmark.experiments.instructor_coordinate_projection.diagnostics import (
    _ParameterIgnoringMap,
)
from benchmark.experiments.instructor_coordinate_projection.model import (
    CoordinateProjectionNet,
)

from ..config import ModelConfig


def build_cpn(config: ModelConfig) -> CoordinateProjectionNet:
    return CoordinateProjectionNet(
        width=config.cpn_width,
        depth=config.cpn_depth,
        zero_last_layer=config.cpn_zero_last_layer,
    )


def as_two_arg_map(cpn: CoordinateProjectionNet) -> _ParameterIgnoringMap:
    """Wrap the CPN so the pullback operators can call it with a parameter slot."""

    return _ParameterIgnoringMap(cpn)
