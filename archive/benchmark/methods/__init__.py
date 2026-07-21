"""Interchangeable design methods scored by the benchmark.

Each method implements :class:`~benchmark.methods.base.ShapeMethod`: given a
:class:`~benchmark.task.DesignTask`, it reports the design objective and its
shape gradient, plus the wall-clock it took. The driver in
:mod:`benchmark.run_benchmark` treats them all identically.
"""

from .base import ShapeMethod
from .devsim import DevsimFiniteDifferenceMethod
from .pinn import CoordinateProjectionPINNMethod
from .instructor_coordinate_projection import InstructorCoordinateProjectionMethod

__all__ = [
    "ShapeMethod",
    "DevsimFiniteDifferenceMethod",
    "CoordinateProjectionPINNMethod",
    "InstructorCoordinateProjectionMethod",
]
