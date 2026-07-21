"""Shared field interfaces, geometry helpers, and validation metrics."""

from .backends import FieldBackend, ScatteredFieldBackend, TorchFieldBackend
from .metrics import (
    compare_fields,
    fit_local_quadratic,
    minimax_escape_barrier,
    quadrupole_metrics,
)

__all__ = [
    "FieldBackend",
    "ScatteredFieldBackend",
    "TorchFieldBackend",
    "compare_fields",
    "fit_local_quadratic",
    "minimax_escape_barrier",
    "quadrupole_metrics",
]
