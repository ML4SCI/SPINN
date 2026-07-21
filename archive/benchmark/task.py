"""Shared design task definition and scorecard data model.

Everything here is method-agnostic: it describes *what* is being optimized and
*how results are recorded*, but nothing about how any particular method solves
it. Keeping this separate is what lets the PINN and the finite-element solver
be scored on exactly the same terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DesignTask:
    """A one-parameter Paul-trap shape-design problem.

    The single design variable is the electrode-center ``distance``. The design
    objective is the local quadrupole efficiency at the trap center, the same
    scalar used throughout the coordinate-projection code. The benchmark is
    deliberately one-dimensional first so that a classical finite-difference
    gradient is cheap and unambiguous; the same scorecard generalizes to a
    parameter vector later.
    """

    distance: float = 0.125
    distance_min: float = 0.11
    distance_max: float = 0.145
    radius: float = 0.05
    rf_voltage: float = 300.0
    gradient_epsilon: float = 0.0025
    world_size_x: float = 2.5
    world_size_y: float = 2.0

    def stencil(self) -> list[float]:
        """Centered five-point stencil ``[-2h, -h, +h, +2h]`` plus the center."""

        h = self.gradient_epsilon
        return [
            self.distance - 2.0 * h,
            self.distance - h,
            self.distance + h,
            self.distance + 2.0 * h,
        ]

    def validate_stencil_in_range(self) -> None:
        points = self.stencil()
        if points[0] < self.distance_min or points[-1] > self.distance_max:
            raise ValueError(
                "finite-difference stencil leaves the design interval; "
                "reduce gradient_epsilon or move the evaluation distance"
            )


@dataclass
class GradientComparison:
    """How a method's shape gradient compares with the classical reference."""

    method_gradient: float
    reference_gradient: float

    @property
    def absolute_error(self) -> float:
        return abs(self.method_gradient - self.reference_gradient)

    @property
    def relative_error(self) -> float:
        denom = max(abs(self.reference_gradient), 1e-12)
        return self.absolute_error / denom

    @property
    def sign_agrees(self) -> bool:
        """Whether both gradients point the optimizer the same direction."""

        return (self.method_gradient >= 0) == (self.reference_gradient >= 0)


@dataclass
class MethodResult:
    """Everything one method reports for one task, before cross-comparison."""

    name: str
    available: bool
    objective: float | None = None
    gradient: float | None = None
    setup_seconds: float = 0.0
    eval_seconds: float = 0.0
    deterministic: bool = True
    notes: str = ""
    # Filled in by the driver once the reference is known.
    gradient_vs_reference: GradientComparison | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.gradient_vs_reference is not None:
            data["gradient_vs_reference"] = {
                "method_gradient": self.gradient_vs_reference.method_gradient,
                "reference_gradient": self.gradient_vs_reference.reference_gradient,
                "absolute_error": self.gradient_vs_reference.absolute_error,
                "relative_error": self.gradient_vs_reference.relative_error,
                "sign_agrees": self.gradient_vs_reference.sign_agrees,
            }
        return data


@dataclass
class Scorecard:
    """The full benchmark result: the task, the reference, and every method."""

    task: DesignTask
    reference_method: str
    results: list[MethodResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": asdict(self.task),
            "reference_method": self.reference_method,
            "results": [result.to_dict() for result in self.results],
        }
