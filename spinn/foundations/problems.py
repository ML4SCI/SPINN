"""Backend-independent electrostatic problem specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CircularElectrode:
    """A circular Dirichlet boundary in a two-dimensional problem."""

    name: str
    center: tuple[float, float]
    radius: float
    voltage: float


@dataclass(frozen=True)
class ElectrostaticProblem:
    """A two-dimensional Laplace problem with circular electrodes."""

    problem_id: str
    outer_kind: str
    outer_size: tuple[float, float]
    outer_voltage: float
    electrodes: tuple[CircularElectrode, ...]
    r0: float
    voltage_span: float
    description: str

    def __post_init__(self) -> None:
        if self.outer_kind not in {"disk", "rectangle"}:
            raise ValueError("outer_kind must be 'disk' or 'rectangle'")
        if any(size <= 0 for size in self.outer_size):
            raise ValueError("outer_size values must be positive")
        if self.r0 <= 0 or self.voltage_span <= 0:
            raise ValueError("r0 and voltage_span must be positive")
        if not self.electrodes:
            raise ValueError("at least one electrode is required")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``xmin, xmax, ymin, ymax`` for the outer boundary."""
        if self.outer_kind == "disk":
            radius = self.outer_size[0]
            return -radius, radius, -radius, radius
        width, height = self.outer_size
        return -width / 2, width / 2, -height / 2, height / 2

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the problem."""
        return asdict(self)

    def inside_outer(
        self, points: np.ndarray, *, include_boundary: bool = True
    ) -> np.ndarray:
        """Return a mask selecting points inside the outer boundary."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        x, y = points[:, 0], points[:, 1]
        if self.outer_kind == "disk":
            radius_squared = x**2 + y**2
            limit = self.outer_size[0] ** 2
            return (
                radius_squared <= limit if include_boundary else radius_squared < limit
            )

        xmin, xmax, ymin, ymax = self.bounds
        if include_boundary:
            return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        return (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)

    def inside_electrode(self, points: np.ndarray) -> np.ndarray:
        """Return a mask selecting points inside any electrode."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        inside = np.zeros(len(points), dtype=bool)
        for electrode in self.electrodes:
            center = np.asarray(electrode.center)
            inside |= np.sum((points - center) ** 2, axis=1) <= electrode.radius**2
        return inside

    def in_vacuum(self, points: np.ndarray) -> np.ndarray:
        """Return a mask selecting points in the open vacuum region."""
        return self.inside_outer(
            points, include_boundary=False
        ) & ~self.inside_electrode(points)


def canonical_four_rod() -> ElectrostaticProblem:
    """Return the normalized symmetric four-rod benchmark."""
    r0, voltage_span = 1.0, 1.0
    radius = 1.145 * r0
    center = r0 + radius
    return ElectrostaticProblem(
        problem_id="canonical_four_rod_v1",
        outer_kind="disk",
        outer_size=(5.0 * r0, 5.0 * r0),
        outer_voltage=0.0,
        electrodes=(
            CircularElectrode("East", (+center, 0.0), radius, +voltage_span / 2),
            CircularElectrode("West", (-center, 0.0), radius, +voltage_span / 2),
            CircularElectrode("North", (0.0, +center), radius, -voltage_span / 2),
            CircularElectrode("South", (0.0, -center), radius, -voltage_span / 2),
        ),
        r0=r0,
        voltage_span=voltage_span,
        description="Normalized symmetric four-rod benchmark.",
    )


def dale_four_rod() -> ElectrostaticProblem:
    """Return the original rectangular, unipolar problem defaults."""
    radius, center = 0.05, 0.125
    return ElectrostaticProblem(
        problem_id="dale_original_v1",
        outer_kind="rectangle",
        outer_size=(2.5, 2.0),
        outer_voltage=0.0,
        electrodes=(
            CircularElectrode("North", (0.0, +center), radius, 0.0),
            CircularElectrode("South", (0.0, -center), radius, 0.0),
            CircularElectrode("West", (-center, 0.0), radius, 300.0),
            CircularElectrode("East", (+center, 0.0), radius, 300.0),
        ),
        r0=center - radius,
        voltage_span=300.0,
        description="Exact geometry and voltage convention from the original PINN.",
    )
