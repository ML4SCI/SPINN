from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Electrode:
    name: str
    cx: float
    cy: float
    radius: float
    voltage_sign: float  # +1 -> +V0/2 rail, -1 -> -V0/2 rail


@dataclass(frozen=True)
class FourRodTrap:
    r0: float = 1.0
    rod_radius: float = 1.145
    outer_radius: float = 5.0
    V0: float = 1.0
    # asymmetry knobs for stage D (fractional perturbations of rod centers/radii)
    center_perturb: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    radius_perturb: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def rod_radius_abs(self) -> float:
        return self.rod_radius * self.r0

    @property
    def center_distance(self) -> float:
        return self.r0 + self.rod_radius_abs

    def electrode_voltage(self, electrode: Electrode) -> float:
        """RF voltage convention for eta normalized to an ideal eta of 1.

        The eta definition uses ``V0`` as the peak-to-peak RF voltage scale for
        the quadrupole ``V = V0/(2 r0^2) (x^2 - y^2)``. Therefore each RF rail
        is driven at ``+/- V0/2`` rather than ``+/- V0``.
        """
        return 0.5 * electrode.voltage_sign * self.V0

    def electrodes(self) -> list[Electrode]:
        d = self.center_distance
        rr = self.rod_radius_abs
        # +V0/2 on x-axis (East, West), -V0/2 on y-axis (North, South)
        base = [
            ("East", d, 0.0, +1.0),
            ("West", -d, 0.0, +1.0),
            ("North", 0.0, d, -1.0),
            ("South", 0.0, -d, -1.0),
        ]
        out: list[Electrode] = []
        for (name, cx, cy, sign), dc, dr in zip(
            base, self.center_perturb, self.radius_perturb
        ):
            scale = 1.0 + dc
            out.append(Electrode(name, cx * scale, cy * scale, rr * (1.0 + dr), sign))
        return out

    # --- signed-distance helpers (positive = inside vacuum) ----------------
    def electrode_sdf(self, points: np.ndarray) -> np.ndarray:
        """Distance from each rod surface; positive outside the rod metal.

        Returns the minimum over the four rods, shaped ``(N,)``.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        dists = []
        for e in self.electrodes():
            d = np.hypot(points[:, 0] - e.cx, points[:, 1] - e.cy) - e.radius
            dists.append(d)
        return np.min(np.stack(dists, axis=1), axis=1)

    def outer_sdf(self, points: np.ndarray) -> np.ndarray:
        """Distance inside the outer disk; positive inside, zero on boundary."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return self.outer_radius - np.hypot(points[:, 0], points[:, 1])

    def in_vacuum(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask: inside the outer disk and outside every rod."""
        return (self.outer_sdf(points) > margin) & (self.electrode_sdf(points) > margin)

    # --- sampling ----------------------------------------------------------
    def sample_interior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Rejection-sample ``n`` points in the vacuum region (disk minus rods)."""
        out = np.empty((0, 2), dtype=float)
        R = self.outer_radius
        while len(out) < n:
            batch = rng.uniform(-R, R, size=(max(n * 2, 1024), 2))
            keep = self.in_vacuum(batch, margin=1e-3)
            out = np.vstack([out, batch[keep]])
        return out[:n]

    def sample_electrode_boundaries(
        self, n_per_electrode: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Points on each rod surface and their prescribed voltages (``+/-V0/2``)."""
        points = []
        voltages = []
        for e in self.electrodes():
            theta = rng.uniform(0.0, 2.0 * np.pi, n_per_electrode)
            circle = np.column_stack(
                [e.cx + e.radius * np.cos(theta), e.cy + e.radius * np.sin(theta)]
            )
            points.append(circle)
            voltages.append(np.full((n_per_electrode, 1), self.electrode_voltage(e)))
        return (
            np.vstack(points).astype(float),
            np.vstack(voltages).astype(float),
        )

    def sample_outer_boundary(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Points on the grounded outer circle (``V = 0``)."""
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        points = np.column_stack(
            [self.outer_radius * np.cos(theta), self.outer_radius * np.sin(theta)]
        )
        return points.astype(float), np.zeros((n, 1), dtype=float)

    def center_probe_grid(self, radius: float, grid: int) -> np.ndarray:
        """Regular grid of points within ``|r| <= radius`` for eta evaluation."""
        axis = np.linspace(-radius, radius, grid)
        gx, gy = np.meshgrid(axis, axis)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        keep = np.hypot(pts[:, 0], pts[:, 1]) <= radius
        return pts[keep]


def build_trap(config) -> FourRodTrap:
    """Construct a :class:`FourRodTrap` from a :class:`spinn.config.Config`."""
    exp = config.experiment
    perturb_c = (0.0, 0.0, 0.0, 0.0)
    perturb_r = (0.0, 0.0, 0.0, 0.0)
    if exp.geometry == "perturbed_four_rod":
        # mild asymmetry for the robustness/recovery stage (plan stage D)
        perturb_c = (0.06, -0.04, 0.03, -0.05)
        perturb_r = (0.05, -0.03, 0.04, -0.02)
    return FourRodTrap(
        r0=exp.r0,
        rod_radius=exp.rod_radius,
        outer_radius=exp.outer_radius,
        V0=exp.V0,
        center_perturb=perturb_c,
        radius_perturb=perturb_r,
    )
