"""Derivative-aware electrostatic and Paul-trap validation metrics."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .backends import FieldBackend


E_CHARGE = 1.602e-19
AMU = 1.66054e-27


def error_statistics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    finite = np.isfinite(prediction) & np.isfinite(reference)
    if not np.any(finite):
        raise ValueError("no finite values are available for comparison")
    error = prediction[finite] - reference[finite]
    absolute = np.abs(error)
    denominator = np.linalg.norm(reference[finite])
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
        "relative_l2": float(np.linalg.norm(error) / denominator) if denominator else float("nan"),
    }


def compare_fields(
    candidate: FieldBackend,
    reference: FieldBackend,
    points: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compare potential, electric field, and Hessian on a shared point set."""

    candidate_phi = candidate.potential(points)
    reference_phi = reference.potential(points)
    candidate_grad = candidate.gradient(points)
    reference_grad = reference.gradient(points)
    candidate_hessian = candidate.hessian(points)
    reference_hessian = reference.hessian(points)
    return {
        "potential_V": error_statistics(candidate_phi, reference_phi),
        "electric_field_V_per_unit": error_statistics(-candidate_grad, -reference_grad),
        "hessian_V_per_unit2": error_statistics(candidate_hessian, reference_hessian),
    }


def fit_local_quadratic(
    points: np.ndarray,
    potential: np.ndarray,
    center: np.ndarray | None = None,
    radius: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Fit a local quadratic and return potential, gradient, and Hessian."""

    points = np.asarray(points, dtype=float)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("fit_local_quadratic currently supports two-dimensional fields")
    if center is None:
        center = np.zeros(2)
    center = np.asarray(center, dtype=float)
    relative = points - center
    if radius is not None:
        keep = np.linalg.norm(relative, axis=1) <= radius
        relative = relative[keep]
        potential = potential[keep]
    if len(relative) < 12:
        raise ValueError("at least 12 nearby samples are required for a stable quadratic fit")

    x, y = relative[:, 0], relative[:, 1]
    design = np.column_stack([
        np.ones(len(relative)),
        x,
        y,
        0.5 * x**2,
        x * y,
        0.5 * y**2,
    ])
    coefficients, _, _, _ = np.linalg.lstsq(design, potential, rcond=None)
    hessian = np.array([
        [coefficients[3], coefficients[4]],
        [coefficients[4], coefficients[5]],
    ])
    predicted = design @ coefficients
    return {
        "potential": float(coefficients[0]),
        "gradient": coefficients[1:3],
        "hessian": hessian,
        "fit_rmse": float(np.sqrt(np.mean((predicted - potential) ** 2))),
        "n_points": int(len(relative)),
    }


def quadrupole_metrics(
    potential_hessian: np.ndarray,
    characteristic_radius: float,
    voltage_span: float,
    *,
    geometry_unit_to_m: float = 1e-2,
    ion_mass_amu: float = 40.0,
    rf_frequency_hz: float = 10.2e6,
) -> dict[str, object]:
    """Compute local quadrupole efficiency and RF secular frequencies."""

    hessian = np.asarray(potential_hessian, dtype=float)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    strength = 0.5 * (eigenvalues[-1] - eigenvalues[0])
    efficiency = characteristic_radius**2 * strength / voltage_span

    mass = ion_mass_amu * AMU
    omega_rf = 2.0 * np.pi * rf_frequency_hz
    pseudo_scale_eV = E_CHARGE / (4.0 * mass * omega_rf**2)
    hessian_per_m2 = hessian / geometry_unit_to_m**2
    pseudo_hessian_eV_per_m2 = 2.0 * pseudo_scale_eV * (hessian_per_m2.T @ hessian_per_m2)
    secular_omega2 = E_CHARGE * np.linalg.eigvalsh(pseudo_hessian_eV_per_m2) / mass
    frequencies = np.sqrt(np.clip(secular_omega2, 0.0, None)) / (2.0 * np.pi)

    trace_scale = max(np.max(np.abs(eigenvalues)), 1e-15)
    return {
        "potential_hessian": hessian.tolist(),
        "potential_hessian_eigenvalues": eigenvalues.tolist(),
        "laplace_trace_relative": float(abs(np.trace(hessian)) / trace_scale),
        "quadrupole_strength_V_per_unit2": float(strength),
        "quadrupole_efficiency": float(efficiency),
        "secular_frequencies_Hz": frequencies.tolist(),
    }


@dataclass(frozen=True)
class EscapeBarrier:
    barrier: float
    saddle_value: float
    start_value: float
    target_index: tuple[int, int]


def minimax_escape_barrier(
    values: np.ndarray,
    valid_mask: np.ndarray,
    start_index: tuple[int, int],
) -> EscapeBarrier:
    """Find the lowest discrete escape barrier on a structured 2D grid.

    The path cost is the maximum field value encountered along a path. Escape
    occurs at the outer world edge or at a valid cell neighboring an excluded
    electrode. This is a graph approximation to the lowest saddle barrier.
    """

    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.shape != valid.shape:
        raise ValueError("values and valid_mask must have the same shape")
    sy, sx = start_index
    if not valid[sy, sx]:
        raise ValueError("start_index must identify a valid grid cell")

    height, width = values.shape
    targets = np.zeros_like(valid)
    targets[0, :] = valid[0, :]
    targets[-1, :] = valid[-1, :]
    targets[:, 0] = valid[:, 0]
    targets[:, -1] = valid[:, -1]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if valid[y, x] and (
                not valid[y - 1, x]
                or not valid[y + 1, x]
                or not valid[y, x - 1]
                or not valid[y, x + 1]
            ):
                targets[y, x] = True

    costs = np.full(values.shape, np.inf)
    costs[sy, sx] = values[sy, sx]
    queue = [(float(costs[sy, sx]), sy, sx)]
    visited = np.zeros_like(valid)
    while queue:
        cost, y, x = heapq.heappop(queue)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if targets[y, x] and (y, x) != (sy, sx):
            return EscapeBarrier(
                barrier=float(cost - values[sy, sx]),
                saddle_value=float(cost),
                start_value=float(values[sy, sx]),
                target_index=(y, x),
            )
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and valid[ny, nx] and not visited[ny, nx]:
                new_cost = max(cost, float(values[ny, nx]))
                if new_cost < costs[ny, nx]:
                    costs[ny, nx] = new_cost
                    heapq.heappush(queue, (new_cost, ny, nx))
    raise ValueError("no escape target is reachable from start_index")
