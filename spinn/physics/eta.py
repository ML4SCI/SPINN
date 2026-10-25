"""Geometric efficiency ``eta`` estimators and harmonic purity.

``eta = (r0^2 / V0) d^2V/dx^2`` at the trap center, normalized so the ideal
quadrupole ``V = V0/(2 r0^2) (x^2 - y^2)`` has ``eta = 1``.
"""

from __future__ import annotations

import numpy as np
import torch


def center_d2v_dx2(
    field, *, device="cpu", dtype=torch.float32, create_graph: bool = False
) -> torch.Tensor:
    """``d^2 field / dx^2`` at the origin by nested autograd, shaped ``(1,)``."""
    x = torch.zeros(1, 2, device=device, dtype=dtype, requires_grad=True)
    v = field(x)
    grad = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    return torch.autograd.grad(
        grad[:, 0], x, torch.ones_like(grad[:, 0]), create_graph=create_graph
    )[0][:, 0]


def eta_autograd(
    backend, r0: float, V0: float, *, device="cpu", dtype=torch.float32
) -> float:
    return float(
        (r0**2 / V0) * center_d2v_dx2(backend, device=device, dtype=dtype).item()
    )


def eta_fit(
    points: np.ndarray,
    potential: np.ndarray,
    r0: float,
    V0: float,
    radius: float | None = None,
) -> dict:
    """Quadratic-fit eta on a center disk. Returns eta plus fit diagnostics."""
    points = np.asarray(points, dtype=float)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if radius is not None:
        keep = np.hypot(points[:, 0], points[:, 1]) <= radius
        points, potential = points[keep], potential[keep]
    # 5 quadratic coefficients => need >= 6 points; use a small margin for stability.
    if len(points) < 8:
        raise ValueError("at least 8 near-center samples are required for eta_fit")
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack([x**2 - y**2, x * y, x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, potential, rcond=None)
    a, b, *_ = coef
    predicted = design @ coef
    rss = float(np.sum((predicted - potential) ** 2))
    tss = float(np.sum((potential - potential.mean()) ** 2)) or 1.0
    return {
        "eta_fit": float(2.0 * a * r0**2 / V0),
        "a_quadrupole": float(a),
        "b_xy": float(b),
        "fit_rmse": float(np.sqrt(np.mean((predicted - potential) ** 2))),
        "fit_r2": float(1.0 - rss / tss),
        "n_points": len(points),
    }


def harmonic_purity(
    points: np.ndarray,
    potential: np.ndarray,
    radius: float,
    max_order: int = 6,
) -> dict:
    """Cylindrical-harmonic amplitudes ``|c_m|`` of ``V(r, theta)`` near center.

    Fits ``V = c0 + sum_m r^m (a_m cos(m theta) + b_m sin(m theta))`` for even
    ``m`` up to ``max_order`` (the quadrupole is ``m = 2``) and reports the
    purity ``|c2| / sum_m |c_m|``.
    """
    points = np.asarray(points, dtype=float)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    keep = np.hypot(points[:, 0], points[:, 1]) <= radius
    points, potential = points[keep], potential[keep]
    r = np.hypot(points[:, 0], points[:, 1])
    theta = np.arctan2(points[:, 1], points[:, 0])
    orders = range(2, max_order + 1, 2)
    cols = [np.ones_like(r)]
    for m in orders:
        cols += [r**m * np.cos(m * theta), r**m * np.sin(m * theta)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), potential, rcond=None)
    amps = {
        m: float(np.hypot(coef[2 * i + 1], coef[2 * i + 2]))
        for i, m in enumerate(orders)
    }
    total = sum(amps.values()) or 1.0
    return {
        "amplitudes": amps,
        "c2": amps.get(2, 0.0),
        "purity_c2": float(amps.get(2, 0.0) / total),
    }


def eta_from_backend_on_grid(
    backend,
    center_grid: np.ndarray,
    r0: float,
    V0: float,
    radius: float,
    *,
    device="cpu",
    dtype=torch.float32,
) -> dict:
    """Evaluate the backend on a center grid and return all eta estimates."""
    with torch.no_grad():
        x = torch.as_tensor(center_grid, device=device, dtype=dtype)
        v = backend(x).cpu().numpy().reshape(-1)
    fit = eta_fit(center_grid, v, r0, V0, radius=radius)
    purity = harmonic_purity(center_grid, v, radius=radius)
    return {
        "eta_autograd": eta_autograd(backend, r0, V0, device=device, dtype=dtype),
        **fit,
        "purity_c2": purity["purity_c2"],
        "c2": purity["c2"],
    }
