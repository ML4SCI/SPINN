"""Independent finite-difference Laplace solve used to validate PINN eta."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from ..geometry.export_fem import DeformedGeometry
from .eta import eta_fit, harmonic_purity


def _rasterize_dirichlet(
    geom: DeformedGeometry, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean Dirichlet mask and prescribed values on the ``(ys, xs)`` grid."""
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    fixed = np.zeros(len(pts), dtype=bool)
    value = np.zeros(len(pts))
    for polygon, voltage in geom.electrodes:
        inside = Path(polygon).contains_points(pts)
        fixed |= inside
        value[inside] = voltage
    # Everything outside the outer disk is grounded and not solved for.
    outside = np.hypot(pts[:, 0], pts[:, 1]) >= geom.outer_radius
    fixed |= outside
    value[outside] = 0.0
    return fixed.reshape(gx.shape), value.reshape(gx.shape)


def solve_laplace_fd(geom: DeformedGeometry, grid: int = 256) -> dict:
    """Five-point Laplace solve on a ``grid x grid`` box; returns field and axes."""
    R = geom.outer_radius
    xs = np.linspace(-R, R, grid)
    ys = np.linspace(-R, R, grid)
    fixed, value = _rasterize_dirichlet(geom, xs, ys)
    fixed_flat, value_flat = fixed.ravel(), value.ravel()
    n = grid * grid

    # Fixed nodes: identity rows. Free nodes: -4 on the diagonal.
    fixed_k = np.flatnonzero(fixed_flat)
    free_i, free_j = np.nonzero(~fixed)
    free_k = free_i * grid + free_j
    rows = [fixed_k, free_k]
    cols = [fixed_k, free_k]
    data = [np.ones(fixed_k.size), np.full(free_k.size, -4.0)]
    b = np.zeros(n)
    b[fixed_k] = value_flat[fixed_k]

    # +1 couplings to free neighbours; fixed neighbours fold into the RHS.
    # Neighbours beyond the box edge act grounded (contribute nothing).
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = free_i + di, free_j + dj
        inside = (ni >= 0) & (ni < grid) & (nj >= 0) & (nj < grid)
        nb = np.where(inside, ni * grid + nj, 0)
        nb_fixed = inside & fixed_flat[nb]
        nb_free = inside & ~fixed_flat[nb]
        rows.append(free_k[nb_free])
        cols.append(nb[nb_free])
        data.append(np.ones(int(nb_free.sum())))
        np.add.at(b, free_k[nb_fixed], -value_flat[nb[nb_fixed]])

    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    field = spsolve(A, b).reshape(grid, grid)
    return {
        "field": field,
        "xs": xs,
        "ys": ys,
        "h": float(xs[1] - xs[0]),
        "fixed": fixed,
    }


def fem_eta(
    geom: DeformedGeometry,
    r0: float,
    V0: float,
    grid: int = 256,
    center_radius: float = 0.25,
) -> dict:
    """Solve the FD problem and recompute eta from the solved field."""
    solved = solve_laplace_fd(geom, grid=grid)
    gx, gy = np.meshgrid(solved["xs"], solved["ys"])
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    vals = solved["field"].ravel()
    radial = np.hypot(pts[:, 0], pts[:, 1])
    # On coarse grids the nominal center disk may hold too few nodes for a
    # stable quadratic fit; widen it just enough to capture >= 16 nodes. On the
    # production grid (>= 256) the nominal radius already captures hundreds.
    fit_radius = center_radius
    while (
        np.count_nonzero(radial <= fit_radius) < 16 and fit_radius < geom.outer_radius
    ):
        fit_radius *= 1.5
    keep = radial <= fit_radius
    fit = eta_fit(pts[keep], vals[keep], r0, V0, radius=fit_radius)
    purity = harmonic_purity(pts[keep], vals[keep], radius=fit_radius)
    return {
        "eta_fem": fit["eta_fit"],
        "fem_fit_rmse": fit["fit_rmse"],
        "fem_fit_r2": fit["fit_r2"],
        "fem_purity_c2": purity["purity_c2"],
        "grid": grid,
    }
