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
    """return grids for electrode and outer dirichlet."""
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    fixed = np.zeros(len(pts), dtype=bool)
    value = np.zeros(len(pts), dtype=float)

    # electrodes (interior of each closed polygon)
    for polygon, voltage in geom.electrodes:
        inside = Path(polygon).contains_points(pts)
        fixed |= inside
        value[inside] = voltage

    # outside the outer disk is excluded (treated as Dirichlet 0 / not solved)
    r = np.hypot(pts[:, 0], pts[:, 1])
    outside = r >= geom.outer_radius
    fixed |= outside
    value[outside] = 0.0
    shape = gx.shape
    return fixed.reshape(shape), value.reshape(shape)


def solve_laplace_fd(geom: DeformedGeometry, grid: int = 256) -> dict:
    """Solve Laplace on the grid; return the field and coordinate axes."""
    R = geom.outer_radius
    xs = np.linspace(-R, R, grid)
    ys = np.linspace(-R, R, grid)
    h = xs[1] - xs[0]
    fixed, value = _rasterize_dirichlet(geom, xs, ys)

    n = grid * grid
    idx = np.arange(n).reshape(grid, grid)
    fixed_flat = fixed.ravel()
    value_flat = value.ravel()

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    b = np.zeros(n)

    # Fixed nodes: identity rows.
    fixed_k = idx.ravel()[fixed_flat]
    rows.append(fixed_k)
    cols.append(fixed_k)
    data.append(np.ones(fixed_k.size))
    b[fixed_k] = value_flat[fixed_flat]

    # Free nodes: -4 on the diagonal, +1 to free neighbours, fixed neighbours
    # folded into the right-hand side. Boundary-of-box neighbours act grounded.
    free_mask = ~fixed
    free_k = idx[free_mask]
    rows.append(free_k)
    cols.append(free_k)
    data.append(np.full(free_k.size, -4.0))

    for shift, axis in ((-1, 0), (1, 0), (-1, 1), (1, 1)):
        neighbour = np.roll(idx, shift, axis=axis)
        # mark wrapped (off-box) neighbours as invalid -> grounded contribution 0
        valid = np.ones((grid, grid), dtype=bool)
        if axis == 0:
            if shift == -1:
                valid[-1, :] = False
            else:
                valid[0, :] = False
        else:
            if shift == -1:
                valid[:, -1] = False
            else:
                valid[:, 0] = False
        nb = neighbour[free_mask]
        vd = valid[free_mask]
        nb_fixed = fixed_flat[nb] & vd
        nb_free = (~fixed_flat[nb]) & vd
        # contribution from free neighbours -> matrix entry +1
        rows.append(free_k[nb_free])
        cols.append(nb[nb_free])
        data.append(np.ones(int(nb_free.sum())))
        # contribution from fixed neighbours -> rhs (-value)
        np.add.at(b, free_k[nb_fixed], -value_flat[nb[nb_fixed]])

    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    sol = spsolve(A, b)
    field = sol.reshape(grid, grid)
    return {"field": field, "xs": xs, "ys": ys, "h": float(h), "fixed": fixed}


def fem_eta(geom: DeformedGeometry, r0: float, V0: float, grid: int = 256, center_radius: float = 0.25) -> dict:
    """Solve the FD problem and recompute eta from the solved field."""
    solved = solve_laplace_fd(geom, grid=grid)
    xs, ys, field = solved["xs"], solved["ys"], solved["field"]
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    vals = field.ravel()
    radial = np.hypot(pts[:, 0], pts[:, 1])
    # On coarse grids the nominal center disk may hold too few nodes for a
    # stable quadratic fit; widen it just enough to capture >= 16 nodes. On the
    # production grid (>= 256) the nominal radius already captures hundreds.
    fit_radius = center_radius
    while np.count_nonzero(radial <= fit_radius) < 16 and fit_radius < geom.outer_radius:
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
        "_solved": solved,
    }
