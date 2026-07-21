"""eta_fit recovers eta=1 on a perfect normalized quadrupole, and the FD solver
recovers a clean, grid-convergent quadrupole on the circular four-rod trap."""

import numpy as np

from spinn.config import Config
from spinn.geometry.export_fem import reference_geometry
from spinn.geometry.four_rod import build_trap
from spinn.physics.eta import eta_fit, harmonic_purity
from spinn.physics.fem_reference import fem_eta


def test_eta_fit_perfect_quadrupole():
    r0, V0 = 1.0, 1.0
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.25, 0.25, size=(3000, 2))
    v = (V0 / (2 * r0**2)) * (pts[:, 0] ** 2 - pts[:, 1] ** 2)
    fit = eta_fit(pts, v, r0, V0, radius=0.25)
    assert abs(fit["eta_fit"] - 1.0) < 1e-6
    assert fit["fit_r2"] > 0.999
    purity = harmonic_purity(pts, v, radius=0.25)
    assert purity["purity_c2"] > 0.999


def test_fd_solver_recovers_clean_quadrupole():
    config = Config()
    trap = build_trap(config)
    geom = reference_geometry(trap)
    res = fem_eta(geom, r0=1.0, V0=1.0, grid=160, center_radius=0.25)
    # circular-rod field near center is a near-pure quadrupole
    assert res["fem_fit_r2"] > 0.999
    assert res["fem_purity_c2"] > 0.95
    assert res["eta_fem"] > 0.0  # positive curvature toward the +V0/2 axis
