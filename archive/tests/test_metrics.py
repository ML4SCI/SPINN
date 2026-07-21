from __future__ import annotations

import numpy as np

from PINN.common.metrics import (
    AMU,
    E_CHARGE,
    fit_local_quadratic,
    minimax_escape_barrier,
    quadrupole_metrics,
)


def test_local_quadratic_fit_recovers_gradient_and_hessian():
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.1, 0.1, size=(500, 2))
    x, y = points[:, 0], points[:, 1]
    potential = 2.0 + 3.0 * x - 4.0 * y + 0.5 * 8.0 * x**2 + 2.0 * x * y - 0.5 * 8.0 * y**2
    fit = fit_local_quadratic(points, potential)
    np.testing.assert_allclose(fit["gradient"], [3.0, -4.0], atol=1e-11)
    np.testing.assert_allclose(fit["hessian"], [[8.0, 2.0], [2.0, -8.0]], atol=1e-11)


def test_quadrupole_efficiency_normalization():
    hessian = np.diag([100.0, -100.0])
    metrics = quadrupole_metrics(
        hessian,
        characteristic_radius=0.1,
        voltage_span=1.0,
    )
    assert np.isclose(metrics["quadrupole_efficiency"], 1.0)
    assert metrics["laplace_trace_relative"] == 0.0
    mass = 40.0 * AMU
    omega_rf = 2.0 * np.pi * 10.2e6
    curvature_per_m2 = 100.0 / 1e-4
    expected_hz = (
        E_CHARGE * curvature_per_m2 / (np.sqrt(2.0) * mass * omega_rf)
    ) / (2.0 * np.pi)
    np.testing.assert_allclose(
        metrics["secular_frequencies_Hz"],
        [expected_hz, expected_hz],
    )


def test_minimax_escape_finds_lowest_saddle_path():
    values = np.array([
        [9.0, 9.0, 3.0, 9.0, 9.0],
        [9.0, 8.0, 2.0, 8.0, 9.0],
        [9.0, 7.0, 0.0, 7.0, 9.0],
        [9.0, 8.0, 4.0, 8.0, 9.0],
        [9.0, 9.0, 5.0, 9.0, 9.0],
    ])
    result = minimax_escape_barrier(values, np.ones_like(values, dtype=bool), (2, 2))
    assert result.saddle_value == 3.0
    assert result.barrier == 3.0
