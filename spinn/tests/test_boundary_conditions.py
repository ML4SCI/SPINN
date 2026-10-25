"""Electrode boundary sampling carries the correct +/-V0/2 voltages, the outer
boundary is grounded, and the boundary loss is the mean squared voltage misfit."""

import numpy as np
import torch

from spinn.config import Config
from spinn.geometry.collocation import sample_collocation_batch
from spinn.geometry.four_rod import build_trap
from spinn.losses.boundary import boundary_loss
from spinn.models.composed import ComposedField, IdentityShapeNetwork


def test_electrode_voltages_signed():
    config = Config()
    trap = build_trap(config)
    rng = np.random.default_rng(0)
    _, volts = trap.sample_electrode_boundaries(50, rng)
    volts = volts.reshape(-1)
    # x-axis rods at +V0/2, y-axis rods at -V0/2
    assert np.isclose(volts.max(), 0.5 * config.experiment.V0)
    assert np.isclose(volts.min(), -0.5 * config.experiment.V0)


def test_outer_boundary_grounded():
    config = Config()
    trap = build_trap(config)
    rng = np.random.default_rng(0)
    pts, volts = trap.sample_outer_boundary(100, rng)
    assert np.allclose(volts, 0.0)
    assert np.allclose(np.hypot(pts[:, 0], pts[:, 1]), trap.outer_radius)


def test_boundary_loss_of_zero_field_is_mean_squared_voltage():
    config = Config()
    trap = build_trap(config)
    batch = sample_collocation_batch(trap, np.random.default_rng(1), 16, 16, 16)

    class ZeroField(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)

    loss = boundary_loss(ComposedField(IdentityShapeNetwork(), ZeroField()), batch)
    expected = torch.mean(batch.electrode_voltages**2) + torch.mean(
        batch.outer_voltages**2
    )
    assert torch.allclose(loss, expected, atol=1e-6)
