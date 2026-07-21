"""The Jacobian hinge penalizes det(J) below eps and is zero at identity; the
constraint losses are finite and differentiable."""

import torch

from spinn.config import Config
from spinn.geometry.four_rod import build_trap
from spinn.losses.geometry_constraints import (
    anchor_loss,
    curvature_loss,
    gap_loss,
    jacobian_loss,
    min_det_jacobian,
)
from spinn.models.shape_network import build_shape_network


def _net():
    config = Config()
    trap = build_trap(config)
    return config, trap, build_shape_network(config, trap).float()


def test_jacobian_hinge_zero_at_identity():
    config, trap, net = _net()
    z = torch.randn(128, 2, requires_grad=True)
    loss = jacobian_loss(net, z, eps=config.shape_network.jacobian_eps)
    # identity has det = 1 > eps, so the hinge is zero
    assert float(loss.detach()) < 1e-8
    assert min_det_jacobian(net, z) > 0.9


def test_constraints_finite_and_backprop():
    config, trap, net = _net()
    z = torch.randn(64, 2, requires_grad=True)
    total = (
        jacobian_loss(net, z, eps=0.05)
        + gap_loss(net, trap, g_min=0.05)
        + curvature_loss(net, trap, kappa_max=5.0)
        + anchor_loss(net, trap, z)
    )
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert any(torch.any(g != 0) for g in grads) or float(total) == 0.0
