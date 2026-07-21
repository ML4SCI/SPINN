"""The residual shape network starts as the exact identity (zero-init final)."""

import torch

from spinn.config import Config
from spinn.geometry.four_rod import build_trap
from spinn.models.shape_network import build_shape_network
from spinn.physics.laplace import map_jacobian


def test_starts_at_identity():
    config = Config()
    trap = build_trap(config)
    net = build_shape_network(config, trap).float()
    z = torch.randn(64, 2, requires_grad=True)
    x = net(z)
    assert torch.allclose(x, z, atol=1e-6)


def test_identity_jacobian_is_one():
    config = Config()
    trap = build_trap(config)
    net = build_shape_network(config, trap).float()
    z = torch.randn(32, 2, requires_grad=True)
    _, jac = map_jacobian(net, z)
    det = torch.linalg.det(jac)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_symmetry_preserves_quadrupole_planes():
    config = Config()
    config.shape_network.symmetric = True
    trap = build_trap(config)
    net = build_shape_network(config, trap).float()
    # break the zero init so the deformation is nonzero, then check mirror parity
    for p in net.network[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
    z = torch.randn(50, 2, requires_grad=True)
    disp = net.displacement(z)
    z_mx = z.detach().clone()
    z_mx[:, 0] *= -1
    z_mx.requires_grad_(True)
    disp_mx = net.displacement(z_mx)
    # dx is odd in x, dy is even in x
    assert torch.allclose(disp_mx[:, 0], -disp[:, 0], atol=1e-5)
    assert torch.allclose(disp_mx[:, 1], disp[:, 1], atol=1e-5)
