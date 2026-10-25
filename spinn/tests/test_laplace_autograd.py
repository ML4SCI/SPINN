"""The pullback Laplacian reduces to the Cartesian one under the identity map,
and both reproduce a known analytic harmonic field's Laplacian (= 0)."""

import torch

from spinn.models.composed import IdentityShapeNetwork
from spinn.physics.laplace import cartesian_laplacian, pullback_laplacian


def _harmonic(x):
    # V = x^2 - y^2 is harmonic: Delta V = 0
    return x[:, 0:1] ** 2 - x[:, 1:2] ** 2


def test_cartesian_laplacian_of_harmonic_is_zero():
    x = torch.randn(64, 2, requires_grad=True)
    v = _harmonic(x)
    lap = cartesian_laplacian(v, x)
    assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-4)


def test_pullback_equals_cartesian_under_identity():
    identity = IdentityShapeNetwork()
    z = torch.randn(64, 2, requires_grad=True)
    v = _harmonic(z)
    pull = pullback_laplacian(v, z, identity)
    assert torch.allclose(pull, torch.zeros_like(pull), atol=1e-4)


def test_pullback_nonzero_for_nonharmonic():
    identity = IdentityShapeNetwork()
    z = torch.randn(64, 2, requires_grad=True)
    v = z[:, 0:1] ** 2 + z[:, 1:2] ** 2  # Delta = 4
    pull = pullback_laplacian(v, z, identity)
    assert torch.allclose(pull, 4.0 * torch.ones_like(pull), atol=1e-3)
