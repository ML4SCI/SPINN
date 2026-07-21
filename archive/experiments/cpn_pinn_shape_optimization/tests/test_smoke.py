from __future__ import annotations

import pytest
import torch

from experiments.cpn_pinn_shape_optimization.config import make_preset
from experiments.cpn_pinn_shape_optimization.gradient_checks import check_gradient_flow
from experiments.cpn_pinn_shape_optimization.losses import (
    boundary_loss,
    laplace_residual_loss,
    pseudopotential_target_loss,
)
from experiments.cpn_pinn_shape_optimization.models.coupled_model import CoupledCPNPINN
from experiments.cpn_pinn_shape_optimization.models.cpn_wrapper import build_cpn
from experiments.cpn_pinn_shape_optimization.models.pinn_wrapper import build_pinn
from experiments.cpn_pinn_shape_optimization.pseudopotential import (
    derive_k_from_secular_freq,
    psi_ideal,
)
from experiments.cpn_pinn_shape_optimization.train import set_trainable, train
from experiments.cpn_pinn_shape_optimization.config import PhysicsConfig


@pytest.fixture(autouse=True)
def _float32_default():
    """Pin float32 so cross-module test ordering cannot change the default dtype."""

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


def _build_model():
    config = make_preset("C")
    model = CoupledCPNPINN(build_pinn(config.model), build_cpn(config.model)).float()
    return model, config


def test_imports_have_no_training_side_effects():
    model, _ = _build_model()
    assert sum(p.numel() for p in model.parameters()) > 0


def test_cpn_maps_2d_to_2d():
    model, _ = _build_model()
    z = torch.randn(16, 2)
    assert model.cpn(z).shape == (16, 2)


def test_pinn_returns_scalar_potential():
    model, _ = _build_model()
    x = torch.randn(16, 2)
    assert model.pinn(x).shape == (16, 1)


def test_coupled_forward_shapes():
    model, _ = _build_model()
    z = torch.randn(8, 2, requires_grad=True)
    u, x = model(z)
    assert u.shape == (8, 1)
    assert x.shape == (8, 2)


def test_losses_return_finite_scalars():
    model, config = _build_model()
    z = torch.randn(32, 2, requires_grad=True) * 0.2
    u, x = model(z)
    k = derive_k_from_secular_freq(config.physics)
    pde = laplace_residual_loss(u, z, model.map)
    pseudo = pseudopotential_target_loss(u, z, x, model.map, k, config.physics, config.loss.fit_radius)
    for loss in (pde, pseudo):
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_psi_ideal_is_quadratic():
    physics = PhysicsConfig()
    inner = psi_ideal(torch.tensor([[0.1, 0.0]]), 1.0, physics)
    outer = psi_ideal(torch.tensor([[0.2, 0.0]]), 1.0, physics)
    assert torch.isclose(outer / inner, torch.tensor(4.0), atol=1e-4)


def test_gradient_flow_reaches_both_groups():
    model, config = _build_model()
    z = torch.randn(64, 2, requires_grad=True) * 0.2
    u, x = model(z)
    k = derive_k_from_secular_freq(config.physics)
    loss = laplace_residual_loss(u, z, model.map) + pseudopotential_target_loss(
        u, z, x, model.map, k, config.physics, config.loss.fit_radius
    )
    flow = check_gradient_flow(model, loss)
    assert flow["pinn_grads_finite_nonzero"]
    assert flow["cpn_grads_finite_nonzero"]


def test_set_trainable_flags_per_variation():
    model, _ = _build_model()
    config_a = make_preset("A")
    set_trainable(model, config_a)
    assert all(not p.requires_grad for p in model.pinn_parameters())
    assert all(p.requires_grad for p in model.cpn_parameters())

    config_c = make_preset("C")
    set_trainable(model, config_c)
    assert all(p.requires_grad for p in model.pinn_parameters())
    assert all(p.requires_grad for p in model.cpn_parameters())


def test_one_training_step_runs():
    config = make_preset("C")
    config.training.steps = 2
    config.training.n_interior = 64
    config.training.n_world = 32
    config.training.n_electrode = 32
    config.training.eval_every = 1
    config.training.print_every = 100
    result = train(config)
    assert result["gradient_flow"]["cpn_grads_finite_nonzero"]
