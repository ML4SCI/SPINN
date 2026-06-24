from __future__ import annotations

import torch

from experiments.potential_pinn_architectures.config import make_config
from experiments.potential_pinn_architectures.geometry import training_batch
from experiments.potential_pinn_architectures.losses import laplace_residual_loss, total_loss
from experiments.potential_pinn_architectures.models import MODEL_KINDS, build_model
from experiments.potential_pinn_architectures.train import train_one


def small_config(kind: str):
    return make_config(
        kind,
        [
            "training.steps=2",
            "training.n_interior=16",
            "training.n_world=16",
            "training.n_electrode=16",
            "training.eval_every=1",
            "training.print_every=1000",
            "model.pixel_grid_resolution=8",
            "model.pixel_num_grids=2",
            "model.pig_num_gaussians=16",
        ],
    )


def test_models_predict_scalar_potential():
    points = torch.randn(12, 2, requires_grad=True)
    for kind in MODEL_KINDS:
        model = build_model(small_config(kind))
        potential = model(points)
        assert potential.shape == (12, 1)


def test_laplace_loss_is_scalar_and_backward_reaches_parameters():
    points = torch.randn(10, 2, requires_grad=True)
    for kind in MODEL_KINDS:
        model = build_model(small_config(kind))
        loss = laplace_residual_loss(model, points)
        assert loss.ndim == 0
        loss.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_total_loss_backpropagates_for_each_architecture():
    for kind in MODEL_KINDS:
        config = small_config(kind)
        model = build_model(config)
        batch = training_batch(config, __import__("numpy").random.default_rng(0))
        loss, terms = total_loss(model, batch, config.training.bc_weight)
        assert loss.ndim == 0
        assert {"pde", "bc", "total"} == set(terms)
        loss.backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.parameters()
        )


def test_tiny_training_run_without_devsim(tmp_path):
    config = small_config("pixel")
    train_one(config, output_dir=tmp_path, run_devsim_compare=False)
    assert (tmp_path / "training_history.csv").exists()
    assert (tmp_path / "model_state_dict.pt").exists()

