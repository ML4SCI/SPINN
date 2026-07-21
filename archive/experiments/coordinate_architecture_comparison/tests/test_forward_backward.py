"""Forward/backward, loss, and metric smoke tests (no full training)."""

import pytest
import torch

from experiments.coordinate_architecture_comparison.config import ExperimentConfig
from experiments.coordinate_architecture_comparison.models import MODEL_KINDS, build_model
from experiments.coordinate_architecture_comparison.training.losses import (
    CoordinateMSELoss,
    MeanEuclideanCoordinateLoss,
)
from experiments.coordinate_architecture_comparison.training.metrics import (
    evaluate_coordinate_error,
)


def _build(kind):
    config = ExperimentConfig()
    config.model.kind = kind
    return build_model(config)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_backward_populates_gradients(kind):
    model = _build(kind)
    z = torch.randn(24, 2)
    loss = MeanEuclideanCoordinateLoss()(model(z), z)
    loss.backward()

    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("loss_fn", [MeanEuclideanCoordinateLoss(), CoordinateMSELoss()])
def test_losses_return_scalar(loss_fn):
    predicted = torch.randn(10, 2)
    target = torch.randn(10, 2)
    value = loss_fn(predicted, target)

    assert value.ndim == 0
    assert torch.isfinite(value)


def test_metrics_run_and_return_floats():
    model = _build("vanilla")
    z = torch.randn(40, 2)
    metrics = evaluate_coordinate_error(model, z)

    assert {"mean_euclidean_error", "mse"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())
