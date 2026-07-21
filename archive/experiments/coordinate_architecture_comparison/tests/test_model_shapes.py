"""Each architecture maps a batch of 2D coordinates to 2D coordinates."""

import pytest
import torch

from experiments.coordinate_architecture_comparison.config import ExperimentConfig
from experiments.coordinate_architecture_comparison.models import MODEL_KINDS, build_model


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_forward_preserves_2d_batch_shape(kind):
    config = ExperimentConfig()
    config.model.kind = kind
    model = build_model(config)

    z = torch.randn(17, 2)
    out = model(z)

    assert out.shape == z.shape


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_zero_initialized_models_start_near_identity(kind):
    config = ExperimentConfig()
    config.model.kind = kind
    model = build_model(config)

    z = torch.randn(32, 2)
    residual = (model(z) - z).abs().max().item()

    assert residual < 1e-3
