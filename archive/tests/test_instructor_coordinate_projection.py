from __future__ import annotations

import torch

from benchmark.experiments.instructor_coordinate_projection.model import (
    CoordinateProjectionNet,
    TrapGeometry,
    mean_identity_distance_loss,
)
from benchmark.experiments.instructor_coordinate_projection.diagnostics import map_diagnostics


def test_model_maps_batch_of_2d_inputs_to_2d_outputs():
    model = CoordinateProjectionNet(width=8, depth=2)
    z = torch.randn(16, 2)
    x = model(z)
    assert x.shape == (16, 2)


def test_model_forward_is_residual_around_identity():
    model = CoordinateProjectionNet(width=8, depth=2, zero_last_layer=True)
    z = torch.randn(16, 2)
    assert torch.allclose(model(z), z)


def test_model_supports_backward_pass():
    model = CoordinateProjectionNet(width=8, depth=2)
    z = torch.randn(16, 2)
    loss = mean_identity_distance_loss(model, z)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_map_diagnostics_returns_expected_fields():
    model = CoordinateProjectionNet(width=8, depth=2, zero_last_layer=True)
    diagnostics = map_diagnostics(model, TrapGeometry(), n_points=128, seed=0)
    for key in ("rms_identity_error", "det_jacobian_min", "condition_number_max"):
        assert key in diagnostics
    assert diagnostics["jacobian_determinant_all_positive"] is True


def test_benchmark_wrapper_reports_diagnostics_without_gradient():
    from benchmark.methods.instructor_coordinate_projection import (
        InstructorCoordinateProjectionMethod,
    )
    from benchmark.task import DesignTask

    method = InstructorCoordinateProjectionMethod(training_steps=2, seed=0)
    result = method.evaluate(DesignTask())
    assert result.available
    assert result.gradient is None
    assert "rms_identity_error" in result.extra


def test_existing_benchmark_methods_still_import():
    from benchmark.methods import (
        CoordinateProjectionPINNMethod,
        DevsimFiniteDifferenceMethod,
        InstructorCoordinateProjectionMethod,
        ShapeMethod,
    )

    assert issubclass(CoordinateProjectionPINNMethod, ShapeMethod)
    assert issubclass(DevsimFiniteDifferenceMethod, ShapeMethod)
    assert issubclass(InstructorCoordinateProjectionMethod, ShapeMethod)
