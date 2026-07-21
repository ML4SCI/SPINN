from __future__ import annotations

import numpy as np
import pytest
import torch

from PINN.Coordinate_Projection.constraints import anchor_loss, jacobian_constraints
from PINN.Coordinate_Projection.maps import (
    AffineMap,
    ElectrodeTranslationMap,
    IdentityMap,
    RadialDeformationMap,
    ResidualCoordinateMap,
)
from PINN.Coordinate_Projection.model import (
    CoordinateProjectionPINN,
    ParametricPotentialMLP,
    PhysicalPotentialMLP,
)
from PINN.Coordinate_Projection.operators import (
    cartesian_laplacian,
    physical_hessian,
    pullback_laplacian,
)
from PINN.Coordinate_Projection.train import (
    DEFAULT_CONFIG,
    inside_reference_electrodes,
    sample_electrode_boundaries,
    sample_world_boundary,
)
from PINN.Coordinate_Projection.validate_gradient import finite_difference_derivatives


torch.set_default_dtype(torch.float64)


def test_identity_pullback_matches_cartesian_laplacian():
    points = torch.tensor([[0.2, -0.3], [-0.4, 0.1]], requires_grad=True)
    potential = points[:, :1] ** 2 + points[:, 1:] ** 2
    residual = pullback_laplacian(potential, points, IdentityMap())
    assert torch.allclose(residual, torch.full_like(residual, 4.0), atol=1e-10)


@pytest.mark.parametrize(
    "coordinate_map",
    [
        AffineMap(torch.tensor([[1.2, 0.15], [0.0, 0.8]])),
        RadialDeformationMap(amplitude=0.06, length_scale=0.8),
    ],
)
def test_manufactured_harmonic_solution_survives_coordinate_map(coordinate_map):
    points = torch.tensor(
        [[0.12, 0.08], [-0.25, 0.18], [0.33, -0.21]],
        requires_grad=True,
    )
    physical = coordinate_map(points)
    potential = physical[:, :1] ** 2 - physical[:, 1:] ** 2
    residual = pullback_laplacian(potential, points, coordinate_map)
    assert torch.max(torch.abs(residual)).item() < 2e-8


def test_physical_hessian_recovers_manufactured_quadratic():
    coordinate_map = AffineMap(torch.tensor([[1.1, 0.2], [-0.1, 0.9]]))
    points = torch.tensor([[0.1, -0.2]], requires_grad=True)
    physical = coordinate_map(points)
    potential = physical[:, :1] ** 2 - physical[:, 1:] ** 2
    hessian = physical_hessian(potential, points, coordinate_map)[0]
    expected = torch.tensor([[2.0, 0.0], [0.0, -2.0]])
    assert torch.allclose(hessian, expected, atol=1e-9)


def test_electrode_translation_is_exact_near_electrodes_and_anchored_far_away():
    coordinate_map = ElectrodeTranslationMap()
    distance = torch.tensor(0.14)
    east_boundary = torch.tensor([[0.175, 0.0], [0.125, 0.05]])
    mapped = coordinate_map(east_boundary, distance)
    assert torch.allclose(mapped - east_boundary, torch.tensor([[0.015, 0.0], [0.015, 0.0]]))

    world = torch.tensor([[1.25, 0.2], [-0.8, -1.0]])
    assert torch.allclose(coordinate_map(world, distance), world)
    assert anchor_loss(coordinate_map, world, distance).item() == 0.0


def test_residual_coordinate_map_starts_as_identity():
    coordinate_map = ResidualCoordinateMap()
    points = torch.randn(20, 2)
    assert torch.allclose(coordinate_map(points), points)
    assert torch.count_nonzero(coordinate_map.residual(points)) == 0


def test_residual_coordinate_map_matches_equation_13_after_deformation():
    coordinate_map = ResidualCoordinateMap(width=8, depth=2)
    output = coordinate_map.network[-1]
    with torch.no_grad():
        output.bias.copy_(torch.tensor([0.1, -0.2]))
    points = torch.randn(10, 2)
    expected_residual = torch.tensor([0.1, -0.2]).expand_as(points)
    assert torch.allclose(coordinate_map(points), points + expected_residual)
    coordinate_map.reset_to_identity()
    assert torch.allclose(coordinate_map(points), points)


def test_paper_residual_map_uses_six_hidden_layers_of_width_50_by_default():
    coordinate_map = ResidualCoordinateMap()
    hidden_layers = [
        layer
        for layer in coordinate_map.network
        if isinstance(layer, torch.nn.Linear)
    ][:-1]
    assert len(hidden_layers) == 6
    assert all(layer.out_features == 50 for layer in hidden_layers)


def test_paper_shape_and_field_networks_compose_on_projected_coordinates():
    shape = ResidualCoordinateMap(width=8, depth=2)
    field = PhysicalPotentialMLP(width=8, depth=2)
    model = CoordinateProjectionPINN(shape, field)
    reference = torch.randn(12, 2, requires_grad=True)
    physical, potential = model(reference)
    assert torch.allclose(physical, reference)
    assert potential.shape == (12, 1)
    residual = cartesian_laplacian(potential, physical)
    assert residual.shape == (12, 1)


@pytest.mark.parametrize("distance_value", [0.11, 0.145])
def test_translation_map_stays_orientation_preserving_in_configured_range(distance_value):
    coordinate_map = ElectrodeTranslationMap()
    axis = torch.linspace(-0.3, 0.3, 81)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1).requires_grad_(True)
    distance = torch.full((len(points), 1), distance_value)
    diagnostics = jacobian_constraints(coordinate_map, points, distance)
    assert torch.min(diagnostics["determinant"]).item() > 0.15
    assert torch.max(diagnostics["condition"]).item() < 6.0


def test_boundary_samplers_respect_reference_geometry():
    geometry = DEFAULT_CONFIG["geometry"]
    world = sample_world_boundary(
        100,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    half_x = geometry["world_size_x"] / 2.0
    half_y = geometry["world_size_y"] / 2.0
    on_boundary = (
        torch.isclose(torch.abs(world[:, 0]), torch.tensor(half_x))
        | torch.isclose(torch.abs(world[:, 1]), torch.tensor(half_y))
    )
    assert torch.all(on_boundary)

    electrode, values = sample_electrode_boundaries(
        20,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert torch.all(inside_reference_electrodes(electrode, geometry))
    assert set(values.flatten().tolist()) == {0.0, 1.0}


def test_relu_is_rejected_for_strong_form_laplacian():
    with pytest.raises(ValueError, match="second-order"):
        ParametricPotentialMLP(activation="relu")


def test_five_point_geometry_derivative_is_high_order():
    epsilon = 0.1
    offsets = [-2.0, -1.0, 1.0, 2.0]
    values = [(1.0 + offset * epsilon) ** 5 for offset in offsets]
    three_point, five_point = finite_difference_derivatives(values, epsilon)
    assert abs(five_point - 5.0) < abs(three_point - 5.0)
    assert np.isclose(five_point, 5.0 - 4.0 * epsilon**4)
