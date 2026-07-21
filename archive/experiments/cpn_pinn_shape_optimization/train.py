"""Single training loop, switchable across the five variations.

The same loop runs every variation; ``maybe_pretrain`` handles the optional
PINN-first and CPN-identity warm-up stages, and ``set_trainable`` decides which
parameter groups the main optimizer updates. The loop is deliberately lightly
constrained: pathological maps are expected, so the only safeguards are an
identity warm start, gradient clipping, and skipping any step whose loss is not
finite. The minimum Jacobian determinant is logged each evaluation so map
degeneracy is observed rather than hidden.
"""

from __future__ import annotations

import numpy as np
import torch

from PINN.Coordinate_Projection.operators import map_jacobian

from .config import ExperimentConfig
from .geometry import (
    build_geometry,
    build_sampler,
    electrode_boundary_points_and_voltages,
    world_boundary_points_and_voltages,
)
from .gradient_checks import check_gradient_flow
from .losses import (
    boundary_loss,
    deformation_loss,
    jacobian_regularizer,
    laplace_residual_loss,
    pseudopotential_target_loss,
    smoothness_loss,
)
from .models.coupled_model import CoupledCPNPINN
from .models.cpn_wrapper import build_cpn
from .models.pinn_wrapper import build_pinn
from .pseudopotential import derive_k_from_secular_freq

from benchmark.experiments.instructor_coordinate_projection.model import (
    TrainingConfig as IdentityTrainingConfig,
    train_identity_map,
)


def build_components(config: ExperimentConfig) -> CoupledCPNPINN:
    pinn = build_pinn(config.model)
    cpn = build_cpn(config.model)
    # Sampled batches are float32; pin the parameters to match regardless of the
    # process-wide default dtype (some test modules set it to float64).
    return CoupledCPNPINN(pinn, cpn).float()


def _tensor(array: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, requires_grad=requires_grad)


def _sample_reference_tensors(config: ExperimentConfig, sampler, rng):
    geometry = build_geometry(config.geometry)
    interior = _tensor(sampler.interior(config.training.n_interior), requires_grad=True)
    world_points, world_values = world_boundary_points_and_voltages(
        geometry, config.training.n_world, rng
    )
    electrode_points, electrode_values = electrode_boundary_points_and_voltages(
        geometry, config.physics, config.training.n_electrode // 4, rng
    )
    return (
        interior,
        _tensor(world_points),
        _tensor(world_values),
        _tensor(electrode_points),
        _tensor(electrode_values),
    )


def _pretrain_pinn_identity(model: CoupledCPNPINN, config: ExperimentConfig, rng) -> None:
    """Train the PINN on the true Cartesian domain (CPN held at identity).

    Used by the PINN-first variations so the field network already solves the
    boundary-value problem before any deformation is introduced.
    """

    for parameter in model.cpn_parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.pinn_parameters(), lr=config.training.lr)
    sampler = build_sampler(build_geometry(config.geometry), config.seed + 1)
    for _ in range(config.training.pinn_pretrain_steps):
        interior, world_p, world_v, el_p, el_v = _sample_reference_tensors(config, sampler, rng)
        optimizer.zero_grad()
        potential, _ = model(interior)
        loss = config.loss.lambda_pde * laplace_residual_loss(potential, interior, model.map)
        loss = loss + config.loss.lambda_bc * boundary_loss(
            model, world_p, world_v, el_p, el_v, voltage_scale=config.physics.rf_voltage
        )
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.pinn_parameters(), config.training.grad_clip)
            optimizer.step()
    for parameter in model.cpn_parameters():
        parameter.requires_grad_(True)


def maybe_pretrain(model: CoupledCPNPINN, config: ExperimentConfig, rng) -> None:
    flags = config.variation_flags
    if flags is None:
        return
    if flags.cpn_pretrained_identity and not config.model.cpn_zero_last_layer:
        geometry = build_geometry(config.geometry)
        identity_config = IdentityTrainingConfig(steps=config.training.cpn_pretrain_steps)
        train_identity_map(model.cpn, geometry, identity_config, seed=config.seed)
    if flags.pinn_pretrained:
        _pretrain_pinn_identity(model, config, rng)


def set_trainable(model: CoupledCPNPINN, config: ExperimentConfig) -> torch.optim.Optimizer:
    flags = config.variation_flags
    pinn_trainable = not (flags is not None and flags.pinn_frozen)
    cpn_trainable = not (flags is not None and flags.cpn_frozen)
    for parameter in model.pinn_parameters():
        parameter.requires_grad_(pinn_trainable)
    for parameter in model.cpn_parameters():
        parameter.requires_grad_(cpn_trainable)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=config.training.lr)


def _total_loss(model, config, batch, k):
    interior, world_p, world_v, el_p, el_v = batch
    potential, physical = model(interior)
    loss_pde = laplace_residual_loss(potential, interior, model.map)
    loss_pseudo = pseudopotential_target_loss(
        potential, interior, physical, model.map, k, config.physics, config.loss.fit_radius
    )
    loss_bc = boundary_loss(model, world_p, world_v, el_p, el_v, voltage_scale=config.physics.rf_voltage)
    total = (
        config.loss.lambda_pde * loss_pde
        + config.loss.lambda_pseudo * loss_pseudo
        + config.loss.lambda_bc * loss_bc
    )
    if config.loss.lambda_deform > 0:
        total = total + config.loss.lambda_deform * deformation_loss(model.cpn, interior)
    if config.loss.lambda_smooth > 0:
        total = total + config.loss.lambda_smooth * smoothness_loss(model.cpn, interior)
    if config.loss.lambda_jac > 0:
        total = total + config.loss.lambda_jac * jacobian_regularizer(model.map, interior)
    terms = {
        "total": float(total.detach()),
        "pde": float(loss_pde.detach()),
        "pseudo": float(loss_pseudo.detach()),
        "bc": float(loss_bc.detach()),
    }
    return total, terms


def _min_jacobian_determinant(model: CoupledCPNPINN, sampler) -> float:
    points = _tensor(sampler.interior(1024), requires_grad=True)
    _, jacobian = map_jacobian(model.map, points)
    return float(torch.linalg.det(jacobian).min().detach())


def train(config: ExperimentConfig) -> dict:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    model = build_components(config)
    maybe_pretrain(model, config, rng)
    optimizer = set_trainable(model, config)

    k = derive_k_from_secular_freq(config.physics)
    geometry = build_geometry(config.geometry)
    sampler = build_sampler(geometry, config.seed)

    history: list[dict] = []
    skipped_steps = 0
    for step in range(config.training.steps):
        batch = _sample_reference_tensors(config, sampler, rng)
        optimizer.zero_grad()
        total, terms = _total_loss(model, config, batch, k)
        if not np.isfinite(terms["total"]):
            skipped_steps += 1
            continue
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.training.grad_clip
        )
        optimizer.step()

        if step % config.training.eval_every == 0 or step + 1 == config.training.steps:
            terms["step"] = step
            terms["det_jacobian_min"] = _min_jacobian_determinant(model, sampler)
            history.append(terms)
        if step % config.training.print_every == 0:
            print(
                f"{step:6d} total={terms['total']:.3e} pde={terms['pde']:.3e} "
                f"pseudo={terms['pseudo']:.3e} bc={terms['bc']:.3e}",
                flush=True,
            )

    diagnostic_batch = _sample_reference_tensors(config, sampler, rng)
    diagnostic_loss, _ = _total_loss(model, config, diagnostic_batch, k)
    gradient_flow = check_gradient_flow(model, diagnostic_loss)

    return {
        "model": model,
        "k_quadratic_eV_per_m2": k,
        "history": history,
        "skipped_steps": skipped_steps,
        "gradient_flow": gradient_flow,
    }
