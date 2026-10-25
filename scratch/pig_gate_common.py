import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import CollocationBatch as Batch, sample_collocation_batch as sample_batch
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork


class QuadrupoleOutput(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def even(self, points):
        return 0.25 * (
            self.backend(points)
            + self.backend(points * points.new_tensor([-1.0, 1.0]))
            + self.backend(points * points.new_tensor([1.0, -1.0]))
            + self.backend(-points)
        )

    def forward(self, points):
        return 0.5 * (self.even(points) - self.even(points[:, [1, 0]]))


def uniform_batch(trap, n_interior=1024, n_electrode=64, n_outer=256, interior_seed=0):
    base = sample_batch(trap, np.random.default_rng(interior_seed), n_interior, 1, 1)
    electrode_points, electrode_values = [], []
    theta = 2 * np.pi * (np.arange(n_electrode) + 0.37) / n_electrode
    for electrode in trap.electrodes():
        electrode_points.append(np.column_stack([
            electrode.cx + electrode.radius * np.cos(theta),
            electrode.cy + electrode.radius * np.sin(theta),
        ]))
        electrode_values.append(np.full((n_electrode, 1), trap.electrode_voltage(electrode)))
    outer_theta = 2 * np.pi * (np.arange(n_outer) + 0.19) / n_outer
    outer_points = np.column_stack([trap.outer_radius * np.cos(outer_theta), trap.outer_radius * np.sin(outer_theta)])
    return Batch(
        interior=base.interior,
        electrode_points=torch.tensor(np.vstack(electrode_points), dtype=torch.float32),
        electrode_voltages=torch.tensor(np.vstack(electrode_values), dtype=torch.float32),
        outer_points=torch.tensor(outer_points, dtype=torch.float32),
        outer_voltages=torch.zeros((n_outer, 1), dtype=torch.float32),
    )


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


def exact_field(trap, points):
    sources, boundaries, values = [], [], []
    for electrode in trap.electrodes():
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
        boundaries.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0, 0, 1.3 * trap.outer_radius, 128, 0.011))
    boundaries.append(circle_points(0, 0, trap.outer_radius, 256))
    values.append(np.zeros(256))
    sources = np.vstack(sources)
    boundary = np.vstack(boundaries)
    matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, np.concatenate(values), rcond=1e-12)
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = np.log(np.sqrt(radius2)) @ coefficients[:-1] + coefficients[-1]
    field = -np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, field


def harmonic_coefficients(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    matrix = np.column_stack([np.ones(len(points))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return coefficients[1], coefficients[3]


def q_ratio(points, potential):
    a2, a6 = harmonic_coefficients(points, potential)
    return a6 / a2


def staged_train(trap, probe, seed, adam_steps=500, lbfgs_outer=25, lbfgs_inner=20):
    batch = uniform_batch(trap)
    probe_tensor = torch.tensor(probe, dtype=torch.float32)
    torch.manual_seed(seed)
    backend = PigBackend(outer_radius=trap.outer_radius, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
    field_model = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field_model)
    adam = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
    trajectory = []

    for step in range(adam_steps):
        adam.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        (lpde + 100 * lbc).backward()
        adam.step()
        if step % 100 == 0 or step + 1 == adam_steps:
            with torch.no_grad():
                q = q_ratio(probe, field_model(probe_tensor).numpy().ravel())
            trajectory.append((step, q, "Adam"))
            print(f"seed={seed} Adam step={step:04d} q={q:+.6e}")

    closure_calls = [0]
    lbfgs = torch.optim.LBFGS(backend.parameters(), lr=0.5, max_iter=lbfgs_inner, max_eval=lbfgs_inner * 2, history_size=50, line_search_fn="strong_wolfe")
    for _ in range(lbfgs_outer):
        def closure():
            lbfgs.zero_grad()
            potential, _ = model(batch.interior)
            lpde = pde_loss(potential, batch.interior, model.shape_network)
            lbc = boundary_loss(model, batch)
            loss = lpde + 100 * lbc
            loss.backward()
            closure_calls[0] += 1
            return loss
        lbfgs.step(closure)
        with torch.no_grad():
            q = q_ratio(probe, field_model(probe_tensor).numpy().ravel())
        trajectory.append((adam_steps + closure_calls[0], q, "LBFGS"))
        print(f"seed={seed} LBFGS calls={closure_calls[0]:04d} q={q:+.6e}")
    return model, field_model, trajectory, closure_calls[0]
