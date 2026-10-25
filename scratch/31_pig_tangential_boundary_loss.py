import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import CollocationBatch as Batch, sample_collocation_batch as sample_batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork


STEPS = 2000
ALPHA = 0.10
EXACT_Q = -6.409705e-3
OUTPUT = "scratch/31_pig_tangential_boundary_loss.png"
HYPOTHESIS = "A reference-free tangential boundary derivative penalty suppresses the structured angular voltage error and stabilizes a6/a2 better than voltage MSE alone."
CONTROL = np.array([
    [0, -1.370893e-2], [100, -3.930599e-2], [200, -4.383397e-2], [300, -2.504599e-2],
    [400, -8.021577e-3], [500, 4.626002e-3], [600, 1.160969e-2], [700, 1.296215e-2],
    [800, 1.027798e-2], [900, 6.095132e-3], [1000, 3.062510e-3], [1100, 1.059615e-3],
    [1200, -1.255300e-3], [1300, -3.904584e-3], [1400, -6.574689e-3], [1500, -8.854717e-3],
    [1600, -1.036529e-2], [1700, -1.097603e-2], [1800, -1.081941e-2], [1900, -1.012365e-2],
    [1999, -9.096535e-3],
])


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


def uniform_batch(trap, n_electrode=64, n_outer=256):
    base = sample_batch(trap, np.random.default_rng(0), 1024, 1, 1)
    electrode_points, electrode_values, electrode_tangents = [], [], []
    theta = 2 * np.pi * (np.arange(n_electrode) + 0.37) / n_electrode
    tangent = np.column_stack([-np.sin(theta), np.cos(theta)])
    for electrode in trap.electrodes():
        electrode_points.append(np.column_stack([
            electrode.cx + electrode.radius * np.cos(theta),
            electrode.cy + electrode.radius * np.sin(theta),
        ]))
        electrode_values.append(np.full((n_electrode, 1), trap.electrode_voltage(electrode)))
        electrode_tangents.append(tangent)
    outer_theta = 2 * np.pi * (np.arange(n_outer) + 0.19) / n_outer
    outer_points = np.column_stack([trap.outer_radius * np.cos(outer_theta), trap.outer_radius * np.sin(outer_theta)])
    batch = Batch(
        interior=base.interior,
        electrode_points=torch.tensor(np.vstack(electrode_points), dtype=torch.float32, requires_grad=True),
        electrode_voltages=torch.tensor(np.vstack(electrode_values), dtype=torch.float32),
        outer_points=torch.tensor(outer_points, dtype=torch.float32, requires_grad=True),
        outer_voltages=torch.zeros((n_outer, 1), dtype=torch.float32),
    )
    tangents = torch.tensor(np.vstack(electrode_tangents), dtype=torch.float32)
    outer_tangents = torch.tensor(np.column_stack([-np.sin(outer_theta), np.cos(outer_theta)]), dtype=torch.float32)
    return batch, tangents, outer_tangents


def tangential_loss(model, batch, electrode_tangents, outer_tangents, rod_radius, outer_radius):
    electrode_values, _ = model(batch.electrode_points)
    electrode_gradient = torch.autograd.grad(electrode_values.sum(), batch.electrode_points, create_graph=True)[0]
    electrode_ds = torch.sum(electrode_gradient * electrode_tangents, dim=1)
    outer_values, _ = model(batch.outer_points)
    outer_gradient = torch.autograd.grad(outer_values.sum(), batch.outer_points, create_graph=True)[0]
    outer_ds = torch.sum(outer_gradient * outer_tangents, dim=1)
    return rod_radius**2 * torch.mean(electrode_ds**2) + outer_radius**2 * torch.mean(outer_ds**2)


def harmonic_coefficients(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    matrix = np.column_stack([np.ones(len(points))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return coefficients[1], coefficients[3]


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.25, outer_radius=5.0, V0=1.0)
batch, electrode_tangents, outer_tangents = uniform_batch(trap)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
probe = probe[np.hypot(probe[:, 0], probe[:, 1]) <= 0.6]
probe_tensor = torch.tensor(probe, dtype=torch.float32)
east = trap.electrodes()[0]
dense_theta = np.linspace(-np.pi, np.pi, 721)
dense_east = np.column_stack([east.cx + east.radius * np.cos(dense_theta), east.cy + east.radius * np.sin(dense_theta)])
dense_east_tensor = torch.tensor(dense_east, dtype=torch.float32, requires_grad=True)

torch.manual_seed(3)
backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
field_model = QuadrupoleOutput(backend)
model = ComposedField(IdentityShapeNetwork(), field_model)
optimizer = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
records = []
best = None

for step in range(STEPS):
    optimizer.zero_grad()
    potential, _ = model(batch.interior)
    lpde = pde_loss(potential, batch.interior, model.shape_network)
    lbc = boundary_loss(model, batch)
    ltan = tangential_loss(model, batch, electrode_tangents, outer_tangents, trap.rod_radius_abs, trap.outer_radius)
    (lpde + 100 * (lbc + ALPHA * ltan)).backward()
    optimizer.step()
    if step % 100 == 0 or step + 1 == STEPS:
        with torch.no_grad():
            center_prediction = field_model(probe_tensor).numpy().ravel()
        dense_values = field_model(dense_east_tensor)
        dense_gradient = torch.autograd.grad(dense_values.sum(), dense_east_tensor)[0]
        dense_tangent = torch.tensor(np.column_stack([-np.sin(dense_theta), np.cos(dense_theta)]), dtype=torch.float32)
        dense_ds = torch.sum(dense_gradient * dense_tangent, dim=1).detach().numpy()
        dense_error = dense_values.detach().numpy().ravel() - trap.electrode_voltage(east)
        a2, a6 = harmonic_coefficients(probe, center_prediction)
        q = a6 / a2
        row = (step, lpde.item(), lbc.item(), ltan.item(), q, abs(q - EXACT_Q), np.sqrt(np.mean(dense_error**2)), np.sqrt(np.mean((east.radius * dense_ds)**2)))
        records.append(row)
        if best is None or row[5] < best[0]:
            best = (row[5], step, dense_error.copy())
        print(f"step={step:04d} q={q:+.6e} |dq|={row[5]:.3e} PDE={row[1]:.3e} BC={row[2]:.3e} tan={row[3]:.3e} denseV={row[6]:.3e} denseTan={row[7]:.3e}")

records = np.array(records)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].plot(CONTROL[:, 0], CONTROL[:, 1], marker="o", markersize=3, label="voltage MSE only")
axes[0, 0].plot(records[:, 0], records[:, 4], marker="o", markersize=3, label=f"+ tangential, alpha={ALPHA}")
axes[0, 0].axhline(EXACT_Q, color="black", linestyle="--", label="exact")
axes[0, 0].axhline(0, color="grey", linewidth=1)
axes[0, 0].set(xlabel="step", ylabel="signed a6/a2", title="Physical harmonic trajectory")
axes[0, 0].legend()

axes[0, 1].semilogy(records[:, 0], records[:, 1], marker="o", label="PDE")
axes[0, 1].semilogy(records[:, 0], records[:, 2], marker="o", label="voltage boundary")
axes[0, 1].semilogy(records[:, 0], ALPHA * records[:, 3], marker="o", label="weighted tangential boundary")
axes[0, 1].set(xlabel="step", ylabel="loss term", title="What the optimizer is balancing")
axes[0, 1].legend()

axes[1, 0].semilogy(records[:, 0], records[:, 6], marker="o", label="dense voltage RMSE")
axes[1, 0].semilogy(records[:, 0], records[:, 7], marker="o", label="dense dimensionless tangent RMSE")
axes[1, 0].set(xlabel="step", ylabel="dense east-electrode error", title="Does the structured surface mode shrink?")
axes[1, 0].legend()

axes[1, 1].plot(np.degrees(dense_theta), dense_error, label="final")
axes[1, 1].plot(np.degrees(dense_theta), best[2], linestyle="--", label=f"best-q step {best[1]}")
axes[1, 1].axhline(0, color="black", linewidth=1)
axes[1, 1].set(xlabel="angle around east electrode (degrees)", ylabel="potential error / V0", title="Remaining angular boundary error")
axes[1, 1].legend()

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: surface-gradient matching stabilizes the electrode harmonic", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

control_final = abs(CONTROL[-1, 1] - EXACT_Q)
control_best = np.min(np.abs(CONTROL[:, 1] - EXACT_Q))
print(f"SUMMARY control final |dq|={control_final:.3e} best |dq|={control_best:.3e}")
print(f"SUMMARY tangential final q={records[-1,4]:+.6e} final |dq|={records[-1,5]:.3e}; best step={best[1]} best |dq|={best[0]:.3e}")
print("CONCLUSION: hypothesis rejected as a stabilizer. Tangential matching improves best attainable harmonic error from 1.65e-4 to 4.17e-5 and cuts final dense voltage RMSE to 5.01e-4, but it still overshoots and worsens final harmonic error from 2.69e-3 to 4.97e-3. The redundant derivative condition reconditions the path without providing a safe endpoint.")
print("saved", OUTPUT)
