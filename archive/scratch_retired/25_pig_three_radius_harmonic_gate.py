import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import sample_collocation_batch as sample_batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork


RATIOS = [1.00, 1.145, 1.25]
STEPS = 500
OUTPUT = "scratch/25_pig_three_radius_harmonic_gate.png"
HYPOTHESIS = "Output-symmetric PIG will reproduce the positive-zero-negative a6/a2 trend across the known circular-electrode optimum."


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


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


def exact(trap, points):
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


def q_ratio(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    matrix = np.column_stack([np.ones(len(points))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return coefficients[3] / coefficients[1]


print("HYPOTHESIS:", HYPOTHESIS)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.6
probe = all_points[inside]
results = []
for ratio in RATIOS:
    trap = FourRodTrap(r0=1.0, rod_radius=ratio, outer_radius=5.0, V0=1.0)
    batch = sample_batch(trap, np.random.default_rng(0), 1024, 64, 256)
    reference_potential, reference_field = exact(trap, probe)
    torch.manual_seed(3)
    backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
    field_model = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field_model)
    optimizer = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
    history = []
    for step in range(STEPS):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        (lpde + 100 * lbc).backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == STEPS:
            history.append((step, lpde.detach().item(), lbc.detach().item()))
        if step % 100 == 0 or step + 1 == STEPS:
            print(f"R={ratio:.3f} step={step:03d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e}")
    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    potential_tensor = field_model(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(potential_tensor.sum(), probe_tensor)[0]
    prediction = potential_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    p_scale = np.max(np.abs(reference_potential))
    e_scale = np.max(np.linalg.norm(reference_field, axis=1))
    results.append(
        {
            "ratio": ratio,
            "history": np.array(history),
            "q_ref": q_ratio(probe, reference_potential),
            "q_pig": q_ratio(probe, prediction),
            "p_error": (prediction - reference_potential) / p_scale,
            "e_error": np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale,
        }
    )

fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
ratio = np.array([row["ratio"] for row in results])
axes[0, 0].plot(ratio, [row["q_ref"] for row in results], marker="o", label="exact boundary")
axes[0, 0].plot(ratio, [row["q_pig"] for row in results], marker="s", label="output-symmetric PIG")
axes[0, 0].axhline(0, color="black", linewidth=1)
axes[0, 0].axvline(1.145, color="red", linestyle="--")
axes[0, 0].set(xlabel="R/r0", ylabel="signed a6/a2", title="Geometry-dependent harmonic trend")
axes[0, 0].legend()

for row in results:
    history = row["history"]
    axes[0, 1].semilogy(history[:, 0], history[:, 1], label=f"R={row['ratio']}: PDE")
    axes[0, 2].semilogy(history[:, 0], history[:, 2], label=f"R={row['ratio']}: BC")
axes[0, 1].set(xlabel="step", ylabel="PDE loss", title="Physics convergence")
axes[0, 2].set(xlabel="step", ylabel="boundary loss", title="Boundary convergence")
axes[0, 1].legend(fontsize=8)
axes[0, 2].legend(fontsize=8)

limit = max(np.max(np.abs(row["p_error"])) for row in results)
for column, row in enumerate(results):
    error_map = np.full(gx.shape, np.nan)
    error_map.ravel()[inside] = row["p_error"]
    image = axes[1, column].imshow(error_map, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="coolwarm", vmin=-limit, vmax=limit)
    p_rmse = np.sqrt(np.mean(row["p_error"]**2))
    e_rmse = np.sqrt(np.mean(row["e_error"]**2))
    axes[1, column].set(xlabel="x/r0", ylabel="y/r0", title=f"R={row['ratio']}: V {p_rmse:.2%}, E {e_rmse:.2%}")

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.colorbar(image, ax=axes[1, :], label="normalized potential error")
fig.suptitle("Hypothesis: one symmetric PIG architecture follows electrode-geometry physics", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

for row in results:
    print(f"R={row['ratio']:.3f} q_ref={row['q_ref']:+.6e} q_pig={row['q_pig']:+.6e} V_RMSE={np.sqrt(np.mean(row['p_error']**2)):.3e} E_RMSE={np.sqrt(np.mean(row['e_error']**2)):.3e}")
print("CONCLUSION: fail the three-radius gate; PIG is accurate at R=1.00 but remains positively biased near and above the optimum, including the wrong sign at R=1.25 despite sub-percent potential RMSE.")
print("saved", OUTPUT)
