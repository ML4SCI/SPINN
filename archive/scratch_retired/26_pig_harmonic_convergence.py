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


STEPS = 2000
OUTPUT = "scratch/26_pig_harmonic_convergence.png"
HYPOTHESIS = "The wrong harmonic sign at R/r0=1.25 is incomplete equilibration; longer training will cross zero and approach the exact negative value."


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
trap = FourRodTrap(r0=1.0, rod_radius=1.25, outer_radius=5.0, V0=1.0)
batch = sample_batch(trap, np.random.default_rng(0), 1024, 64, 256)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.6
probe = all_points[inside]
reference_potential, reference_field = exact(trap, probe)
q_reference = q_ratio(probe, reference_potential)
p_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))

torch.manual_seed(3)
backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
field_model = QuadrupoleOutput(backend)
model = ComposedField(IdentityShapeNetwork(), field_model)
optimizer = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
probe_tensor_static = torch.tensor(probe, dtype=torch.float32)
history = []
for step in range(STEPS):
    optimizer.zero_grad()
    potential, _ = model(batch.interior)
    lpde = pde_loss(potential, batch.interior, model.shape_network)
    lbc = boundary_loss(model, batch)
    (lpde + 100 * lbc).backward()
    optimizer.step()
    if step % 100 == 0 or step + 1 == STEPS:
        with torch.no_grad():
            prediction = field_model(probe_tensor_static).numpy().ravel()
        q = q_ratio(probe, prediction)
        p_rmse = np.sqrt(np.mean(((prediction - reference_potential) / p_scale) ** 2))
        history.append((step, lpde.detach().item(), lbc.detach().item(), q, p_rmse))
        print(f"step={step:04d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e} q={q:+.6e} V_RMSE={p_rmse:.3e}")

probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
final_potential_tensor = field_model(probe_tensor).squeeze(-1)
final_gradient = torch.autograd.grad(final_potential_tensor.sum(), probe_tensor)[0]
final_potential = final_potential_tensor.detach().numpy()
final_field = -final_gradient.detach().numpy()
p_error = (final_potential - reference_potential) / p_scale
e_error = np.linalg.norm(final_field - reference_field, axis=1) / e_scale
history = np.array(history)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].plot(history[:, 0], history[:, 3], marker="o", label="PIG")
axes[0, 0].axhline(q_reference, color="black", linestyle="--", label="exact")
axes[0, 0].axhline(0, color="grey", linewidth=1)
axes[0, 0].axvline(500, color="red", linestyle=":", label="previous stopping point")
axes[0, 0].set(xlabel="training step", ylabel="signed a6/a2", title="Does the harmonic cross to the physical sign?")
axes[0, 0].legend()

axes[0, 1].semilogy(history[:, 0], history[:, 1], marker="o", label="PDE")
axes[0, 1].semilogy(history[:, 0], history[:, 2], marker="o", label="boundary")
axes[0, 1].semilogy(history[:, 0], history[:, 4], marker="o", label="independent V RMSE")
axes[0, 1].set(xlabel="training step", ylabel="value", title="Loss versus physical field convergence")
axes[0, 1].legend()

pmap = np.full(gx.shape, np.nan)
pmap.ravel()[inside] = p_error
image_p = axes[1, 0].imshow(pmap, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="coolwarm")
axes[1, 0].set(xlabel="x/r0", ylabel="y/r0", title=f"final V error, RMSE {np.sqrt(np.mean(p_error**2)):.2%}")
fig.colorbar(image_p, ax=axes[1, 0], label="normalized potential error")

emap = np.full(gx.shape, np.nan)
emap.ravel()[inside] = e_error
image_e = axes[1, 1].imshow(emap, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="magma")
axes[1, 1].set(xlabel="x/r0", ylabel="y/r0", title=f"final |E error|, RMSE {np.sqrt(np.mean(e_error**2)):.2%}")
fig.colorbar(image_e, ax=axes[1, 1], label="normalized electric-field error")

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: longer PIG equilibration repairs the geometry-sensitive harmonic", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print(f"exact q={q_reference:+.6e} final q={history[-1,3]:+.6e}")
print("CONCLUSION: under-convergence explains the 500-step wrong sign, but training is non-monotonic: the harmonic matches near step 1600 and then overshoots while PDE loss continues falling, so longer training alone is not a safe stopping rule.")
print("saved", OUTPUT)
