import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import sample_collocation_batch as sample_batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.mlp import MLPBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork


RATIO = 1.00
STEPS = 500
OUTPUT = "scratch/16_mlp_symmetry_gate.png"


class QuadrupoleSymmetry(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def even(self, points):
        x, y = points[:, 0:1], points[:, 1:2]
        return 0.25 * (
            self.base(torch.cat([x, y], 1))
            + self.base(torch.cat([-x, y], 1))
            + self.base(torch.cat([x, -y], 1))
            + self.base(torch.cat([-x, -y], 1))
        )

    def forward(self, points):
        swapped = points[:, [1, 0]]
        return 0.5 * (self.even(points) - self.even(swapped))


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


axis = np.linspace(-0.60, 0.60, 61)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.60
probe = all_points[inside]
z = probe[:, 0] + 1j * probe[:, 1]
orders = [2, 6, 10, 14, 18]
fit_matrix = np.column_stack([np.ones(len(probe))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
trap = FourRodTrap(r0=1.0, rod_radius=RATIO, outer_radius=5.0, V0=1.0)


sources, boundary, values = [], [], []
for electrode in trap.electrodes():
    sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
    boundary.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
    values.append(np.full(72, trap.electrode_voltage(electrode)))
sources.append(circle_points(0, 0, 1.30 * trap.outer_radius, 128, 0.011))
boundary.append(circle_points(0, 0, trap.outer_radius, 256))
values.append(np.zeros(256))
sources = np.vstack(sources)
boundary = np.vstack(boundary)
matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
coefficients, *_ = np.linalg.lstsq(matrix, np.concatenate(values), rcond=1e-12)
reference = kernel(probe, sources) @ coefficients[:-1] + coefficients[-1]
reference_fit, *_ = np.linalg.lstsq(fit_matrix, reference, rcond=None)
q_reference = reference_fit[3] / reference_fit[1]


results = []
for label, symmetric in [("plain MLP", False), ("quadrupole-symmetric MLP", True)]:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    base = MLPBackend(hidden_layers=3, width=32)
    backend = QuadrupoleSymmetry(base) if symmetric else base
    model = ComposedField(IdentityShapeNetwork(), backend)
    optimizer = torch.optim.Adam(backend.parameters(), lr=2.0e-3)
    batch = sample_batch(trap, rng, 2048, 128, 512)
    history = []
    for step in range(STEPS):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        (lpde + 100.0 * lbc).backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == STEPS:
            history.append((step, lpde.detach().item(), lbc.detach().item()))
    with torch.no_grad():
        prediction = backend(torch.tensor(probe, dtype=torch.float32)).numpy().ravel()
    fit, *_ = np.linalg.lstsq(fit_matrix, prediction, rcond=None)
    q = fit[3] / fit[1]
    scale = np.max(np.abs(reference))
    error = np.full(gx.shape, np.nan)
    error.ravel()[inside] = (prediction - reference) / scale
    rmse = np.sqrt(np.mean((prediction - reference) ** 2)) / scale
    results.append((label, np.array(history), q, error, rmse))
    print(f"{label}: q={q:+.6e} reference={q_reference:+.6e} normalized_RMSE={rmse:.4f} PDE={history[-1][1]:.3e} BC={history[-1][2]:.3e}")


fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
labels = ["reference", "plain", "symmetric"]
q_values = [q_reference, results[0][2], results[1][2]]
bars = axes[0, 0].bar(labels, q_values, color=["black", "tab:blue", "tab:orange"])
axes[0, 0].axhline(0.0, color="black", linewidth=1)
axes[0, 0].bar_label(bars, fmt="%.4f")
axes[0, 0].set(ylabel=r"signed $a_6/a_2$", title="Harmonic sign at R/r0=1.00")

for label, history, *_ in results:
    axes[0, 1].semilogy(history[:, 0], history[:, 1], label=f"PDE: {label}")
    axes[0, 1].semilogy(history[:, 0], history[:, 2], linestyle="--", label=f"BC: {label}")
axes[0, 1].set(xlabel="training step", ylabel="loss term", title="Training behavior")
axes[0, 1].legend(fontsize=7)

limit = max(np.nanmax(np.abs(results[0][3])), np.nanmax(np.abs(results[1][3])))
display = np.column_stack([results[0][3], results[1][3]])
image = axes[1, 0].imshow(display, origin="lower", extent=(-1.2, 1.2, -0.6, 0.6), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
axes[1, 0].axvline(0.0, color="black")
axes[1, 0].text(-0.6, 0.56, "plain", ha="center", va="top")
axes[1, 0].text(0.6, 0.56, "symmetric", ha="center", va="top")
axes[1, 0].set(xlabel="two center-disc maps", ylabel=r"$y/r_0$", title="MLP minus exact potential, normalized")
fig.colorbar(image, ax=axes[1, 0])

axes[1, 1].bar(["plain", "symmetric"], [results[0][4], results[1][4]], color=["tab:blue", "tab:orange"])
axes[1, 1].set(ylabel="normalized center-field RMSE", title="Spatial field error")
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Neural-backend gate 2: impose known quadrupole symmetry", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("saved", OUTPUT)
