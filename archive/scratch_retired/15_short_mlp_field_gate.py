import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.collocation import sample_collocation_batch as sample_batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.mlp import MLPBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork


RATIOS = [1.00, 1.145, 1.25]
TRAIN_STEPS = 500
INTERIOR = 2048
ELECTRODE_POINTS = 128
OUTER_POINTS = 512
OUTPUT = "scratch/15_short_mlp_field_gate.png"


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


axis = np.linspace(-0.60, 0.60, 61)
gx, gy = np.meshgrid(axis, axis)
probe_all = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(probe_all[:, 0], probe_all[:, 1]) <= 0.60
probe = probe_all[inside]
z = probe[:, 0] + 1j * probe[:, 1]
orders = [2, 6, 10, 14, 18]
fit_matrix = np.column_stack([np.ones(len(probe))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])


def exact_field(trap):
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
    return kernel(probe, sources) @ coefficients[:-1] + coefficients[-1]


def q_from_field(values):
    fit, *_ = np.linalg.lstsq(fit_matrix, values, rcond=None)
    return fit[3] / fit[1]


rows = []
histories = []
error_maps = []
for ratio in RATIOS:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    trap = FourRodTrap(r0=1.0, rod_radius=ratio, outer_radius=5.0, V0=1.0)
    backend = MLPBackend(hidden_layers=3, width=32)
    model = ComposedField(IdentityShapeNetwork(), backend)
    optimizer = torch.optim.Adam(backend.parameters(), lr=2.0e-3)
    batch = sample_batch(trap, rng, INTERIOR, ELECTRODE_POINTS, OUTER_POINTS)
    history = []
    for step in range(TRAIN_STEPS):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        loss = lpde + 100.0 * lbc
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step + 1 == TRAIN_STEPS:
            history.append((step, lpde.detach().item(), lbc.detach().item()))
            print(f"R/r0={ratio:.3f} step={step:03d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e}")
    with torch.no_grad():
        prediction = backend(torch.tensor(probe, dtype=torch.float32)).numpy().ravel()
    reference = exact_field(trap)
    q_mlp = q_from_field(prediction)
    q_reference = q_from_field(reference)
    scale = np.max(np.abs(reference))
    error = np.full(gx.shape, np.nan)
    error.ravel()[inside] = (prediction - reference) / scale
    error_maps.append(error)
    rows.append((ratio, q_mlp, q_reference, np.sqrt(np.mean((prediction - reference) ** 2)) / scale))
    histories.append(np.array(history))
    print(f"R/r0={ratio:.3f} q_mlp={q_mlp:+.5e} q_reference={q_reference:+.5e} normalized_RMSE={rows[-1][-1]:.3e}")


rows = np.array(rows)
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
axes[0, 0].plot(rows[:, 0], rows[:, 2], marker="o", label="exact-boundary reference")
axes[0, 0].plot(rows[:, 0], rows[:, 1], marker="s", label="500-step MLP")
axes[0, 0].axhline(0.0, color="black", linewidth=1)
axes[0, 0].axvline(1.145, color="red", linestyle="--")
axes[0, 0].set(xlabel=r"electrode radius $R/r_0$", ylabel=r"signed $a_6/a_2$", title="Does the neural solve recover the harmonic trend?")
axes[0, 0].legend()

for ratio, history in zip(RATIOS, histories):
    axes[0, 1].semilogy(history[:, 0], history[:, 1], label=f"PDE R={ratio}")
    axes[0, 1].semilogy(history[:, 0], history[:, 2], linestyle="--", label=f"BC R={ratio}")
axes[0, 1].set(xlabel="training step", ylabel="loss term", title="Short-solve training behavior")
axes[0, 1].legend(fontsize=7, ncol=2)

limit = max(np.nanmax(np.abs(item)) for item in error_maps)
display = np.column_stack(error_maps)
image = axes[1, 0].imshow(display, origin="lower", extent=(-1.8, 1.8, -0.6, 0.6), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
for x in (-0.6, 0.6):
    axes[1, 0].axvline(x, color="black", linewidth=1)
for index, ratio in enumerate(RATIOS):
    axes[1, 0].text(-1.2 + 1.2 * index, 0.56, f"R={ratio}", ha="center", va="top")
axes[1, 0].set(xlabel="three center-disc error maps", ylabel=r"$y/r_0$", title="MLP minus exact potential, normalized")
fig.colorbar(image, ax=axes[1, 0])

axes[1, 1].plot(rows[:, 0], rows[:, 3], marker="o")
axes[1, 1].set(xlabel=r"electrode radius $R/r_0$", ylabel="normalized center-field RMSE", title="Spatial error gate")
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Neural-backend gate 1: short fixed-geometry MLP solves", fontsize=15)
fig.savefig(OUTPUT, dpi=180)
print("saved", OUTPUT)
