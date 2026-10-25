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
EXACT_Q = -6.409705e-3
OUTPUT = "scratch/30_pig_boundary_quadrature.png"
HYPOTHESIS = "Uniform angular quadrature resolves the smooth electrode-error mode better than 64 random boundary points, and increasing uniform density from 64 to 256 stabilizes a6/a2."
CONTROL = np.array([
    [0, -1.306740e-2], [100, -4.051578e-2], [200, -4.675744e-2], [300, -2.928343e-2],
    [400, -1.246286e-2], [500, 1.211572e-3], [600, 1.015245e-2], [700, 1.354622e-2],
    [800, 1.258343e-2], [900, 9.628328e-3], [1000, 6.311906e-3], [1100, 3.271239e-3],
    [1200, 1.073380e-3], [1300, -8.336607e-4], [1400, -2.941398e-3], [1500, -5.030000e-3],
    [1600, -6.859872e-3], [1700, -8.237322e-3], [1800, -9.037564e-3], [1900, -9.248483e-3],
    [1999, -8.981902e-3],
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


def uniform_batch(trap, n_electrode):
    base = sample_batch(trap, np.random.default_rng(0), 1024, 1, 1)
    electrode_points, electrode_values = [], []
    theta = 2 * np.pi * (np.arange(n_electrode) + 0.37) / n_electrode
    for electrode in trap.electrodes():
        electrode_points.append(np.column_stack([
            electrode.cx + electrode.radius * np.cos(theta),
            electrode.cy + electrode.radius * np.sin(theta),
        ]))
        electrode_values.append(np.full((n_electrode, 1), trap.electrode_voltage(electrode)))
    outer_theta = 2 * np.pi * (np.arange(256) + 0.19) / 256
    outer_points = np.column_stack([trap.outer_radius * np.cos(outer_theta), trap.outer_radius * np.sin(outer_theta)])
    return Batch(
        interior=base.interior,
        electrode_points=torch.tensor(np.vstack(electrode_points), dtype=torch.float32),
        electrode_voltages=torch.tensor(np.vstack(electrode_values), dtype=torch.float32),
        outer_points=torch.tensor(outer_points, dtype=torch.float32),
        outer_voltages=torch.zeros((256, 1), dtype=torch.float32),
    )


def harmonic_coefficients(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    matrix = np.column_stack([np.ones(len(points))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return coefficients[1], coefficients[3]


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.25, outer_radius=5.0, V0=1.0)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
probe = probe[np.hypot(probe[:, 0], probe[:, 1]) <= 0.6]
probe_tensor = torch.tensor(probe, dtype=torch.float32)
east = trap.electrodes()[0]
dense_theta = np.linspace(-np.pi, np.pi, 721)
dense_east = np.column_stack([east.cx + east.radius * np.cos(dense_theta), east.cy + east.radius * np.sin(dense_theta)])
dense_east_tensor = torch.tensor(dense_east, dtype=torch.float32)
results = {}

for n_electrode in [64, 256]:
    label = f"uniform {n_electrode}"
    batch = uniform_batch(trap, n_electrode)
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
        (lpde + 100 * lbc).backward()
        optimizer.step()
        if step % 100 == 0 or step + 1 == STEPS:
            with torch.no_grad():
                center_prediction = field_model(probe_tensor).numpy().ravel()
                east_error = field_model(dense_east_tensor).numpy().ravel() - trap.electrode_voltage(east)
            a2, a6 = harmonic_coefficients(probe, center_prediction)
            q = a6 / a2
            dense_rms = np.sqrt(np.mean(east_error**2))
            row = (step, lpde.item(), lbc.item(), q, abs(q - EXACT_Q), dense_rms)
            records.append(row)
            if best is None or row[4] < best[0]:
                best = (row[4], step, east_error.copy())
            print(f"{label:>11s} step={step:04d} q={q:+.6e} |dq|={abs(q-EXACT_Q):.3e} denseEast={dense_rms:.3e}")
    results[label] = {"records": np.array(records), "final_error": east_error, "best": best}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].plot(CONTROL[:, 0], CONTROL[:, 1], marker="o", markersize=3, label="random 64 (Plot 29)")
for label, result in results.items():
    records = result["records"]
    axes[0, 0].plot(records[:, 0], records[:, 3], marker="o", markersize=3, label=label)
axes[0, 0].axhline(EXACT_Q, color="black", linestyle="--", label="exact")
axes[0, 0].axhline(0, color="grey", linewidth=1)
axes[0, 0].set(xlabel="step", ylabel="signed a6/a2", title="Does angular quadrature stabilize the harmonic?")
axes[0, 0].legend(fontsize=8)

for label, result in results.items():
    records = result["records"]
    axes[0, 1].semilogy(records[:, 0], records[:, 5], marker="o", markersize=3, label=label)
axes[0, 1].set(xlabel="step", ylabel="dense east-electrode RMSE", title="Unseen surface accuracy")
axes[0, 1].legend()

for label, result in results.items():
    axes[1, 0].plot(np.degrees(dense_theta), result["final_error"], label=f"{label} final")
    axes[1, 0].plot(np.degrees(dense_theta), result["best"][2], linestyle="--", label=f"{label} best-q step {result['best'][1]}")
axes[1, 0].axhline(0, color="black", linewidth=1)
axes[1, 0].set(xlabel="angle around east electrode (degrees)", ylabel="potential error / V0", title="Structured surface mode at best and final checkpoints")
axes[1, 0].legend(fontsize=7)

labels = ["random 64", *results.keys()]
final_errors = [abs(CONTROL[-1, 1] - EXACT_Q)] + [result["records"][-1, 4] for result in results.values()]
best_errors = [np.min(np.abs(CONTROL[:, 1] - EXACT_Q))] + [result["best"][0] for result in results.values()]
x = np.arange(len(labels))
axes[1, 1].bar(x - 0.18, final_errors, 0.36, label="final")
axes[1, 1].bar(x + 0.18, best_errors, 0.36, label="best observed")
axes[1, 1].set_xticks(x, labels, rotation=15)
axes[1, 1].set_yscale("log")
axes[1, 1].set(ylabel="absolute harmonic error", title="Physical improvement gate")
axes[1, 1].legend()

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: uniform electrode quadrature resolves the hidden boundary mode", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

for label, result in results.items():
    records = result["records"]
    print(f"SUMMARY {label}: final q={records[-1,3]:+.6e} final |dq|={records[-1,4]:.3e}; best step={result['best'][1]} best |dq|={result['best'][0]:.3e}; final dense east RMSE={records[-1,5]:.3e}")
print("CONCLUSION: uniform quadrature changes the crossing time but does not improve final error; density is decisively rejected. Uniform 64 and 256 have the same trajectory, final dense surface RMSE, and harmonic error to about 1e-6. Sixty-four points already resolve the smooth boundary mode; adding points supplies no new gradient information.")
print("saved", OUTPUT)
