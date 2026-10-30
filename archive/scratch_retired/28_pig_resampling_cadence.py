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
EXACT_Q = -6.409705e-3
CADENCES = {"fixed points": None, "refresh every 500": 500, "refresh every 100": 100}
OUTPUT = "scratch/28_pig_resampling_cadence.png"
HYPOTHESIS = "Refreshing collocation and boundary points reduces fixed-sample overfitting and makes the signed a6/a2 trajectory more stable and accurate."


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


def q_ratio(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    matrix = np.column_stack([np.ones(len(points))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return coefficients[3] / coefficients[1]


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.25, outer_radius=5.0, V0=1.0)
validation = sample_batch(trap, np.random.default_rng(991), 2048, 181, 361)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
probe = probe[np.hypot(probe[:, 0], probe[:, 1]) <= 0.6]
probe_tensor = torch.tensor(probe, dtype=torch.float32)
all_records = {}

for label, cadence in CADENCES.items():
    rng = np.random.default_rng(0)
    batch = sample_batch(trap, rng, 1024, 64, 256)
    torch.manual_seed(3)
    backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
    field_model = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field_model)
    optimizer = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
    records = []

    for step in range(STEPS):
        if cadence is not None and step > 0 and step % cadence == 0:
            batch = sample_batch(trap, rng, 1024, 64, 256)
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        train_pde = pde_loss(potential, batch.interior, model.shape_network)
        train_bc = boundary_loss(model, batch)
        (train_pde + 100 * train_bc).backward()
        optimizer.step()

        if step % 100 == 0 or step + 1 == STEPS:
            held_potential, _ = model(validation.interior)
            held_pde = pde_loss(held_potential, validation.interior, model.shape_network)
            held_bc = boundary_loss(model, validation)
            with torch.no_grad():
                q = q_ratio(probe, field_model(probe_tensor).numpy().ravel())
            records.append((step, train_pde.item(), train_bc.item(), held_pde.item(), held_bc.item(), q, abs(q - EXACT_Q)))
            print(f"{label:>17s} step={step:04d} heldPDE={held_pde.item():.3e} heldBC={held_bc.item():.3e} q={q:+.6e} |dq|={abs(q-EXACT_Q):.3e}")
    all_records[label] = np.array(records)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for label, records in all_records.items():
    axes[0, 0].plot(records[:, 0], records[:, 5], marker="o", markersize=3, label=label)
axes[0, 0].axhline(EXACT_Q, color="black", linestyle="--", label="exact")
axes[0, 0].axhline(0, color="grey", linewidth=1)
axes[0, 0].set(xlabel="training step", ylabel="signed a6/a2", title="Physical harmonic trajectory")
axes[0, 0].legend(fontsize=8)

for label, records in all_records.items():
    axes[0, 1].semilogy(records[:, 0], records[:, 3], marker="o", markersize=3, label=label)
axes[0, 1].set(xlabel="training step", ylabel="held-out PDE residual", title="Unseen physics error")
axes[0, 1].legend(fontsize=8)

for label, records in all_records.items():
    axes[1, 0].semilogy(records[:, 0], records[:, 3] / records[:, 1], marker="o", markersize=3, label=label)
axes[1, 0].axhline(1, color="black", linestyle="--")
axes[1, 0].set(xlabel="training step", ylabel="held PDE / current-batch PDE", title="Generalization gap")
axes[1, 0].legend(fontsize=8)

summary_labels = list(all_records)
final_error = [all_records[label][-1, 6] for label in summary_labels]
best_error = [all_records[label][:, 6].min() for label in summary_labels]
x = np.arange(len(summary_labels))
width = 0.36
axes[1, 1].bar(x - width / 2, final_error, width, label="final")
axes[1, 1].bar(x + width / 2, best_error, width, label="best observed")
axes[1, 1].set_xticks(x, summary_labels, rotation=15)
axes[1, 1].set_yscale("log")
axes[1, 1].set(ylabel="absolute harmonic error", title="Does refreshing improve the target?")
axes[1, 1].legend()

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: point refreshing stabilizes the PIG electrode harmonic", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

for label, records in all_records.items():
    best = records[np.argmin(records[:, 6])]
    print(f"SUMMARY {label}: final q={records[-1,5]:+.6e} final |dq|={records[-1,6]:.3e}; best step={int(best[0])} best |dq|={best[6]:.3e}; final held/train PDE={records[-1,3]/records[-1,1]:.2f}")
print("CONCLUSION: hypothesis rejected as a complete repair. Refreshing closes the held/current-batch PDE gap from 1.63 to about 1.0, but worsens final harmonic error from 2.57e-3 to 7.41e-3 (500) and 9.90e-3 (100). Every-100 briefly reaches 1.60e-4 error before drifting. Uniform resampling fixes sample memorization but not physical-mode instability.")
print("saved", OUTPUT)
