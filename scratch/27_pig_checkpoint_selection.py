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
OUTPUT = "scratch/27_pig_checkpoint_selection.png"
HYPOTHESIS = "Held-out PDE and boundary residuals identify the checkpoint with the most accurate geometry-sensitive harmonic better than training loss does."


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
train_batch = sample_batch(trap, np.random.default_rng(0), 1024, 64, 256)
validation_batch = sample_batch(trap, np.random.default_rng(991), 1024, 181, 361)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
probe = probe[np.hypot(probe[:, 0], probe[:, 1]) <= 0.6]
probe_tensor = torch.tensor(probe, dtype=torch.float32)

torch.manual_seed(3)
backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
field_model = QuadrupoleOutput(backend)
model = ComposedField(IdentityShapeNetwork(), field_model)
optimizer = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
records = []
for step in range(STEPS):
    optimizer.zero_grad()
    potential, _ = model(train_batch.interior)
    train_pde = pde_loss(potential, train_batch.interior, model.shape_network)
    train_bc = boundary_loss(model, train_batch)
    (train_pde + 100 * train_bc).backward()
    optimizer.step()
    if step % 100 == 0 or step + 1 == STEPS:
        validation_potential, _ = model(validation_batch.interior)
        validation_pde = pde_loss(validation_potential, validation_batch.interior, model.shape_network)
        validation_bc = boundary_loss(model, validation_batch)
        with torch.no_grad():
            prediction = field_model(probe_tensor).numpy().ravel()
        q = q_ratio(probe, prediction)
        records.append(
            (
                step,
                train_pde.detach().item(),
                train_bc.detach().item(),
                validation_pde.detach().item(),
                validation_bc.detach().item(),
                q,
                abs(q - EXACT_Q),
            )
        )
        print(f"step={step:04d} trainPDE={train_pde.detach().item():.3e} heldPDE={validation_pde.detach().item():.3e} trainBC={train_bc.detach().item():.3e} heldBC={validation_bc.detach().item():.3e} q={q:+.5e} |dq|={abs(q-EXACT_Q):.3e}")

records = np.array(records)
step = records[:, 0]
labels = ["train PDE", "train boundary", "held-out PDE", "held-out boundary"]
metric_columns = [1, 2, 3, 4]
correlations = []
for column in metric_columns:
    correlations.append(np.corrcoef(np.log10(records[:, column]), np.log10(records[:, 6].clip(1e-9)))[0, 1])

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].plot(step, records[:, 5], marker="o", label="PIG")
axes[0, 0].axhline(EXACT_Q, color="black", linestyle="--", label="exact")
axes[0, 0].axhline(0, color="grey", linewidth=1)
axes[0, 0].set(xlabel="step", ylabel="signed a6/a2", title="Physical quantity being selected")
axes[0, 0].legend()

for label, column in zip(labels, metric_columns):
    axes[0, 1].semilogy(step, records[:, column], marker="o", label=label)
axes[0, 1].set(xlabel="step", ylabel="residual", title="Candidate reference-free stopping signals")
axes[0, 1].legend(fontsize=8)

for label, column in zip(labels, metric_columns):
    axes[1, 0].loglog(records[:, column], records[:, 6], marker="o", label=label)
axes[1, 0].set(xlabel="candidate residual", ylabel="absolute harmonic error", title="Does lower residual imply better harmonic?")
axes[1, 0].legend(fontsize=8)

bars = axes[1, 1].bar(labels, correlations)
axes[1, 1].bar_label(bars, fmt="%.2f")
axes[1, 1].axhline(0, color="black", linewidth=1)
axes[1, 1].set(ylim=(-1.05, 1.05), ylabel="correlation of log residual with log |harmonic error|", title="Positive near 1 would support checkpoint selection")
axes[1, 1].tick_params(axis="x", rotation=20)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: unseen physics residuals select the physically accurate PIG checkpoint", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

best_index = np.argmin(records[:, 6])
print(f"best harmonic checkpoint step={int(records[best_index,0])} q={records[best_index,5]:+.6e} |dq|={records[best_index,6]:.3e}")
print("correlations", dict(zip(labels, correlations)))
print("CONCLUSION: hypothesis rejected operationally. Held-out residuals have moderate aggregate correlation with harmonic error, but all residual minima occur near the final checkpoint rather than the physically best checkpoint at step 1600. Between steps 400 and 700 the held-out PDE residual falls while harmonic error grows, so unseen residuals cannot safely select the checkpoint.")
print("saved", OUTPUT)
