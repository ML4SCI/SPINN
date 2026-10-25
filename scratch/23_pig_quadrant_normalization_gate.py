import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import sample_collocation_batch as sample_batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.decoders import DecoderMLP
from spinn.models.composed import ComposedField, IdentityShapeNetwork


STEPS = 500
K = 100
OUTPUT = "scratch/23_pig_quadrant_normalization_gate.png"
HYPOTHESIS = "Mean normalization caused the tied-PIG slowdown; sqrt normalization should restore feature scale without the fourfold gradient growth of a raw sum."


class QuadrantPig(nn.Module):
    kind = "pig"

    def __init__(self, normalization):
        super().__init__()
        self.normalization = normalization
        self.raw_means = nn.Parameter(torch.logit(torch.rand(4, K, 2).clamp(0.02, 0.98)))
        raw_sigma = torch.log(torch.expm1(torch.tensor(0.09)))
        self.raw_sigmas = nn.Parameter(torch.full((4, K, 2), raw_sigma))
        self.weights = nn.Parameter(torch.normal(0.0, 0.01, (4, K)))
        self.decoder = DecoderMLP(4, layers=2, width=16)
        self.register_buffer("mirrors", torch.tensor([[[1., 0.], [0., 1.]], [[-1., 0.], [0., 1.]], [[1., 0.], [0., -1.]], [[-1., 0.], [0., -1.]]]))

    def features(self, points):
        coords = points / 5.0
        means = torch.sigmoid(self.raw_means)
        sigmas = torch.nn.functional.softplus(self.raw_sigmas) + 0.01
        pieces = []
        for mirror in self.mirrors:
            diff = (coords @ mirror).unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
            exponent = torch.sum((diff / sigmas.unsqueeze(1)) ** 2, dim=-1)
            gaussian = torch.exp(torch.clamp(-0.5 * exponent, min=-60.0, max=0.0))
            pieces.append((gaussian * self.weights.unsqueeze(1)).sum(dim=-1).T)
        divisor = {"sum": 1.0, "sqrt": 2.0, "mean": 4.0}[self.normalization]
        return torch.stack(pieces).sum(dim=0) / divisor

    def forward(self, points):
        features = self.features(points)
        swapped_features = self.features(points[:, [1, 0]])
        return 0.5 * (self.decoder(features) - self.decoder(swapped_features))

    def groups(self):
        return [
            {"params": list(self.decoder.parameters()) + [self.weights], "lr": 1e-3},
            {"params": [self.raw_means], "lr": 1e-4},
            {"params": [self.raw_sigmas], "lr": 1e-4},
        ]


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
trap = FourRodTrap(r0=1.0, rod_radius=1.0, outer_radius=5.0, V0=1.0)
batch = sample_batch(trap, np.random.default_rng(0), 1024, 64, 256)
axis = np.linspace(-0.6, 0.6, 41)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.6
probe = all_points[inside]
reference_potential, reference_field = exact(trap, probe)
q_reference = q_ratio(probe, reference_potential)
p_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))

results = []
for normalization in ["mean", "sqrt", "sum"]:
    torch.manual_seed(3)
    backend = QuadrantPig(normalization)
    model = ComposedField(IdentityShapeNetwork(), backend)
    optimizer = torch.optim.Adam(backend.groups())
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
        if step % 100 == 0 or step + 1 == STEPS:
            print(f"{normalization:4s} step={step:03d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e}")
    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    potential_tensor = backend(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(potential_tensor.sum(), probe_tensor)[0]
    prediction = potential_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    results.append(
        {
            "normalization": normalization,
            "history": np.array(history),
            "q": q_ratio(probe, prediction),
            "p_error": (prediction - reference_potential) / p_scale,
            "e_error": np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale,
        }
    )

fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
for row in results:
    history = row["history"]
    axes[0, 0].semilogy(history[:, 0], history[:, 1], label=row["normalization"])
    axes[0, 1].semilogy(history[:, 0], history[:, 2], label=row["normalization"])
axes[0, 0].set(xlabel="training step", ylabel="PDE loss", title="Physics convergence")
axes[0, 1].set(xlabel="training step", ylabel="boundary loss", title="Boundary convergence")

q_values = [q_reference] + [row["q"] for row in results]
bars = axes[0, 2].bar(["reference", "mean", "sqrt", "sum"], q_values, color=["black", "tab:green", "tab:orange", "tab:blue"])
axes[0, 2].bar_label(bars, fmt="%.4f")
axes[0, 2].axhline(0, color="black", linewidth=1)
axes[0, 2].set(ylabel="signed a6/a2", title="Harmonic gate")

limit = max(np.max(np.abs(row["p_error"])) for row in results)
for column, row in enumerate(results):
    error_map = np.full(gx.shape, np.nan)
    error_map.ravel()[inside] = row["p_error"]
    image = axes[1, column].imshow(error_map, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="coolwarm", vmin=-limit, vmax=limit)
    p_rmse = np.sqrt(np.mean(row["p_error"]**2))
    e_rmse = np.sqrt(np.mean(row["e_error"]**2))
    axes[1, column].set(xlabel="x/r0", ylabel="y/r0", title=f"{row['normalization']}: V {p_rmse:.2%}, E {e_rmse:.2%}")

for ax in axes.ravel():
    ax.grid(alpha=0.2)
    if ax in axes[0, :2]:
        ax.legend()
fig.colorbar(image, ax=axes[1, :], label="normalized potential error")
fig.suptitle("Hypothesis: tied-PIG failure is caused by orbit normalization", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print(f"reference q={q_reference:+.6e}")
for row in results:
    print(row["normalization"], f"q={row['q']:+.6e}", f"V_RMSE={np.sqrt(np.mean(row['p_error']**2)):.3e}", f"E_RMSE={np.sqrt(np.mean(row['e_error']**2)):.3e}")
print("CONCLUSION: reject normalization as the sole cause; lower PDE loss from sqrt/sum does not translate to physical field or harmonic accuracy, so pre-decoder feature aggregation is the deeper problem.")
print("saved", OUTPUT)
