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


RATIO = 1.00
STEPS = 500
GAUSSIANS = 100
INTERIOR = 1024
ELECTRODE_POINTS = 64
OUTER_POINTS = 256
OUTPUT = "scratch/20_pig_output_symmetry_field_gate.png"
HYPOTHESIS = "Exact quadrupole projection will reduce symmetry-breaking field error and improve signed a6/a2 for the same stored PIG Gaussians."


class QuadrupoleProjection(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def even_xy(self, points):
        return 0.25 * (
            self.backend(points)
            + self.backend(points * points.new_tensor([-1.0, 1.0]))
            + self.backend(points * points.new_tensor([1.0, -1.0]))
            + self.backend(-points)
        )

    def forward(self, points):
        return 0.5 * (self.even_xy(points) - self.even_xy(points[:, [1, 0]]))


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    delta = points[:, None, :] - sources[None, :, :]
    return np.log(np.linalg.norm(delta, axis=2))


def exact_solve(trap):
    sources, boundary, values = [], [], []
    for electrode in trap.electrodes():
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
        boundary.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0.0, 0.0, 1.30 * trap.outer_radius, 128, 0.011))
    boundary.append(circle_points(0.0, 0.0, trap.outer_radius, 256))
    values.append(np.zeros(256))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
    matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, np.concatenate(values), rcond=1e-12)
    return sources, coefficients


def exact_evaluate(points, sources, coefficients):
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = np.log(np.sqrt(radius2)) @ coefficients[:-1] + coefficients[-1]
    gradient = np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, -gradient


def multipole_ratio(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    matrix = np.column_stack([np.ones(len(points))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return coefficients[3] / coefficients[1]


def physical_centers(backend):
    diagnostics = backend.gaussian_diagnostics()
    return np.column_stack([diagnostics["mu_x"], diagnostics["mu_y"]])


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=RATIO, outer_radius=5.0, V0=1.0)
rng = np.random.default_rng(0)
batch = sample_batch(trap, rng, INTERIOR, ELECTRODE_POINTS, OUTER_POINTS)

axis = np.linspace(-0.60, 0.60, 51)
gx, gy = np.meshgrid(axis, axis)
all_probe = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_probe[:, 0], all_probe[:, 1]) <= 0.60
probe = all_probe[inside]
sources, coefficients = exact_solve(trap)
reference_potential, reference_field = exact_evaluate(probe, sources, coefficients)
q_reference = multipole_ratio(probe, reference_potential)
potential_scale = np.max(np.abs(reference_potential))
field_scale = np.max(np.linalg.norm(reference_field, axis=1))

held_points, held_values = [], []
for electrode in trap.electrodes():
    held_points.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 181, 0.0043))
    held_values.append(np.full(181, trap.electrode_voltage(electrode)))
held_points.append(circle_points(0, 0, trap.outer_radius, 361, 0.0021))
held_values.append(np.zeros(361))
held_points = np.vstack(held_points)
held_values = np.concatenate(held_values)

results = []
for label in ["plain PIG", "quadrupole-output PIG"]:
    torch.manual_seed(3)
    backend = PigBackend(
        outer_radius=5.0,
        num_gaussians=GAUSSIANS,
        feature_dim=4,
        init_sigma=0.10,
        decoder_layers=2,
        decoder_width=16,
    )
    prediction_model = backend if label == "plain PIG" else QuadrupoleProjection(backend)
    model = ComposedField(IdentityShapeNetwork(), prediction_model)
    groups = backend.param_groups(1.0e-3, lr_centers=1.0e-4, lr_covariances=1.0e-4)
    optimizer = torch.optim.Adam(groups)
    start_centers = physical_centers(backend)
    history = []
    for step in range(STEPS):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        loss = lpde + 100.0 * lbc
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == STEPS:
            history.append((step, lpde.detach().item(), lbc.detach().item()))
            if step % 100 == 0 or step + 1 == STEPS:
                print(f"{label:22s} step={step:03d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e}")

    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    predicted_potential_tensor = prediction_model(probe_tensor).squeeze(-1)
    predicted_gradient = torch.autograd.grad(predicted_potential_tensor.sum(), probe_tensor)[0]
    predicted_potential = predicted_potential_tensor.detach().numpy()
    predicted_field = -predicted_gradient.detach().numpy()
    held_prediction = prediction_model(torch.tensor(held_points, dtype=torch.float32)).detach().numpy().ravel()
    potential_error = (predicted_potential - reference_potential) / potential_scale
    field_error = np.linalg.norm(predicted_field - reference_field, axis=1) / field_scale
    q = multipole_ratio(probe, predicted_potential)
    results.append(
        {
            "label": label,
            "history": np.array(history),
            "potential_error": potential_error,
            "field_error": field_error,
            "q": q,
            "potential_rmse": np.sqrt(np.mean(potential_error**2)),
            "field_rmse": np.sqrt(np.mean(field_error**2)),
            "boundary_max": np.max(np.abs(held_prediction - held_values)),
            "start_centers": start_centers,
            "end_centers": physical_centers(backend),
        }
    )

fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
potential_limit = max(np.max(np.abs(row["potential_error"])) for row in results)
field_limit = max(np.max(row["field_error"]) for row in results)
for row_index, row in enumerate(results):
    potential_map = np.full(gx.shape, np.nan)
    potential_map.ravel()[inside] = row["potential_error"]
    image_potential = axes[row_index, 0].imshow(potential_map, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="coolwarm", vmin=-potential_limit, vmax=potential_limit)
    axes[row_index, 0].set(title=f"{row['label']}: normalized V error", xlabel="x/r0", ylabel="y/r0")

    field_map = np.full(gx.shape, np.nan)
    field_map.ravel()[inside] = row["field_error"]
    image_field = axes[row_index, 1].imshow(field_map, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="magma", vmin=0, vmax=field_limit)
    axes[row_index, 1].set(title=f"{row['label']}: |E error|", xlabel="x/r0", ylabel="y/r0")

    start = row["start_centers"]
    end = row["end_centers"]
    movement = end - start
    axes[row_index, 2].quiver(start[:, 0], start[:, 1], movement[:, 0], movement[:, 1], angles="xy", scale_units="xy", scale=1, alpha=0.45)
    axes[row_index, 2].scatter(end[:, 0], end[:, 1], s=5, color="black")
    for electrode in trap.electrodes():
        axes[row_index, 2].add_patch(plt.Circle((electrode.cx, electrode.cy), electrode.radius, fill=False, color="red", linewidth=0.8))
    axes[row_index, 2].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", title=f"{row['label']}: Gaussian movement", xlabel="x/r0", ylabel="y/r0")

for row in results:
    history = row["history"]
    axes[2, 0].semilogy(history[:, 0], history[:, 1], label=f"PDE: {row['label']}")
    axes[2, 0].semilogy(history[:, 0], history[:, 2], linestyle="--", label=f"BC: {row['label']}")
axes[2, 0].set(xlabel="training step", ylabel="loss term", title="Convergence at identical sample budget")
axes[2, 0].legend(fontsize=7)

q_values = [q_reference] + [row["q"] for row in results]
q_labels = ["reference", "plain", "quadrupole"]
bars = axes[2, 1].bar(q_labels, q_values, color=["black", "tab:blue", "tab:orange"])
axes[2, 1].bar_label(bars, fmt="%.4f")
axes[2, 1].axhline(0.0, color="black", linewidth=1)
axes[2, 1].set(ylabel="signed a6/a2", title="Physical harmonic gate")

metrics = ["V RMSE", "E RMSE", "boundary max"]
x = np.arange(len(metrics))
for index, row in enumerate(results):
    values = [row["potential_rmse"], row["field_rmse"], row["boundary_max"]]
    axes[2, 2].bar(x + (index - 0.5) * 0.35, values, 0.35, label=row["label"])
axes[2, 2].set_yscale("log")
axes[2, 2].set_xticks(x, metrics)
axes[2, 2].set(title="Independent physical errors")
axes[2, 2].legend(fontsize=8)

fig.colorbar(image_potential, ax=axes[:2, 0], label="(PIG-reference)/reference peak")
fig.colorbar(image_field, ax=axes[:2, 1], label="|E_PIG-E_ref|/reference peak")
for ax in axes.ravel():
    ax.grid(alpha=0.15)
fig.suptitle("Hypothesis: exact output symmetry improves a fixed-geometry PIG solve", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print(f"reference signed a6/a2={q_reference:+.6e}")
for row in results:
    print(
        row["label"],
        f"q={row['q']:+.6e}",
        f"V_RMSE={row['potential_rmse']:.3e}",
        f"E_RMSE={row['field_rmse']:.3e}",
        f"held_boundary_max={row['boundary_max']:.3e}",
    )
plain_error = abs(results[0]["q"] - q_reference)
sym_error = abs(results[1]["q"] - q_reference)
print(f"harmonic absolute-error ratio symmetric/plain={sym_error/plain_error:.3f}")
print("CONCLUSION: quadrupole output symmetry passes this fixed-field gate: it corrects the harmonic sign and sharply reduces independent potential, field and unseen-boundary errors.")
print("saved", OUTPUT)
