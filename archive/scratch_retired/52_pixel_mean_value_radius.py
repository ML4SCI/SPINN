import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.models.backends.pixel import PixelBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from pig_gate_common import QuadrupoleOutput, exact_field, uniform_batch


OUTPUT = "scratch/52_pixel_mean_value_radius.png"
HYPOTHESIS = "The previous mean-value circles were too nonlocal: smaller circles should preserve the harmonicity gain while recovering fixed-field accuracy, until float32 cancellation makes the Taylor-scaled residual noisy."
RADIUS_SETS = {"small 0.04/0.08": [0.04, 0.08], "middle 0.08/0.16": [0.08, 0.16], "large 0.16/0.32": [0.16, 0.32]}
ADAM_STEPS = 200


class QuinticPixel(PixelBackend):
    @staticmethod
    def _cosine_kernel(weight):
        return 6.0 * weight**5 - 15.0 * weight**4 + 10.0 * weight**3


def safe_centers(trap, count, margin):
    candidates = trap.sample_interior(count * 10, np.random.default_rng(31))
    keep = (trap.electrode_sdf(candidates) > margin) & (trap.outer_sdf(candidates) > margin)
    return torch.tensor(candidates[keep][:count], dtype=torch.float32)


def physics_loss(field, centers, radii):
    theta = torch.arange(24, dtype=centers.dtype) * (2.0 * torch.pi / 24.0)
    directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    center_values = field(centers).squeeze(-1)
    estimates = []
    for radius in radii:
        rings = centers[:, None, :] + radius * directions[None, :, :]
        ring_values = field(rings.reshape(-1, 2)).reshape(len(centers), len(theta))
        estimates.append(4.0 * (ring_values.mean(dim=1) - center_values) / radius**2)
    return torch.mean(torch.stack(estimates) ** 2)


def train(label, radii, trap, batch, centers):
    torch.manual_seed(7)
    backend = QuinticPixel(outer_radius=5.0, grid_resolution=16, feature_dim=4, num_grids=4, decoder_layers=2, decoder_width=16)
    field = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field)
    optimizer = torch.optim.Adam(backend.parameters(), lr=2e-3)
    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        physics = physics_loss(field, centers, radii)
        boundary = boundary_loss(model, batch)
        (physics + 100.0 * boundary).backward()
        optimizer.step()
        if step % 50 == 0 or step + 1 == ADAM_STEPS:
            print(f"{label:17s} step={step:03d} physics={physics.item():.3e} BC={boundary.item():.3e}")
    return model, field


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def mean_value_defect(field, radii):
    rng = np.random.default_rng(19)
    angle = rng.uniform(0.0, 2.0 * np.pi, 48)
    center_radius = 0.42 * np.sqrt(rng.uniform(size=48))
    centers = np.column_stack([center_radius * np.cos(angle), center_radius * np.sin(angle)])
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    with torch.no_grad():
        center_values = field(torch.tensor(centers, dtype=torch.float32)).numpy().ravel()
    defects = []
    for radius in radii:
        rings = centers[:, None, :] + radius * directions[None, :, :]
        with torch.no_grad():
            ring_values = field(torch.tensor(rings.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(len(centers), len(theta))
        defects.append(np.sqrt(np.mean((ring_values.mean(axis=1) - center_values) ** 2)))
    return np.array(defects)


def harmonic_ratios(field, trap, radii, reference=False):
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    ratios = []
    for radius in radii:
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        if reference:
            potential = exact_field(trap, points)[0]
        else:
            with torch.no_grad():
                potential = field(torch.tensor(points, dtype=torch.float32)).numpy().ravel()
        a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / radius**2
        a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / radius**6
        ratios.append(abs(a6 / a2) * radius**4)
    return np.array(ratios)


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0)
batch = uniform_batch(trap, n_interior=384, n_electrode=48, n_outer=128, interior_seed=4)
centers = safe_centers(trap, 128, 0.34)
models = {label: train(label, radii, trap, batch, centers) for label, radii in RADIUS_SETS.items()}

axis = np.linspace(-0.7, 0.7, 61)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
reference_potential, reference_field = exact_field(trap, probe)
v_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))
held_radii = np.linspace(0.02, 0.18, 9)
harmonic_radii = np.linspace(0.12, 0.62, 11)
reference_harmonics = harmonic_ratios(None, trap, harmonic_radii, reference=True)
results = {}
for label, (model, field) in models.items():
    potential, electric_field = potential_field(field, probe)
    electrode_prediction = field(batch.electrode_points).detach().numpy().ravel()
    electrode_target = batch.electrode_voltages.numpy().ravel()
    outer_prediction = field(batch.outer_points).detach().numpy().ravel()
    results[label] = {
        "error": (potential - reference_potential).reshape(gx.shape),
        "v": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
        "e": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
        "bc": max(np.max(np.abs(electrode_prediction - electrode_target)), np.max(np.abs(outer_prediction))),
        "mean_value": mean_value_defect(field, held_radii),
        "harmonics": harmonic_ratios(field, trap, harmonic_radii),
    }
    print(f"SUMMARY {label:17s} V={results[label]['v']:.3e} E={results[label]['e']:.3e} max_BC={results[label]['bc']:.3e}")

labels = list(RADIUS_SETS)
colors = ["tab:blue", "tab:green", "tab:red"]
limit = max(np.max(np.abs(results[label]["error"])) for label in labels)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for ax, label in zip(axes[0], labels):
    image = ax.imshow(results[label]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"{label}\nV={results[label]['v']:.2e}, E={results[label]['e']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")

x = np.arange(len(labels))
width = 0.36
axes[1, 0].bar(x - width / 2, [results[label]["v"] for label in labels], width, label="V error")
axes[1, 0].bar(x + width / 2, [results[label]["e"] for label in labels], width, label="E error")
axes[1, 0].set_xticks(x, ["small", "middle", "large"])
axes[1, 0].set(yscale="log", title="Independent MFS field accuracy", ylabel="normalized spatial RMSE")
for label, color in zip(labels, colors):
    axes[1, 1].plot(held_radii, results[label]["mean_value"], marker="o", color=color, label=label)
    axes[1, 2].plot(harmonic_radii, results[label]["harmonics"], marker="o", color=color, label=label)
axes[1, 1].set(yscale="log", title="Held-out mean-value defect", xlabel="circle radius / r0", ylabel="RMS mean-value error")
axes[1, 2].plot(harmonic_radii, reference_harmonics, color="black", linestyle="--", label="MFS reference")
axes[1, 2].set(yscale="log", title="Physical hexapole contribution", xlabel="measurement radius / r0", ylabel="|a6 r6| / |a2 r2|")
for ax in axes[1]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Mean-value scale test: local PDE approximation versus nonlocal smoothing", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

best_field = min(labels, key=lambda label: results[label]["v"] + results[label]["e"])
best_harmonic = min(labels, key=lambda label: results[label]["mean_value"][-1])
print("PLOT READOUT best field scale:", best_field)
print("PLOT READOUT best held-out harmonicity scale:", best_harmonic)
if best_field == best_harmonic:
    print(f"CONCLUSION: {best_field} wins both independent views; use only that scale in the next PIG transfer experiment.")
else:
    print("CONCLUSION: no circle scale resolves the accuracy/harmonicity trade-off; test a mixed strong-plus-mean-value loss before transferring the idea to PIG.")
print("saved", OUTPUT)
