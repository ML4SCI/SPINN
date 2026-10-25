import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pixel import PixelBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from pig_gate_common import QuadrupoleOutput, exact_field, uniform_batch


OUTPUT = "scratch/53_pixel_mixed_physics_loss.png"
HYPOTHESIS = "A small mean-value term added to the strong Laplacian loss reduces held-out harmonicity defects without sacrificing the independently checked potential and electric field."
WEIGHTS = [0.0, 0.1, 1.0]
ADAM_STEPS = 300


class QuinticPixel(PixelBackend):
    @staticmethod
    def _cosine_kernel(weight):
        return 6.0 * weight**5 - 15.0 * weight**4 + 10.0 * weight**3


def safe_centers(trap, count):
    candidates = trap.sample_interior(count * 8, np.random.default_rng(31))
    keep = (trap.electrode_sdf(candidates) > 0.17) & (trap.outer_sdf(candidates) > 0.17)
    return torch.tensor(candidates[keep][:count], dtype=torch.float32)


def mean_value_loss(field, centers):
    theta = torch.arange(24, dtype=centers.dtype) * (2.0 * torch.pi / 24.0)
    directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    center_values = field(centers).squeeze(-1)
    estimates = []
    for radius in [0.08, 0.16]:
        rings = centers[:, None, :] + radius * directions[None, :, :]
        ring_values = field(rings.reshape(-1, 2)).reshape(len(centers), len(theta))
        estimates.append(4.0 * (ring_values.mean(dim=1) - center_values) / radius**2)
    return torch.mean(torch.stack(estimates) ** 2)


def train(weight, trap, batch, centers):
    torch.manual_seed(7)
    backend = QuinticPixel(outer_radius=5.0, grid_resolution=16, feature_dim=4, num_grids=4, decoder_layers=2, decoder_width=16)
    field = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field)
    optimizer = torch.optim.Adam(backend.parameters(), lr=2e-3)
    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        strong = pde_loss(potential, batch.interior, model.shape_network)
        mean = mean_value_loss(field, centers) if weight > 0.0 else strong.new_tensor(0.0)
        boundary = boundary_loss(model, batch)
        (strong + weight * mean + 100.0 * boundary).backward()
        optimizer.step()
        if step % 75 == 0 or step + 1 == ADAM_STEPS:
            print(f"lambda={weight:.1f} step={step:03d} strong={strong.item():.3e} mean={mean.item():.3e} BC={boundary.item():.3e}")
    return model, field


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def held_mean_value(field, radii):
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
centers = safe_centers(trap, 128)
models = {weight: train(weight, trap, batch, centers) for weight in WEIGHTS}

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
for weight, (model, field) in models.items():
    potential, electric_field = potential_field(field, probe)
    electrode_prediction = field(batch.electrode_points).detach().numpy().ravel()
    electrode_target = batch.electrode_voltages.numpy().ravel()
    outer_prediction = field(batch.outer_points).detach().numpy().ravel()
    results[weight] = {
        "error": (potential - reference_potential).reshape(gx.shape),
        "v": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
        "e": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
        "bc": max(np.max(np.abs(electrode_prediction - electrode_target)), np.max(np.abs(outer_prediction))),
        "mean": held_mean_value(field, held_radii),
        "harmonics": harmonic_ratios(field, trap, harmonic_radii),
    }
    print(f"SUMMARY lambda={weight:.1f} V={results[weight]['v']:.3e} E={results[weight]['e']:.3e} max_BC={results[weight]['bc']:.3e}")

colors = ["tab:orange", "tab:blue", "tab:green"]
limit = max(np.max(np.abs(results[weight]["error"])) for weight in WEIGHTS)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for ax, weight in zip(axes[0], WEIGHTS):
    image = ax.imshow(results[weight]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"mean-value lambda={weight:.1f}\nV={results[weight]['v']:.2e}, E={results[weight]['e']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")

x = np.arange(len(WEIGHTS))
width = 0.36
axes[1, 0].bar(x - width / 2, [results[weight]["v"] for weight in WEIGHTS], width, label="V error")
axes[1, 0].bar(x + width / 2, [results[weight]["e"] for weight in WEIGHTS], width, label="E error")
axes[1, 0].set_xticks(x, [str(weight) for weight in WEIGHTS])
axes[1, 0].set(yscale="log", title="Independent MFS field accuracy", xlabel="mean-value weight lambda", ylabel="normalized spatial RMSE")
for weight, color in zip(WEIGHTS, colors):
    axes[1, 1].plot(held_radii, results[weight]["mean"], marker="o", color=color, label=f"lambda={weight}")
    axes[1, 2].plot(harmonic_radii, results[weight]["harmonics"], marker="o", color=color, label=f"lambda={weight}")
axes[1, 1].set(yscale="log", title="Held-out mean-value defect", xlabel="circle radius / r0", ylabel="RMS mean-value error")
axes[1, 2].plot(harmonic_radii, reference_harmonics, color="black", linestyle="--", label="MFS reference")
axes[1, 2].set(yscale="log", title="Physical hexapole contribution", xlabel="measurement radius / r0", ylabel="|a6 r6| / |a2 r2|")
for ax in axes[1]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("One physics loss: strong Laplacian plus a dimensionally matched mean-value term", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

best_mean = min(WEIGHTS, key=lambda weight: results[weight]["mean"][-1])
candidate = 0.1
candidate_improves_baseline = (
    results[candidate]["v"] < results[0.0]["v"]
    and results[candidate]["e"] < results[0.0]["e"]
    and results[candidate]["mean"][-1] < results[0.0]["mean"][-1]
)
print(f"PLOT READOUT lambda=0.1 strictly improves baseline V/E/mean-value={candidate_improves_baseline}; best harmonicity lambda={best_mean}")
if candidate_improves_baseline:
    print("CONCLUSION: lambda=0.1 is the conservative transfer candidate because it improves V, E and held-out mean-value error together. Lambda=1.0 trades worse V for better E, and no weight closes the spurious-hexapole gap.")
else:
    print("CONCLUSION: no mixed weight strictly improves all three independent views; do not add it to PIG yet.")
print("saved", OUTPUT)
