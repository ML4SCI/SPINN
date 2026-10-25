import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pixel import PixelBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from spinn.physics.laplace import pullback_laplacian
from pig_gate_common import QuadrupoleOutput, exact_field, uniform_batch


OUTPUT = "scratch/51_pixel_mean_value_physics.png"
HYPOTHESIS = "For the same C2 PIXEL representation, a Taylor-scaled mean-value constraint can learn the fixed Paul-trap field without second derivatives and will reduce reference-free harmonicity defects relative to the strong-form Laplacian loss."
ADAM_STEPS = 300
MEAN_VALUE_RADII = [0.08, 0.16]


class QuinticPixel(PixelBackend):
    @staticmethod
    def _cosine_kernel(weight):
        return 6.0 * weight**5 - 15.0 * weight**4 + 10.0 * weight**3


def safe_centers(trap, count, margin):
    candidates = trap.sample_interior(count * 8, np.random.default_rng(31))
    keep = (trap.electrode_sdf(candidates) > margin) & (trap.outer_sdf(candidates) > margin)
    return torch.tensor(candidates[keep][:count], dtype=torch.float32)


def scaled_mean_value_loss(field, centers):
    theta = torch.arange(24, dtype=centers.dtype) * (2.0 * torch.pi / 24.0)
    directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    center_values = field(centers).squeeze(-1)
    losses = []
    for radius in MEAN_VALUE_RADII:
        rings = centers[:, None, :] + radius * directions[None, :, :]
        ring_values = field(rings.reshape(-1, 2)).reshape(len(centers), len(theta))
        laplacian_estimate = 4.0 * (ring_values.mean(dim=1) - center_values) / radius**2
        losses.append(torch.mean(laplacian_estimate**2))
    return torch.stack(losses).mean()


def train(mode, trap, batch, centers):
    torch.manual_seed(7)
    backend = QuinticPixel(outer_radius=trap.outer_radius, grid_resolution=16, feature_dim=4, num_grids=4, decoder_layers=2, decoder_width=16)
    field = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field)
    optimizer = torch.optim.Adam(backend.parameters(), lr=2e-3)
    history = []
    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        if mode == "strong Laplacian":
            potential, _ = model(batch.interior)
            physics = pde_loss(potential, batch.interior, model.shape_network)
        else:
            physics = scaled_mean_value_loss(field, centers)
        boundary = boundary_loss(model, batch)
        (physics + 100.0 * boundary).backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == ADAM_STEPS:
            history.append((step, physics.item(), boundary.item()))
            print(f"{mode:17s} step={step:03d} physics={physics.item():.3e} BC={boundary.item():.3e}")
    return model, field, np.array(history)


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def mean_value_defect(field, circle_radii, samples=96):
    rng = np.random.default_rng(19)
    angle = rng.uniform(0.0, 2.0 * np.pi, 48)
    center_radius = 0.42 * np.sqrt(rng.uniform(size=48))
    centers = np.column_stack([center_radius * np.cos(angle), center_radius * np.sin(angle)])
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    with torch.no_grad():
        center_values = field(torch.tensor(centers, dtype=torch.float32)).numpy().ravel()
    defects = []
    for radius in circle_radii:
        ring = centers[:, None, :] + radius * np.column_stack([np.cos(theta), np.sin(theta)])[None, :, :]
        with torch.no_grad():
            ring_values = field(torch.tensor(ring.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(len(centers), samples)
        defects.append(np.sqrt(np.mean((center_values - ring_values.mean(axis=1)) ** 2)))
    return np.array(defects)


def physical_harmonic_ratio_from_values(potential, theta, radius):
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / radius**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / radius**6
    return abs(a6 / a2) * radius**4


def physical_harmonic_ratios(field, trap, radii, reference=False):
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    ratios = []
    for radius in radii:
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        if reference:
            potential = exact_field(trap, points)[0]
        else:
            with torch.no_grad():
                potential = field(torch.tensor(points, dtype=torch.float32)).numpy().ravel()
        ratios.append(physical_harmonic_ratio_from_values(potential, theta, radius))
    return np.array(ratios)


def laplacian_line(model, x):
    points = torch.tensor(np.column_stack([x, np.full_like(x, 0.137)]), dtype=torch.float32, requires_grad=True)
    potential, _ = model(points)
    return pullback_laplacian(potential, points, model.shape_network).detach().numpy().ravel()


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0)
batch = uniform_batch(trap, n_interior=384, n_electrode=48, n_outer=128, interior_seed=4)
centers = safe_centers(trap, 128, max(MEAN_VALUE_RADII) + 0.01)
experiments = {}
for mode in ["strong Laplacian", "mean value"]:
    experiments[mode] = train(mode, trap, batch, centers)

axis = np.linspace(-0.7, 0.7, 61)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
reference_potential, reference_field = exact_field(trap, probe)
v_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))
circle_radii = np.linspace(0.02, 0.18, 9)
harmonic_radii = np.linspace(0.12, 0.62, 11)
reference_harmonics = physical_harmonic_ratios(None, trap, harmonic_radii, reference=True)
line_x = np.linspace(-0.7, 0.7, 1401)
results = {}
for mode, (model, field, history) in experiments.items():
    potential, electric_field = potential_field(field, probe)
    electrode_prediction = field(batch.electrode_points).detach().numpy().ravel()
    electrode_target = batch.electrode_voltages.numpy().ravel()
    outer_prediction = field(batch.outer_points).detach().numpy().ravel()
    results[mode] = {
        "error": (potential - reference_potential).reshape(gx.shape),
        "v_rmse": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
        "e_rmse": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
        "bc_max": max(np.max(np.abs(electrode_prediction - electrode_target)), np.max(np.abs(outer_prediction))),
        "mean_value": mean_value_defect(field, circle_radii),
        "harmonics": physical_harmonic_ratios(field, trap, harmonic_radii),
        "laplacian": laplacian_line(model, line_x),
        "history": history,
    }
    print(f"SUMMARY {mode:17s} V_RMSE={results[mode]['v_rmse']:.3e} E_RMSE={results[mode]['e_rmse']:.3e} max_BC_error={results[mode]['bc_max']:.3e}")

labels = ["strong Laplacian", "mean value"]
colors = ["tab:orange", "tab:green"]
limit = max(np.max(np.abs(results[label]["error"])) for label in labels)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for ax, label in zip(axes[0, :2], labels):
    image = ax.imshow(results[label]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"{label} - MFS reference\nV={results[label]['v_rmse']:.2e}, E={results[label]['e_rmse']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")
for label, color in zip(labels, colors):
    history = results[label]["history"]
    axes[0, 2].plot(history[:, 0], history[:, 2], color=color, marker=".", label=label)
axes[0, 2].set(yscale="log", title="Same boundary condition during training", xlabel="Adam step", ylabel="boundary MSE")

for label, color in zip(labels, colors):
    axes[1, 0].plot(circle_radii, results[label]["mean_value"], marker="o", color=color, label=label)
    axes[1, 1].plot(harmonic_radii, results[label]["harmonics"], marker="o", color=color, label=label)
    axes[1, 2].plot(line_x, results[label]["laplacian"], color=color, linewidth=1.0, label=label)
axes[1, 0].set(yscale="log", title="Held-out mean-value defect", xlabel="averaging-circle radius / r0", ylabel="RMS mean-value error")
axes[1, 1].plot(harmonic_radii, reference_harmonics, color="black", linestyle="--", label="MFS reference")
axes[1, 1].set(yscale="log", title="Physical hexapole contribution", xlabel="measurement radius / r0", ylabel="|a6 r6| / |a2 r2|")
axes[1, 2].set(title="Strong Laplacian evaluated after training", xlabel="x/r0 at y=0.137 r0", ylabel="model Laplacian")
for ax in [axes[0, 2], *axes[1]]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Can the harmonic mean-value identity replace noisy second derivatives in PIXEL?", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

strong_mv = results["strong Laplacian"]["mean_value"][-1]
mean_mv = results["mean value"]["mean_value"][-1]
print(f"PLOT READOUT largest-circle mean-value defect strong={strong_mv:.3e} mean-value={mean_mv:.3e}")
if mean_mv < strong_mv and results["mean value"]["v_rmse"] < results["strong Laplacian"]["v_rmse"]:
    print("CONCLUSION: the derivative-free constraint improves both its held-out harmonic identity and the independent potential field; next test whether it transfers to PIG unchanged.")
elif mean_mv < strong_mv:
    print("CONCLUSION: the derivative-free constraint improves harmonic consistency but not independent field accuracy at this budget; next vary only its circle scale before changing architecture.")
else:
    print("CONCLUSION: the mean-value replacement does not improve its own held-out diagnostic; reject this implementation before touching PIG or production code.")
print("saved", OUTPUT)
