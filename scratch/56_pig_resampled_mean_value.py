import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from pig_gate_common import exact_field, uniform_batch


OUTPUT = "scratch/56_pig_resampled_mean_value.png"
HYPOTHESIS = "Seed 11's held-out mean-value failure is quadrature overfitting: resampling safe centers and radii should improve held-out harmonicity while retaining the mixed loss's field gain."
SEED = 11
ADAM_STEPS = 300
LBFGS_OUTER = 6
LBFGS_INNER = 10


def safe_pool(trap):
    candidates = trap.sample_interior(12000, np.random.default_rng(31))
    keep = (trap.electrode_sdf(candidates) > 0.20) & (trap.outer_sdf(candidates) > 0.20)
    return torch.tensor(candidates[keep][:2048], dtype=torch.float32)


def draw_design(pool, generator, resampled):
    if resampled:
        indices = torch.randperm(len(pool), generator=generator)[:48]
        radii = 0.04 + 0.14 * torch.rand(2, generator=generator)
        return pool[indices], radii.tolist()
    return pool[:48], [0.08, 0.16]


def mean_value_loss(field, centers, radii):
    theta = torch.arange(16, dtype=centers.dtype) * (2.0 * torch.pi / 16.0)
    directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    center_values = field(centers).squeeze(-1)
    estimates = []
    for radius in radii:
        rings = centers[:, None, :] + radius * directions[None, :, :]
        ring_values = field(rings.reshape(-1, 2)).reshape(len(centers), len(theta))
        estimates.append(4.0 * (ring_values.mean(dim=1) - center_values) / radius**2)
    return torch.mean(torch.stack(estimates) ** 2)


def total_loss(model, field, batch, centers, radii):
    potential, _ = model(batch.interior)
    strong = pde_loss(potential, batch.interior, model.shape_network)
    mean = mean_value_loss(field, centers, radii)
    boundary = boundary_loss(model, batch)
    return strong + 0.1 * mean + 100.0 * boundary


def train(label, resampled, batch, pool):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(101)
    field = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16, symmetry="d4")
    model = ComposedField(IdentityShapeNetwork(), field)
    adam = torch.optim.Adam(field.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
    for step in range(ADAM_STEPS):
        centers, radii = draw_design(pool, generator, resampled)
        adam.zero_grad()
        loss = total_loss(model, field, batch, centers, radii)
        loss.backward()
        adam.step()
    lbfgs = torch.optim.LBFGS(field.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=30, line_search_fn="strong_wolfe")
    for outer in range(LBFGS_OUTER):
        centers, radii = draw_design(pool, generator, resampled)
        def closure():
            lbfgs.zero_grad()
            loss = total_loss(model, field, batch, centers, radii)
            loss.backward()
            return loss
        lbfgs.step(closure)
        print(f"{label:10s} LBFGS outer={outer} radii={radii[0]:.3f},{radii[1]:.3f}")
    return field


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def held_mean_value(field, radii):
    rng = np.random.default_rng(19)
    angle = rng.uniform(0.0, 2.0 * np.pi, 64)
    center_radius = 0.42 * np.sqrt(rng.uniform(size=64))
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
batch = uniform_batch(trap, n_interior=512, n_electrode=48, n_outer=128, interior_seed=4)
pool = safe_pool(trap)
models = {"fixed": train("fixed", False, batch, pool), "resampled": train("resampled", True, batch, pool)}

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
for label, field in models.items():
    potential, electric_field = potential_field(field, probe)
    results[label] = {
        "error": (potential - reference_potential).reshape(gx.shape),
        "v": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
        "e": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
        "mean": held_mean_value(field, held_radii),
        "harmonics": harmonic_ratios(field, trap, harmonic_radii),
    }
    print(f"SUMMARY {label:10s} V={results[label]['v']:.3e} E={results[label]['e']:.3e} held_mean={results[label]['mean'][-1]:.3e}")

limit = max(np.max(np.abs(results[label]["error"])) for label in models)
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
for ax, label in zip(axes[0], models):
    image = ax.imshow(results[label]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"{label} quadrature - MFS\nV={results[label]['v']:.2e}, E={results[label]['e']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")
for label, color in [("fixed", "tab:blue"), ("resampled", "tab:green")]:
    axes[1, 0].plot(held_radii, results[label]["mean"], marker="o", color=color, label=label)
    axes[1, 1].plot(harmonic_radii, results[label]["harmonics"], marker="o", color=color, label=label)
axes[1, 0].set(yscale="log", title="Unseen centers and radii", xlabel="circle radius / r0", ylabel="RMS mean-value error")
axes[1, 1].plot(harmonic_radii, reference_harmonics, color="black", linestyle="--", label="MFS reference")
axes[1, 1].set(yscale="log", title="Physical hexapole contribution", xlabel="measurement radius / r0", ylabel="|a6 r6| / |a2 r2|")
for ax in axes[1]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Does stochastic mean-value collocation fix the seed-11 generalization failure?", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

passes = results["resampled"]["v"] < results["fixed"]["v"] and results["resampled"]["e"] < results["fixed"]["e"] and results["resampled"]["mean"][-1] < results["fixed"]["mean"][-1]
print("PLOT READOUT resampling improves V/E/held mean-value:", passes)
if passes:
    print("CONCLUSION: fixed mean-value quadrature was overfitting; resample safe centers and radii in the next multi-seed common-loss gate.")
else:
    print("CONCLUSION: quadrature overfitting does not explain the seed-11 trade-off; keep fixed sampling and investigate an explicit central harmonic architecture next.")
print("saved", OUTPUT)
