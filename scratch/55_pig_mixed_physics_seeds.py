import matplotlib.pyplot as plt
import numpy as np
import torch

from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from pig_gate_common import exact_field, uniform_batch


OUTPUT = "scratch/55_pig_mixed_physics_seeds.png"
HYPOTHESIS = "The lambda=0.1 mixed physics loss lowers PIG field and harmonicity errors for each matched initialization, rather than winning only for seed 3."
SEEDS = [3, 11, 29]
WEIGHTS = [0.0, 0.1]
ADAM_STEPS = 300
LBFGS_OUTER = 6
LBFGS_INNER = 10


def safe_centers(trap):
    candidates = trap.sample_interior(512, np.random.default_rng(31))
    keep = (trap.electrode_sdf(candidates) > 0.17) & (trap.outer_sdf(candidates) > 0.17)
    return torch.tensor(candidates[keep][:48], dtype=torch.float32)


def mean_value_loss(field, centers):
    theta = torch.arange(16, dtype=centers.dtype) * (2.0 * torch.pi / 16.0)
    directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    center_values = field(centers).squeeze(-1)
    estimates = []
    for radius in [0.08, 0.16]:
        rings = centers[:, None, :] + radius * directions[None, :, :]
        ring_values = field(rings.reshape(-1, 2)).reshape(len(centers), len(theta))
        estimates.append(4.0 * (ring_values.mean(dim=1) - center_values) / radius**2)
    return torch.mean(torch.stack(estimates) ** 2)


def total_loss(model, field, batch, centers, weight):
    potential, _ = model(batch.interior)
    strong = pde_loss(potential, batch.interior, model.shape_network)
    mean = mean_value_loss(field, centers) if weight > 0 else strong.new_tensor(0.0)
    boundary = boundary_loss(model, batch)
    return strong + weight * mean + 100.0 * boundary


def train(seed, weight, batch, centers):
    torch.manual_seed(seed)
    field = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16, symmetry="d4")
    model = ComposedField(IdentityShapeNetwork(), field)
    adam = torch.optim.Adam(field.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
    for step in range(ADAM_STEPS):
        adam.zero_grad()
        loss = total_loss(model, field, batch, centers, weight)
        loss.backward()
        adam.step()
    lbfgs = torch.optim.LBFGS(field.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=30, line_search_fn="strong_wolfe")
    for _ in range(LBFGS_OUTER):
        def closure():
            lbfgs.zero_grad()
            loss = total_loss(model, field, batch, centers, weight)
            loss.backward()
            return loss
        lbfgs.step(closure)
    print(f"finished seed={seed} lambda={weight}")
    return field


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def held_mean_value(field, radius=0.18):
    rng = np.random.default_rng(19)
    angle = rng.uniform(0.0, 2.0 * np.pi, 48)
    center_radius = 0.42 * np.sqrt(rng.uniform(size=48))
    centers = np.column_stack([center_radius * np.cos(angle), center_radius * np.sin(angle)])
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    rings = centers[:, None, :] + radius * np.column_stack([np.cos(theta), np.sin(theta)])[None, :, :]
    with torch.no_grad():
        center_values = field(torch.tensor(centers, dtype=torch.float32)).numpy().ravel()
        ring_values = field(torch.tensor(rings.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(len(centers), len(theta))
    return np.sqrt(np.mean((ring_values.mean(axis=1) - center_values) ** 2))


def harmonic_ratio(field, radius=0.5):
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    with torch.no_grad():
        potential = field(torch.tensor(points, dtype=torch.float32)).numpy().ravel()
    a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / radius**2
    a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / radius**6
    return abs(a6 / a2) * radius**4


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0)
batch = uniform_batch(trap, n_interior=512, n_electrode=48, n_outer=128, interior_seed=4)
centers = safe_centers(trap)
axis = np.linspace(-0.7, 0.7, 51)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
reference_potential, reference_field = exact_field(trap, probe)
v_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))

rows = []
for seed in SEEDS:
    for weight in WEIGHTS:
        field = train(seed, weight, batch, centers)
        potential, electric_field = potential_field(field, probe)
        rows.append({
            "seed": seed,
            "weight": weight,
            "v": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
            "e": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
            "mean": held_mean_value(field),
            "harmonic": harmonic_ratio(field),
        })
        print("RESULT", rows[-1])

fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
quantities = [("v", "normalized V RMSE", "Independent potential"), ("e", "normalized E RMSE", "Independent electric field"), ("mean", "RMS mean-value error", "Held-out harmonic identity"), ("harmonic", "|a6 r6| / |a2 r2|", "Hexapole contribution at 0.5 r0")]
all_improve = True
for ax, (key, ylabel, title) in zip(axes.ravel(), quantities):
    for seed in SEEDS:
        baseline = next(row[key] for row in rows if row["seed"] == seed and row["weight"] == 0.0)
        mixed = next(row[key] for row in rows if row["seed"] == seed and row["weight"] == 0.1)
        ax.plot([0, 1], [baseline, mixed], marker="o", label=f"seed {seed}")
        all_improve = all_improve and mixed < baseline
    ax.set_xticks([0, 1], ["Laplacian", "+ 0.1 mean-value"])
    ax.set_yscale("log")
    ax.set(title=title, ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Paired-seed gate: every downward line supports a robust common-loss improvement", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

print("PLOT READOUT every paired V/E/mean-value/hexapole result improves:", all_improve)
if all_improve:
    print("CONCLUSION: lambda=0.1 gives a seed-consistent PIG improvement in all four physical views; it is now justified as an experimental common loss for fixed geometry.")
else:
    print("CONCLUSION: the apparent transfer is not seed-consistent in every physical view; retain the paired plot and do not claim a robust common-loss improvement.")
print("saved", OUTPUT)
