import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import CollocationBatch as Batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from pig_gate_common import exact_field, uniform_batch


OUTPUT = "scratch/57_pig_harmonic_measurement_core.png"
HYPOTHESIS = "Replacing PIG only inside the measurement region by an exactly harmonic D4 multipole head removes nonphysical radial multipole behavior without degrading the independently checked fixed-trap field."
SEED = 3
ADAM_STEPS = 300
LBFGS_OUTER = 6
LBFGS_INNER = 10
CORE_RADIUS = 0.65
BLEND_RADIUS = 0.90


class HarmonicCore(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.coefficients = nn.Parameter(torch.tensor([0.4, 0.0, 0.0], dtype=torch.float32))

    def forward(self, points):
        z = torch.complex(points[:, 0], points[:, 1])
        harmonic = (
            self.coefficients[0] * torch.real(z**2)
            + self.coefficients[1] * torch.real(z**6)
            + self.coefficients[2] * torch.real(z**10)
        )[:, None]
        radius2 = torch.sum(points**2, dim=1, keepdim=True)
        t = ((radius2 - CORE_RADIUS**2) / (BLEND_RADIUS**2 - CORE_RADIUS**2)).clamp(0.0, 1.0)
        smoothstep = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        core_weight = 1.0 - smoothstep
        return core_weight * harmonic + smoothstep * self.backend(points)


def add_annulus_points(batch):
    rng = np.random.default_rng(77)
    theta = rng.uniform(0.0, 2.0 * np.pi, 160)
    radius = np.sqrt(rng.uniform(CORE_RADIUS**2, BLEND_RADIUS**2, 160))
    annulus = torch.tensor(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]), dtype=torch.float32)
    return Batch(
        interior=torch.cat([batch.interior[:352], annulus], dim=0),
        electrode_points=batch.electrode_points,
        electrode_voltages=batch.electrode_voltages,
        outer_points=batch.outer_points,
        outer_voltages=batch.outer_voltages,
    )


def train(mode, batch):
    torch.manual_seed(SEED)
    backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16, symmetry="d4")
    field = backend if mode == "ordinary PIG" else HarmonicCore(backend)
    model = ComposedField(IdentityShapeNetwork(), field)
    groups = backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4)
    if mode != "ordinary PIG":
        groups.append({"params": [field.coefficients], "lr": 1e-3})
    adam = torch.optim.Adam(groups)
    for step in range(ADAM_STEPS):
        adam.zero_grad()
        potential, _ = model(batch.interior)
        strong = pde_loss(potential, batch.interior, model.shape_network)
        boundary = boundary_loss(model, batch)
        (strong + 100.0 * boundary).backward()
        adam.step()
    lbfgs = torch.optim.LBFGS(field.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=30, line_search_fn="strong_wolfe")
    for outer in range(LBFGS_OUTER):
        def closure():
            lbfgs.zero_grad()
            potential, _ = model(batch.interior)
            strong = pde_loss(potential, batch.interior, model.shape_network)
            boundary = boundary_loss(model, batch)
            loss = strong + 100.0 * boundary
            loss.backward()
            return loss
        lbfgs.step(closure)
        print(f"{mode:13s} LBFGS outer={outer}")
    return model, field


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def held_mean_value(field, radii):
    rng = np.random.default_rng(19)
    angle = rng.uniform(0.0, 2.0 * np.pi, 64)
    center_radius = 0.35 * np.sqrt(rng.uniform(size=64))
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


def harmonic_coefficients(field, trap, radii, reference=False):
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    a2_values, a6_values = [], []
    for radius in radii:
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        if reference:
            potential = exact_field(trap, points)[0]
        else:
            with torch.no_grad():
                potential = field(torch.tensor(points, dtype=torch.float32)).numpy().ravel()
        a2_values.append(2.0 * np.mean(potential * np.cos(2.0 * theta)) / radius**2)
        a6_values.append(2.0 * np.mean(potential * np.cos(6.0 * theta)) / radius**6)
    return np.array(a2_values), np.array(a6_values)


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0)
batch = add_annulus_points(uniform_batch(trap, n_interior=512, n_electrode=48, n_outer=128, interior_seed=4))
models = {mode: train(mode, batch) for mode in ["ordinary PIG", "harmonic core"]}

axis = np.linspace(-0.7, 0.7, 61)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
reference_potential, reference_field = exact_field(trap, probe)
v_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))
mean_radii = np.linspace(0.02, 0.18, 9)
measurement_radii = np.linspace(0.12, 0.62, 11)
reference_a2, reference_a6 = harmonic_coefficients(None, trap, measurement_radii, reference=True)
results = {}
for mode, (model, field) in models.items():
    potential, electric_field = potential_field(field, probe)
    electrode_prediction = field(batch.electrode_points).detach().numpy().ravel()
    electrode_target = batch.electrode_voltages.numpy().ravel()
    outer_prediction = field(batch.outer_points).detach().numpy().ravel()
    a2, a6 = harmonic_coefficients(field, trap, measurement_radii)
    results[mode] = {
        "error": (potential - reference_potential).reshape(gx.shape),
        "v": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
        "e": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
        "bc": max(np.max(np.abs(electrode_prediction - electrode_target)), np.max(np.abs(outer_prediction))),
        "mean": held_mean_value(field, mean_radii),
        "a2": a2,
        "a6": a6,
    }
    print(f"SUMMARY {mode:13s} V={results[mode]['v']:.3e} E={results[mode]['e']:.3e} max_BC={results[mode]['bc']:.3e}")

limit = max(np.max(np.abs(results[mode]["error"])) for mode in models)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for ax, mode in zip(axes[0, :2], models):
    image = ax.imshow(results[mode]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"{mode} - MFS\nV={results[mode]['v']:.2e}, E={results[mode]['e']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")
radius_line = np.linspace(0.0, 1.05, 300)
radius2 = radius_line**2
t = np.clip((radius2 - CORE_RADIUS**2) / (BLEND_RADIUS**2 - CORE_RADIUS**2), 0.0, 1.0)
core_weight = 1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)
axes[0, 2].plot(radius_line, core_weight, color="tab:green")
axes[0, 2].axvline(1.0, color="black", linestyle="--", label="nearest electrode surface")
axes[0, 2].set(title="Where the harmonic head is active", xlabel="radius / r0", ylabel="harmonic-core weight")

for mode, color in [("ordinary PIG", "tab:purple"), ("harmonic core", "tab:green")]:
    axes[1, 0].plot(mean_radii, results[mode]["mean"], marker="o", color=color, label=mode)
    axes[1, 1].plot(measurement_radii, results[mode]["a2"], marker="o", color=color, label=mode)
    axes[1, 2].plot(measurement_radii, results[mode]["a6"], marker="o", color=color, label=mode)
axes[1, 0].set(yscale="log", title="Mean-value identity inside harmonic core", xlabel="circle radius / r0", ylabel="RMS mean-value error")
axes[1, 1].plot(measurement_radii, reference_a2, color="black", linestyle="--", label="MFS reference")
axes[1, 1].set(title="Quadrupole coefficient versus fit radius", xlabel="measurement radius / r0", ylabel="a2")
axes[1, 2].plot(measurement_radii, reference_a6, color="black", linestyle="--", label="MFS reference")
axes[1, 2].set(title="Hexapole coefficient versus fit radius", xlabel="measurement radius / r0", ylabel="a6")
for ax in [axes[0, 2], *axes[1]]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Architecture gate: make the measurement region analytic, leave electrodes to PIG", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

core_exact_mean = results["harmonic core"]["mean"].max() < 1e-6
field_not_worse = results["harmonic core"]["v"] <= results["ordinary PIG"]["v"] and results["harmonic core"]["e"] <= results["ordinary PIG"]["e"]
print(f"PLOT READOUT core mean-value exact={core_exact_mean} field not worse={field_not_worse}")
if core_exact_mean and field_not_worse:
    print("CONCLUSION: the harmonic measurement core removes the central analytic failure without a field penalty; next test its shape gradients before any production integration.")
elif core_exact_mean:
    print("CONCLUSION: the harmonic core fixes central analyticity but harms the independently checked field; adjust only the blend annulus before using it for shape optimization.")
else:
    print("CONCLUSION: the supposed harmonic core fails its own mean-value gate; reject the implementation.")
print("saved", OUTPUT)
