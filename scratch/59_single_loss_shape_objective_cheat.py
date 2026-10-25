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


OUTPUT = "scratch/59_single_loss_shape_objective_cheat.png"
HYPOTHESIS = "On a fixed physically anharmonic electrode geometry, putting the desired a6=0 shape objective into the same loss lets the field model report a clean multipole without changing the electrodes, proving that a single joint weighted loss can cheat."
CORE_RADIUS = 0.65
BLEND_RADIUS = 0.90
MEASUREMENT_RADIUS = 0.50
QUALITY_TOLERANCE = 1e-4
ADAM_STEPS = 300
LBFGS_OUTER = 6
LBFGS_INNER = 10


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
        blend = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        return (1.0 - blend) * harmonic + blend * self.backend(points)

    def physical_quality(self):
        a2, a6 = self.coefficients[0], self.coefficients[1]
        return (a6 / a2) * MEASUREMENT_RADIUS**4


def training_batch(trap):
    batch = uniform_batch(trap, n_interior=512, n_electrode=48, n_outer=128, interior_seed=4)
    rng = np.random.default_rng(77)
    theta = rng.uniform(0.0, 2.0 * np.pi, 160)
    radius = np.sqrt(rng.uniform(CORE_RADIUS**2, BLEND_RADIUS**2, 160))
    annulus = torch.tensor(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]), dtype=torch.float32)
    return Batch(torch.cat([batch.interior[:352], annulus]), batch.electrode_points, batch.electrode_voltages, batch.outer_points, batch.outer_voltages)


def train(label, target_quality, batch):
    torch.manual_seed(3)
    backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16, symmetry="d4")
    field = HarmonicCore(backend)
    model = ComposedField(IdentityShapeNetwork(), field)
    groups = backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4)
    groups.append({"params": [field.coefficients], "lr": 1e-3})
    adam = torch.optim.Adam(groups)

    def loss_terms():
        potential, _ = model(batch.interior)
        strong = pde_loss(potential, batch.interior, model.shape_network)
        boundary = boundary_loss(model, batch)
        quality = (field.physical_quality() / QUALITY_TOLERANCE) ** 2 if target_quality else strong.new_tensor(0.0)
        return strong, boundary, quality, strong + 100.0 * boundary + quality

    for step in range(ADAM_STEPS):
        adam.zero_grad()
        strong, boundary, quality, loss = loss_terms()
        loss.backward()
        adam.step()
    lbfgs = torch.optim.LBFGS(field.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=30, line_search_fn="strong_wolfe")
    for outer in range(LBFGS_OUTER):
        def closure():
            lbfgs.zero_grad()
            *_, loss = loss_terms()
            loss.backward()
            return loss
        lbfgs.step(closure)
        strong, boundary, quality, _ = loss_terms()
        print(f"{label:15s} outer={outer} PDE={strong.item():.3e} BC={boundary.item():.3e} quality={quality.item():.3e}")
    return field


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def ring_coefficients(field, trap, radii, reference=False):
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
trap = FourRodTrap(r0=1.0, rod_radius=0.70, outer_radius=5.0, V0=1.0)
batch = training_batch(trap)
models = {
    "physics only": train("physics only", False, batch),
    "single joint loss": train("single joint loss", True, batch),
}

axis = np.linspace(-0.7, 0.7, 61)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
reference_potential, reference_field = exact_field(trap, probe)
v_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))
radii = np.linspace(0.12, 0.62, 11)
reference_a2, reference_a6 = ring_coefficients(None, trap, radii, reference=True)
reference_quality = abs(np.mean(reference_a6) / np.mean(reference_a2)) * MEASUREMENT_RADIUS**4
results = {}
for label, field in models.items():
    potential, electric_field = potential_field(field, probe)
    electrode_prediction = field(batch.electrode_points).detach().numpy().ravel()
    electrode_target = batch.electrode_voltages.numpy().ravel()
    outer_prediction = field(batch.outer_points).detach().numpy().ravel()
    a2, a6 = ring_coefficients(field, trap, radii)
    results[label] = {
        "error": (potential - reference_potential).reshape(gx.shape),
        "v": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
        "e": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
        "bc": max(np.max(np.abs(electrode_prediction - electrode_target)), np.max(np.abs(outer_prediction))),
        "a2": a2,
        "a6": a6,
        "quality": abs(field.physical_quality().item()),
    }
    print(f"SUMMARY {label:15s} V={results[label]['v']:.3e} E={results[label]['e']:.3e} max_BC={results[label]['bc']:.3e} reported_quality={results[label]['quality']:.3e}")
print(f"INDEPENDENT MFS physical quality={reference_quality:.3e}")

limit = max(np.max(np.abs(results[label]["error"])) for label in models)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for ax, label in zip(axes[0, :2], models):
    image = ax.imshow(results[label]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"{label} - MFS\nV={results[label]['v']:.2e}, E={results[label]['e']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")

labels = list(models)
x = np.arange(3)
reported = [reference_quality, results["physics only"]["quality"], results["single joint loss"]["quality"]]
axes[0, 2].bar(x, reported, color=["black", "tab:blue", "tab:red"])
axes[0, 2].axhline(QUALITY_TOLERANCE, color="grey", linestyle="--", label="requested tolerance")
axes[0, 2].set_xticks(x, ["MFS physical", "physics only", "single loss"], rotation=12)
axes[0, 2].set(yscale="log", title="Reported quality on fixed electrodes", ylabel="|a6 r6| / |a2 r2|")

for label, color in [("physics only", "tab:blue"), ("single joint loss", "tab:red")]:
    axes[1, 0].plot(radii, results[label]["a2"], marker="o", color=color, label=label)
    axes[1, 1].plot(radii, results[label]["a6"], marker="o", color=color, label=label)
axes[1, 0].plot(radii, reference_a2, color="black", linestyle="--", label="MFS physical")
axes[1, 1].plot(radii, reference_a6, color="black", linestyle="--", label="MFS physical")
axes[1, 0].set(title="Quadrupole remains identifiable", xlabel="measurement radius / r0", ylabel="a2")
axes[1, 1].set(title="Did the fixed electrodes physically lose a6?", xlabel="measurement radius / r0", ylabel="a6")
axes[1, 2].bar(np.arange(2) - 0.18, [results[label]["v"] for label in labels], 0.36, label="V error")
axes[1, 2].bar(np.arange(2) + 0.18, [results[label]["e"] for label in labels], 0.36, label="E error")
axes[1, 2].set_xticks(np.arange(2), labels)
axes[1, 2].set(yscale="log", title="Independent cost of reporting the target", ylabel="normalized MFS error")
for ax in axes.ravel():
    ax.grid(alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
fig.suptitle("Falsification test: a field network must not satisfy an electrode objective by lying", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

cheat = results["single joint loss"]["quality"] < QUALITY_TOLERANCE and reference_quality > 5.0 * QUALITY_TOLERANCE
print("PLOT READOUT single loss reports pass while fixed physical geometry fails:", cheat)
if cheat:
    print("CONCLUSION: a single weighted field-plus-shape loss is invalid here. Solve the field as an inner problem, then evaluate and differentiate the shape objective through the converged solution.")
else:
    print("CONCLUSION: the field model could not hide the fixed geometry's multipole; this particular joint-loss cheat was not observed.")
print("saved", OUTPUT)
