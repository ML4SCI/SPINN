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


OUTPUT = "scratch/58_scaled_harmonic_core.png"
HYPOTHESIS = "Normalizing each complete windowed harmonic basis by its global maximum balances multipole gradients without the transition blow-up caused by core-edge scaling."
CORE_RADIUS = 0.65
BLEND_RADIUS = 0.90
ORDERS = [2, 6, 10]
ADAM_STEPS = 300
LBFGS_OUTER = 6
LBFGS_INNER = 10

scale_radius = np.linspace(0.0, BLEND_RADIUS, 20001)
scale_t = np.clip((scale_radius**2 - CORE_RADIUS**2) / (BLEND_RADIUS**2 - CORE_RADIUS**2), 0.0, 1.0)
scale_window = 1.0 - (6.0 * scale_t**5 - 15.0 * scale_t**4 + 10.0 * scale_t**3)
BASIS_SCALES = np.array([np.max(scale_window * scale_radius**order) for order in ORDERS])


class HarmonicCore(nn.Module):
    def __init__(self, backend, scale_mode):
        super().__init__()
        self.backend = backend
        self.scale_mode = scale_mode
        initial_a2 = 0.4
        if scale_mode == "raw a_n":
            initial = initial_a2
        elif scale_mode == "core-edge b_n":
            initial = initial_a2 * CORE_RADIUS**2
        else:
            initial = initial_a2 * BASIS_SCALES[0]
        self.coefficients = nn.Parameter(torch.tensor([initial, 0.0, 0.0], dtype=torch.float32))

    def forward(self, points):
        z = torch.complex(points[:, 0], points[:, 1])
        if self.scale_mode == "raw a_n":
            harmonic = sum(coefficient * torch.real(z**order) for coefficient, order in zip(self.coefficients, ORDERS))[:, None]
        elif self.scale_mode == "core-edge b_n":
            harmonic = sum(coefficient * torch.real((z / CORE_RADIUS) ** order) for coefficient, order in zip(self.coefficients, ORDERS))[:, None]
        else:
            harmonic = sum(coefficient * torch.real(z**order) / scale for coefficient, order, scale in zip(self.coefficients, ORDERS, BASIS_SCALES))[:, None]
        radius2 = torch.sum(points**2, dim=1, keepdim=True)
        t = ((radius2 - CORE_RADIUS**2) / (BLEND_RADIUS**2 - CORE_RADIUS**2)).clamp(0.0, 1.0)
        blend = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
        return (1.0 - blend) * harmonic + blend * self.backend(points)

    def physical_coefficients(self):
        values = self.coefficients.detach().numpy()
        if self.scale_mode == "raw a_n":
            return values
        if self.scale_mode == "core-edge b_n":
            return np.array([value / CORE_RADIUS**order for value, order in zip(values, ORDERS)])
        return values / BASIS_SCALES


def batch_with_annulus(trap):
    batch = uniform_batch(trap, n_interior=512, n_electrode=48, n_outer=128, interior_seed=4)
    rng = np.random.default_rng(77)
    theta = rng.uniform(0.0, 2.0 * np.pi, 160)
    radius = np.sqrt(rng.uniform(CORE_RADIUS**2, BLEND_RADIUS**2, 160))
    annulus = torch.tensor(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]), dtype=torch.float32)
    return Batch(torch.cat([batch.interior[:352], annulus]), batch.electrode_points, batch.electrode_voltages, batch.outer_points, batch.outer_voltages)


def train(label, batch):
    torch.manual_seed(3)
    backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16, symmetry="d4")
    field = HarmonicCore(backend, label)
    model = ComposedField(IdentityShapeNetwork(), field)
    groups = backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4)
    groups.append({"params": [field.coefficients], "lr": 1e-3})
    adam = torch.optim.Adam(groups)
    for step in range(ADAM_STEPS):
        adam.zero_grad()
        potential, _ = model(batch.interior)
        loss = pde_loss(potential, batch.interior, model.shape_network) + 100.0 * boundary_loss(model, batch)
        loss.backward()
        adam.step()
    lbfgs = torch.optim.LBFGS(field.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=30, line_search_fn="strong_wolfe")
    for outer in range(LBFGS_OUTER):
        def closure():
            lbfgs.zero_grad()
            potential, _ = model(batch.interior)
            loss = pde_loss(potential, batch.interior, model.shape_network) + 100.0 * boundary_loss(model, batch)
            loss.backward()
            return loss
        lbfgs.step(closure)
        print(f"{label:13s} LBFGS outer={outer}")
    return field


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def fitted_coefficients(field, trap, radii, reference=False):
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    output = {order: [] for order in ORDERS}
    for radius in radii:
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        if reference:
            potential = exact_field(trap, points)[0]
        else:
            with torch.no_grad():
                potential = field(torch.tensor(points, dtype=torch.float32)).numpy().ravel()
        for order in ORDERS:
            output[order].append(2.0 * np.mean(potential * np.cos(order * theta)) / radius**order)
    return {order: np.array(values) for order, values in output.items()}


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0)
batch = batch_with_annulus(trap)
print("windowed basis scales:", BASIS_SCALES)
models = {
    "raw a_n": train("raw a_n", batch),
    "core-edge b_n": train("core-edge b_n", batch),
    "global b_n": train("global b_n", batch),
}

axis = np.linspace(-0.7, 0.7, 61)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
reference_potential, reference_field = exact_field(trap, probe)
v_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))
radii = np.linspace(0.12, 0.62, 11)
reference_coefficients = fitted_coefficients(None, trap, radii, reference=True)
results = {}
for label, field in models.items():
    potential, electric_field = potential_field(field, probe)
    electrode_prediction = field(batch.electrode_points).detach().numpy().ravel()
    electrode_target = batch.electrode_voltages.numpy().ravel()
    outer_prediction = field(batch.outer_points).detach().numpy().ravel()
    results[label] = {
        "error": (potential - reference_potential).reshape(gx.shape),
        "v": np.sqrt(np.mean((potential - reference_potential) ** 2)) / v_scale,
        "e": np.sqrt(np.mean(np.sum((electric_field - reference_field) ** 2, axis=1))) / e_scale,
        "bc": max(np.max(np.abs(electrode_prediction - electrode_target)), np.max(np.abs(outer_prediction))),
        "fit": fitted_coefficients(field, trap, radii),
        "parameters": field.physical_coefficients(),
    }
    print(f"SUMMARY {label:10s} V={results[label]['v']:.3e} E={results[label]['e']:.3e} max_BC={results[label]['bc']:.3e} physical_a={results[label]['parameters']}")

limit = max(np.max(np.abs(results[label]["error"])) for label in models)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for ax, label in zip(axes[0], models):
    image = ax.imshow(results[label]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"{label} - MFS\nV={results[label]['v']:.2e}, E={results[label]['e']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")

for ax, order in zip(axes[1], ORDERS):
    for label, color in [("raw a_n", "tab:green"), ("core-edge b_n", "tab:red"), ("global b_n", "tab:blue")]:
        ax.plot(radii, results[label]["fit"][order], marker="o", color=color, label=label)
    ax.plot(radii, reference_coefficients[order], color="black", linestyle="--", label="MFS reference")
    ax.set(title=f"Physical a{order} versus fit radius", xlabel="measurement radius / r0", ylabel=f"a{order}")
for ax in axes.ravel():
    ax.grid(alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
fig.suptitle("Natural multipole units: optimize voltage amplitudes, report physical coefficients", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

raw_a6_error = abs(results["raw a_n"]["parameters"][1] - np.mean(reference_coefficients[6]))
edge_a6_error = abs(results["core-edge b_n"]["parameters"][1] - np.mean(reference_coefficients[6]))
global_a6_error = abs(results["global b_n"]["parameters"][1] - np.mean(reference_coefficients[6]))
print(f"PLOT READOUT physical a6 error raw={raw_a6_error:.3e} core-edge={edge_a6_error:.3e} global={global_a6_error:.3e}")
if global_a6_error < raw_a6_error and results["global b_n"]["v"] <= results["raw a_n"]["v"]:
    print("CONCLUSION: globally bounded basis scaling improves the weak multipole and the full field; retain global b_n for the shape-gradient test.")
elif global_a6_error < raw_a6_error:
    print("CONCLUSION: globally bounded scaling improves a6 but trades against full-field accuracy; test blend width before shape gradients.")
else:
    print("CONCLUSION: even globally bounded preconditioning does not fix a6; gradient scale is not the main cause, so investigate objective sensitivity instead.")
print("saved", OUTPUT)
