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


OUTPUT = "scratch/50_pixel_c2_physical_gate.png"
HYPOTHESIS = "At identical initialization and training budget, C2 quintic PIXEL interpolation reduces nonphysical Laplacian structure and radial harmonic drift in the fixed four-rod trap compared with the current cosine interpolation."
ADAM_STEPS = 300


class QuinticPixel(PixelBackend):
    @staticmethod
    def _cosine_kernel(weight):
        return 6.0 * weight**5 - 15.0 * weight**4 + 10.0 * weight**3


def train(backend_class, name, trap, batch):
    torch.manual_seed(7)
    backend = backend_class(
        outer_radius=trap.outer_radius,
        grid_resolution=16,
        feature_dim=4,
        num_grids=4,
        decoder_layers=2,
        decoder_width=16,
    )
    field = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field)
    optimizer = torch.optim.Adam(backend.parameters(), lr=2e-3)
    history = []
    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        loss = lpde + 100.0 * lbc
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == ADAM_STEPS:
            history.append((step, lpde.item(), lbc.item()))
            print(f"{name:8s} step={step:03d} PDE={lpde.item():.3e} BC={lbc.item():.3e}")
    return model, field, np.array(history)


def potential_field(field, points):
    coordinates = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    potential = field(coordinates).squeeze(-1)
    gradient = torch.autograd.grad(potential.sum(), coordinates)[0]
    return potential.detach().numpy(), -gradient.detach().numpy()


def laplacian_line(model, x):
    points = torch.tensor(np.column_stack([x, np.full_like(x, 0.137)]), dtype=torch.float32, requires_grad=True)
    potential, _ = model(points)
    return pullback_laplacian(potential, points, model.shape_network).detach().numpy().ravel()


def radial_ratios(field, radii, order=6, samples=360):
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    ratios = []
    for radius in radii:
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        with torch.no_grad():
            potential = field(torch.tensor(points, dtype=torch.float32)).numpy().ravel()
        a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / radius**2
        a6 = 2.0 * np.mean(potential * np.cos(order * theta)) / radius**order
        ratios.append(a6 / a2)
    return np.array(ratios)


def reference_radial_ratios(trap, radii, samples=360):
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    ratios = []
    for radius in radii:
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        potential = exact_field(trap, points)[0]
        a2 = 2.0 * np.mean(potential * np.cos(2.0 * theta)) / radius**2
        a6 = 2.0 * np.mean(potential * np.cos(6.0 * theta)) / radius**6
        ratios.append(a6 / a2)
    return np.array(ratios)


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


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0)
batch = uniform_batch(trap, n_interior=384, n_electrode=48, n_outer=128, interior_seed=4)
experiments = {}
for name, backend_class in [("cosine", PixelBackend), ("quintic", QuinticPixel)]:
    experiments[name] = train(backend_class, name, trap, batch)

axis = np.linspace(-0.7, 0.7, 61)
gx, gy = np.meshgrid(axis, axis)
probe = np.column_stack([gx.ravel(), gy.ravel()])
reference_potential, reference_field = exact_field(trap, probe)
reference_scale = np.max(np.abs(reference_potential))
field_scale = np.max(np.linalg.norm(reference_field, axis=1))

radii = np.linspace(0.12, 0.62, 11)
reference_ratios = reference_radial_ratios(trap, radii)
circle_radii = np.linspace(0.02, 0.18, 9)
line_x = np.linspace(-0.7, 0.7, 1401)
results = {}
for name, (model, field, history) in experiments.items():
    prediction, predicted_field = potential_field(field, probe)
    electrode_prediction = field(batch.electrode_points).detach().numpy().ravel()
    electrode_target = batch.electrode_voltages.numpy().ravel()
    outer_prediction = field(batch.outer_points).detach().numpy().ravel()
    results[name] = {
        "error": (prediction - reference_potential).reshape(gx.shape),
        "v_rmse": np.sqrt(np.mean((prediction - reference_potential) ** 2)) / reference_scale,
        "e_rmse": np.sqrt(np.mean(np.sum((predicted_field - reference_field) ** 2, axis=1))) / field_scale,
        "bc_max": max(np.max(np.abs(electrode_prediction - electrode_target)), np.max(np.abs(outer_prediction))),
        "laplacian": laplacian_line(model, line_x),
        "ratios": radial_ratios(field, radii),
        "mean_value": mean_value_defect(field, circle_radii),
        "history": history,
    }
    print(f"SUMMARY {name:8s} V_RMSE={results[name]['v_rmse']:.3e} E_RMSE={results[name]['e_rmse']:.3e} max_BC_error={results[name]['bc_max']:.3e}")

limit = max(np.max(np.abs(results["cosine"]["error"])), np.max(np.abs(results["quintic"]["error"])))
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
reference_map = reference_potential.reshape(gx.shape)
image = axes[0, 0].imshow(reference_map, origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="coolwarm")
axes[0, 0].set(title="Independent MFS reference", xlabel="x/r0", ylabel="y/r0")
fig.colorbar(image, ax=axes[0, 0], label="V/V0")
for ax, name in zip(axes[0, 1:], ["cosine", "quintic"]):
    image = ax.imshow(results[name]["error"], origin="lower", extent=(-0.7, 0.7, -0.7, 0.7), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(title=f"{name} PIXEL - reference\nV={results[name]['v_rmse']:.2e}, E={results[name]['e_rmse']:.2e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=ax, label="potential error")

for name, color in [("cosine", "tab:blue"), ("quintic", "tab:orange")]:
    axes[1, 0].plot(line_x, results[name]["laplacian"], color=color, linewidth=1.0, label=name)
    axes[1, 1].plot(radii, results[name]["ratios"], marker="o", color=color, label=name)
    axes[1, 2].plot(circle_radii, results[name]["mean_value"], marker="o", color=color, label=name)
axes[1, 0].set(title="Laplacian along y=0.137 r0", xlabel="x/r0", ylabel="model Laplacian")
axes[1, 1].plot(radii, reference_ratios, color="black", linestyle="--", label="MFS reference")
axes[1, 1].set(title="Does fitted a6/a2 stay independent of radius?", xlabel="measurement radius / r0", ylabel="a6/a2")
axes[1, 2].set(title="Reference-free mean-value defect", xlabel="averaging-circle radius / r0", ylabel="RMS |V(center)-mean V(circle)|")
axes[1, 2].set_yscale("log")
for ax in axes[1]:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Physical gate for C2 PIXEL: smoothness only counts if harmonic behavior improves", fontsize=15)
fig.savefig(OUTPUT, dpi=180)

cosine_radial_drift = np.ptp(results["cosine"]["ratios"])
quintic_radial_drift = np.ptp(results["quintic"]["ratios"])
cosine_mv = results["cosine"]["mean_value"][-1]
quintic_mv = results["quintic"]["mean_value"][-1]
print(f"PLOT READOUT radial q drift cosine={cosine_radial_drift:.3e} quintic={quintic_radial_drift:.3e}")
print(f"PLOT READOUT largest-circle mean-value defect cosine={cosine_mv:.3e} quintic={quintic_mv:.3e}")
if quintic_radial_drift < cosine_radial_drift and quintic_mv < cosine_mv:
    print("CONCLUSION: C2 interpolation improves both independent harmonicity plots at equal budget; promote it to an optional PIXEL kernel and test more training seeds next.")
else:
    print("CONCLUSION: C2 smoothness did not improve both physical harmonicity plots at this budget; keep it as a diagnostic candidate and let the plot choose the next mechanism.")
print("saved", OUTPUT)
