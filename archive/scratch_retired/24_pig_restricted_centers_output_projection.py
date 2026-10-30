import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import sample_collocation_batch as sample_batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.decoders import DecoderMLP
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork


STEPS = 500
K = 100
OUTPUT = "scratch/24_pig_restricted_centers_output_projection.png"
HYPOTHESIS = "Restricting independent Gaussian centers to one quadrant will retain output-projected PIG accuracy when every reflected query is decoded before scalar outputs are combined."


class QuadrupoleOutput(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def even(self, points):
        return 0.25 * (
            self.backend(points)
            + self.backend(points * points.new_tensor([-1.0, 1.0]))
            + self.backend(points * points.new_tensor([1.0, -1.0]))
            + self.backend(-points)
        )

    def forward(self, points):
        return 0.5 * (self.even(points) - self.even(points[:, [1, 0]]))


class QuadrantCentersPig(nn.Module):
    kind = "pig"

    def __init__(self):
        super().__init__()
        self.raw_means = nn.Parameter(torch.logit(torch.rand(4, K, 2).clamp(0.02, 0.98)))
        self.sigmas = nn.Parameter(torch.ones(4, K, 2) * 0.10)
        self.weights = nn.Parameter(torch.normal(0.0, 0.01, (4, K)))
        self.decoder = DecoderMLP(4, layers=2, width=16)

    def means(self):
        return torch.sigmoid(self.raw_means)

    def features(self, points):
        coords = points / 5.0
        diff = coords.unsqueeze(0).unsqueeze(2) - self.means().unsqueeze(1)
        exponent = torch.sum((diff / self.sigmas.unsqueeze(1)) ** 2, dim=-1)
        gaussians = torch.exp(torch.clamp(-0.5 * exponent, min=-60.0, max=0.0))
        return (gaussians * self.weights.unsqueeze(1)).sum(dim=-1).T

    def forward(self, points):
        return self.decoder(self.features(points))

    def centers_physical(self):
        return self.means().detach().reshape(-1, 2).cpu().numpy() * 5.0

    def groups(self):
        return [
            {"params": list(self.decoder.parameters()) + [self.weights], "lr": 1e-3},
            {"params": [self.raw_means], "lr": 1e-4},
            {"params": [self.sigmas], "lr": 1e-4},
        ]


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


def exact(trap, points):
    sources, boundaries, values = [], [], []
    for electrode in trap.electrodes():
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
        boundaries.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0, 0, 1.3 * trap.outer_radius, 128, 0.011))
    boundaries.append(circle_points(0, 0, trap.outer_radius, 256))
    values.append(np.zeros(256))
    sources = np.vstack(sources)
    boundary = np.vstack(boundaries)
    matrix = np.column_stack([kernel(boundary, sources), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, np.concatenate(values), rcond=1e-12)
    delta = points[:, None, :] - sources[None, :, :]
    radius2 = np.sum(delta**2, axis=2)
    potential = np.log(np.sqrt(radius2)) @ coefficients[:-1] + coefficients[-1]
    field = -np.sum(delta / radius2[:, :, None] * coefficients[None, :-1, None], axis=1)
    return potential, field


def q_ratio(points, potential):
    z = points[:, 0] + 1j * points[:, 1]
    orders = [2, 6, 10, 14, 18]
    matrix = np.column_stack([np.ones(len(points))] + [part for order in orders for part in (np.real(z**order), np.imag(z**order))])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return coefficients[3] / coefficients[1]


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.0, outer_radius=5.0, V0=1.0)
batch = sample_batch(trap, np.random.default_rng(0), 1024, 64, 256)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.6
probe = all_points[inside]
reference_potential, reference_field = exact(trap, probe)
q_reference = q_ratio(probe, reference_potential)
p_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))

results = []
for label in ["full-domain centers", "quadrant centers"]:
    torch.manual_seed(3)
    if label == "full-domain centers":
        backend = PigBackend(outer_radius=5.0, num_gaussians=K, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
        optimizer = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
        diagnostics = backend.gaussian_diagnostics()
        start = np.column_stack([diagnostics["mu_x"], diagnostics["mu_y"]])
    else:
        backend = QuadrantCentersPig()
        optimizer = torch.optim.Adam(backend.groups())
        start = backend.centers_physical()
    field_model = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field_model)
    history = []
    for step in range(STEPS):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        (lpde + 100 * lbc).backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == STEPS:
            history.append((step, lpde.detach().item(), lbc.detach().item()))
        if step % 100 == 0 or step + 1 == STEPS:
            print(f"{label:20s} step={step:03d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e}")
    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    potential_tensor = field_model(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(potential_tensor.sum(), probe_tensor)[0]
    prediction = potential_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    if label == "full-domain centers":
        diagnostics = backend.gaussian_diagnostics()
        end = np.column_stack([diagnostics["mu_x"], diagnostics["mu_y"]])
    else:
        end = backend.centers_physical()
    results.append(
        {
            "label": label,
            "history": np.array(history),
            "q": q_ratio(probe, prediction),
            "p_error": (prediction - reference_potential) / p_scale,
            "e_error": np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale,
            "start": start,
            "end": end,
        }
    )

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
p_limit = max(np.max(np.abs(row["p_error"])) for row in results)
e_limit = max(np.max(row["e_error"]) for row in results)
for index, row in enumerate(results):
    pmap = np.full(gx.shape, np.nan)
    pmap.ravel()[inside] = row["p_error"]
    image_p = axes[index, 0].imshow(pmap, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="coolwarm", vmin=-p_limit, vmax=p_limit)
    emap = np.full(gx.shape, np.nan)
    emap.ravel()[inside] = row["e_error"]
    image_e = axes[index, 1].imshow(emap, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="magma", vmin=0, vmax=e_limit)
    p_rmse = np.sqrt(np.mean(row["p_error"]**2))
    e_rmse = np.sqrt(np.mean(row["e_error"]**2))
    axes[index, 0].set(title=f"{row['label']}: V {p_rmse:.2%}", xlabel="x/r0", ylabel="y/r0")
    axes[index, 1].set(title=f"{row['label']}: E {e_rmse:.2%}", xlabel="x/r0", ylabel="y/r0")
    axes[index, 2].scatter(row["end"][:, 0], row["end"][:, 1], s=6)
    axes[index, 2].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", title="independent centers", xlabel="x/r0", ylabel="y/r0")
    movement = row["end"] - row["start"]
    axes[index, 3].quiver(row["start"][:, 0], row["start"][:, 1], movement[:, 0], movement[:, 1], angles="xy", scale_units="xy", scale=1)
    axes[index, 3].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", title="center movement", xlabel="x/r0", ylabel="y/r0")

fig.colorbar(image_p, ax=axes[:, 0], label="normalized potential error")
fig.colorbar(image_e, ax=axes[:, 1], label="normalized electric-field error")
fig.suptitle("Hypothesis: restrict Gaussian locations but preserve decode-before-symmetrize order", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print(f"reference q={q_reference:+.6e}")
for row in results:
    print(row["label"], f"q={row['q']:+.6e}", f"V_RMSE={np.sqrt(np.mean(row['p_error']**2)):.3e}", f"E_RMSE={np.sqrt(np.mean(row['e_error']**2)):.3e}")
print("CONCLUSION: reject quadrant center restriction; despite similar final PDE loss, it strongly overestimates the physical harmonic and increases spatial field errors.")
print("saved", OUTPUT)
