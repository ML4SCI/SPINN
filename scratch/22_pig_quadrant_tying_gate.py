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
OUTPUT = "scratch/22_pig_quadrant_tying_gate.png"
HYPOTHESIS = "Positive mirror tying in one quadrant, with the negative voltage-exchange sign applied after decoding, will outperform signed pre-decoder orbits at equal compute."


class OutputQuadrupole(nn.Module):
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


class SwapAntisymmetry(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward(self, points):
        return 0.5 * (self.backend(points) - self.backend(points[:, [1, 0]]))


class QuadrantPig(nn.Module):
    kind = "pig"

    def __init__(self):
        super().__init__()
        initial = torch.rand(4, K, 2).clamp(0.02, 0.98)
        self.raw_means = nn.Parameter(torch.logit(initial))
        initial_sigma = torch.log(torch.expm1(torch.tensor(0.09)))
        self.raw_sigmas = nn.Parameter(torch.full((4, K, 2), initial_sigma))
        self.weights = nn.Parameter(torch.normal(0.0, 0.01, (4, K)))
        self.decoder = DecoderMLP(4, layers=2, width=16)
        self.register_buffer(
            "mirrors",
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[-1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, -1.0]],
                    [[-1.0, 0.0], [0.0, -1.0]],
                ]
            ),
        )

    def means(self):
        return torch.sigmoid(self.raw_means)

    def sigmas(self):
        return torch.nn.functional.softplus(self.raw_sigmas) + 0.01

    def features(self, points):
        coords = points / 5.0
        means = self.means()
        sigmas = self.sigmas()
        pieces = []
        for mirror in self.mirrors:
            canonical_coords = coords @ mirror
            diff = canonical_coords.unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
            exponent = torch.sum((diff / sigmas.unsqueeze(1)) ** 2, dim=-1)
            gaussian = torch.exp(torch.clamp(-0.5 * exponent, min=-60.0, max=0.0))
            pieces.append((gaussian * self.weights.unsqueeze(1)).sum(dim=-1).T)
        return torch.stack(pieces).mean(dim=0)

    def forward(self, points):
        return self.decoder(self.features(points))

    def stored_centers(self):
        return (self.means().detach().cpu().numpy().reshape(-1, 2) * 5.0)

    def generated_centers(self):
        means = self.means().detach()
        generated = [means @ mirror.T for mirror in self.mirrors]
        return (torch.stack(generated, dim=2).reshape(-1, 2).cpu().numpy() * 5.0)

    def param_groups(self):
        return [
            {"params": list(self.decoder.parameters()) + [self.weights], "lr": 1.0e-3},
            {"params": [self.raw_means], "lr": 1.0e-4},
            {"params": [self.raw_sigmas], "lr": 1.0e-4},
        ]


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


def exact_solution(trap, points):
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
rng = np.random.default_rng(0)
batch = sample_batch(trap, rng, 1024, 64, 256)
axis = np.linspace(-0.60, 0.60, 51)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.60
probe = all_points[inside]
reference_potential, reference_field = exact_solution(trap, probe)
q_reference = q_ratio(probe, reference_potential)
potential_scale = np.max(np.abs(reference_potential))
field_scale = np.max(np.linalg.norm(reference_field, axis=1))

results = []
for label in ["output-only", "quadrant-tied"]:
    torch.manual_seed(3)
    if label == "output-only":
        backend = PigBackend(outer_radius=5.0, num_gaussians=K, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
        model = OutputQuadrupole(backend)
        optimizer = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
        diagnostics = backend.gaussian_diagnostics()
        stored_start = np.column_stack([diagnostics["mu_x"], diagnostics["mu_y"]])
    else:
        backend = QuadrantPig()
        model = SwapAntisymmetry(backend)
        optimizer = torch.optim.Adam(backend.param_groups())
        stored_start = backend.stored_centers()
    composed = ComposedField(IdentityShapeNetwork(), model)
    history = []
    for step in range(STEPS):
        optimizer.zero_grad()
        potential, _ = composed(batch.interior)
        lpde = pde_loss(potential, batch.interior, composed.shape_network)
        lbc = boundary_loss(composed, batch)
        (lpde + 100.0 * lbc).backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == STEPS:
            history.append((step, lpde.detach().item(), lbc.detach().item()))
        if step % 100 == 0 or step + 1 == STEPS:
            print(f"{label:14s} step={step:03d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e}")
    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    potential_tensor = model(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(potential_tensor.sum(), probe_tensor)[0]
    prediction = potential_tensor.detach().numpy()
    field = -gradient.detach().numpy()
    if label == "output-only":
        diagnostics = backend.gaussian_diagnostics()
        stored_end = np.column_stack([diagnostics["mu_x"], diagnostics["mu_y"]])
        generated_end = stored_end
    else:
        stored_end = backend.stored_centers()
        generated_end = backend.generated_centers()
    results.append(
        {
            "label": label,
            "history": np.array(history),
            "q": q_ratio(probe, prediction),
            "potential_error": (prediction - reference_potential) / potential_scale,
            "field_error": np.linalg.norm(field - reference_field, axis=1) / field_scale,
            "stored_start": stored_start,
            "stored_end": stored_end,
            "generated_end": generated_end,
        }
    )

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
potential_limit = max(np.max(np.abs(row["potential_error"])) for row in results)
field_limit = max(np.max(row["field_error"]) for row in results)
for index, row in enumerate(results):
    pmap = np.full(gx.shape, np.nan)
    pmap.ravel()[inside] = row["potential_error"]
    image_p = axes[index, 0].imshow(pmap, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="coolwarm", vmin=-potential_limit, vmax=potential_limit)
    axes[index, 0].set(title=f"{row['label']}: V error", xlabel="x/r0", ylabel="y/r0")
    emap = np.full(gx.shape, np.nan)
    emap.ravel()[inside] = row["field_error"]
    image_e = axes[index, 1].imshow(emap, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="magma", vmin=0, vmax=field_limit)
    axes[index, 1].set(title=f"{row['label']}: |E error|", xlabel="x/r0", ylabel="y/r0")
    axes[index, 2].scatter(row["generated_end"][:, 0], row["generated_end"][:, 1], s=2, alpha=0.3)
    axes[index, 2].scatter(row["stored_end"][:, 0], row["stored_end"][:, 1], s=7, color="black")
    axes[index, 2].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", title=f"{row['label']}: stored black", xlabel="x/r0", ylabel="y/r0")
    movement = row["stored_end"] - row["stored_start"]
    axes[index, 3].quiver(row["stored_start"][:, 0], row["stored_start"][:, 1], movement[:, 0], movement[:, 1], angles="xy", scale_units="xy", scale=1)
    axes[index, 3].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", title="independent Gaussian movement", xlabel="x/r0", ylabel="y/r0")

fig.colorbar(image_p, ax=axes[:, 0], label="normalized potential error")
fig.colorbar(image_e, ax=axes[:, 1], label="normalized field error")
fig.suptitle("Hypothesis: positive mirror tying belongs before the nonlinear decoder", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print(f"reference q={q_reference:+.6e}")
for row in results:
    print(
        row["label"],
        f"q={row['q']:+.6e}",
        f"V_RMSE={np.sqrt(np.mean(row['potential_error']**2)):.3e}",
        f"E_RMSE={np.sqrt(np.mean(row['field_error']**2)):.3e}",
    )
print("CONCLUSION: reject quadrant tying with mean normalization at equal compute; it converges more slowly, predicts the wrong harmonic sign, and increases both potential and electric-field error.")
print("saved", OUTPUT)
