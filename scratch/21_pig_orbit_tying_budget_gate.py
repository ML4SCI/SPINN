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


STEPS = 300
INTERIOR = 512
ELECTRODE_POINTS = 32
OUTER_POINTS = 128
OUTPUT = "scratch/21_pig_orbit_tying_budget_gate.png"
HYPOTHESIS = "Fundamental-wedge Gaussian orbits will improve PIG efficiency at matched evaluated cost, while mean normalization prevents gradient inflation."


class QuadrupoleProjection(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def even_xy(self, points):
        return 0.25 * (
            self.backend(points)
            + self.backend(points * points.new_tensor([-1.0, 1.0]))
            + self.backend(points * points.new_tensor([1.0, -1.0]))
            + self.backend(-points)
        )

    def forward(self, points):
        return 0.5 * (self.even_xy(points) - self.even_xy(points[:, [1, 0]]))


class SwapProjection(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward(self, points):
        return 0.5 * (self.backend(points) - self.backend(points[:, [1, 0]]))


class OrbitPig(nn.Module):
    kind = "pig"

    def __init__(self, outer_radius, stored_gaussians, feature_dim=4, init_sigma=0.10):
        super().__init__()
        self.outer_radius = outer_radius
        self.stored_gaussians = stored_gaussians
        self.feature_dim = feature_dim
        initial_major = torch.rand(feature_dim, stored_gaussians).clamp(0.02, 0.98)
        initial_fraction = torch.rand(feature_dim, stored_gaussians).clamp(0.02, 0.98)
        raw_major = torch.logit(initial_major)
        raw_fraction = torch.logit(initial_fraction)
        self.raw_means = nn.Parameter(torch.stack([raw_major, raw_fraction], dim=-1))
        initial_softplus = torch.log(torch.expm1(torch.tensor(init_sigma - 0.01)))
        self.raw_sigmas = nn.Parameter(torch.full((feature_dim, stored_gaussians, 2), initial_softplus))
        self.weights = nn.Parameter(torch.normal(0.0, 0.01, (feature_dim, stored_gaussians)))
        self.decoder = DecoderMLP(feature_dim, layers=2, width=16)
        matrices = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[-1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, -1.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, -1.0], [1.0, 0.0]],
                [[0.0, 1.0], [-1.0, 0.0]],
                [[0.0, -1.0], [-1.0, 0.0]],
            ]
        )
        self.register_buffer("matrices", matrices)
        self.register_buffer("signs", torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]))

    def canonical_means(self):
        major = torch.sigmoid(self.raw_means[..., 0])
        minor = major * torch.sigmoid(self.raw_means[..., 1])
        return torch.stack([major, minor], dim=-1)

    def sigmas(self):
        return torch.nn.functional.softplus(self.raw_sigmas) + 0.01

    def pig_coords(self, points):
        return points / self.outer_radius

    def features(self, points):
        coords = self.pig_coords(points)
        means = self.canonical_means()
        sigmas = self.sigmas()
        orbit_features = []
        for matrix, sign in zip(self.matrices, self.signs):
            coordinates_in_canonical_frame = coords @ matrix
            diff = coordinates_in_canonical_frame.unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
            exponent = torch.sum((diff / sigmas.unsqueeze(1)) ** 2, dim=-1)
            gaussians = torch.exp(torch.clamp(-0.5 * exponent, min=-60.0, max=0.0))
            orbit_features.append(sign * (gaussians * self.weights.unsqueeze(1)).sum(dim=-1).T)
        return torch.stack(orbit_features).mean(dim=0)

    def forward(self, points):
        return self.decoder(self.features(points))

    def generated_centers_physical(self):
        with torch.no_grad():
            means = self.canonical_means()
            generated = [means @ matrix.T for matrix in self.matrices]
            return (torch.stack(generated, dim=2) * self.outer_radius).reshape(-1, 2).cpu().numpy()

    def stored_centers_physical(self):
        with torch.no_grad():
            return (self.canonical_means() * self.outer_radius).reshape(-1, 2).cpu().numpy()

    def param_groups(self):
        return [
            {"params": list(self.decoder.parameters()) + [self.weights], "lr": 1.0e-3},
            {"params": [self.raw_means], "lr": 1.0e-4},
            {"params": [self.raw_sigmas], "lr": 1.0e-4},
        ]


def circle_points(cx, cy, radius, n, phase=0.0):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])


def kernel(points, sources):
    return np.log(np.linalg.norm(points[:, None, :] - sources[None, :, :], axis=2))


def reference(trap, points):
    sources, boundary, values = [], [], []
    for electrode in trap.electrodes():
        sources.append(circle_points(electrode.cx, electrode.cy, 0.62 * electrode.radius, 32, 0.017))
        boundary.append(circle_points(electrode.cx, electrode.cy, electrode.radius, 72))
        values.append(np.full(72, trap.electrode_voltage(electrode)))
    sources.append(circle_points(0, 0, 1.30 * trap.outer_radius, 128, 0.011))
    boundary.append(circle_points(0, 0, trap.outer_radius, 256))
    values.append(np.zeros(256))
    sources = np.vstack(sources)
    boundary = np.vstack(boundary)
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
rng = np.random.default_rng(7)
batch = sample_batch(trap, rng, INTERIOR, ELECTRODE_POINTS, OUTER_POINTS)
axis = np.linspace(-0.60, 0.60, 41)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.60
probe = all_points[inside]
reference_potential, reference_field = reference(trap, probe)
q_reference = q_ratio(probe, reference_potential)
potential_scale = np.max(np.abs(reference_potential))
field_scale = np.max(np.linalg.norm(reference_field, axis=1))

specifications = [
    ("output-only K=100", "output", 100),
    ("orbit K=50, matched eval", "orbit", 50),
    ("orbit K=100, matched stored", "orbit", 100),
]
results = []
for label, architecture, count in specifications:
    torch.manual_seed(5)
    if architecture == "output":
        backend = PigBackend(outer_radius=5.0, num_gaussians=count, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
        model = QuadrupoleProjection(backend)
        optimizer = torch.optim.Adam(backend.param_groups(1.0e-3, lr_centers=1.0e-4, lr_covariances=1.0e-4))
        stored_start = np.column_stack([backend.gaussian_diagnostics()["mu_x"], backend.gaussian_diagnostics()["mu_y"]])
    else:
        backend = OrbitPig(5.0, count)
        model = SwapProjection(backend)
        optimizer = torch.optim.Adam(backend.param_groups())
        stored_start = backend.stored_centers_physical()
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
            print(f"{label:26s} step={step:03d} PDE={lpde.detach().item():.3e} BC={lbc.detach().item():.3e}")

    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    potential_tensor = model(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(potential_tensor.sum(), probe_tensor)[0]
    prediction = potential_tensor.detach().numpy()
    field = -gradient.detach().numpy()
    if architecture == "output":
        diagnostics = backend.gaussian_diagnostics()
        stored_end = np.column_stack([diagnostics["mu_x"], diagnostics["mu_y"]])
        generated_end = stored_end
        evaluated = count * 8
    else:
        stored_end = backend.stored_centers_physical()
        generated_end = backend.generated_centers_physical()
        evaluated = count * 16
    results.append(
        {
            "label": label,
            "history": np.array(history),
            "potential_error": (prediction - reference_potential) / potential_scale,
            "field_error": np.linalg.norm(field - reference_field, axis=1) / field_scale,
            "q": q_ratio(probe, prediction),
            "stored_start": stored_start,
            "stored_end": stored_end,
            "generated_end": generated_end,
            "stored": count,
            "evaluated": evaluated,
        }
    )

fig, axes = plt.subplots(3, 4, figsize=(17, 13), constrained_layout=True)
potential_limit = max(np.max(np.abs(row["potential_error"])) for row in results)
field_limit = max(np.max(row["field_error"]) for row in results)
for index, row in enumerate(results):
    potential_map = np.full(gx.shape, np.nan)
    potential_map.ravel()[inside] = row["potential_error"]
    image_p = axes[index, 0].imshow(potential_map, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="coolwarm", vmin=-potential_limit, vmax=potential_limit)
    axes[index, 0].set(title=f"{row['label']}: V error", xlabel="x/r0", ylabel="y/r0")
    field_map = np.full(gx.shape, np.nan)
    field_map.ravel()[inside] = row["field_error"]
    image_e = axes[index, 1].imshow(field_map, origin="lower", extent=(-0.6, 0.6, -0.6, 0.6), cmap="magma", vmin=0, vmax=field_limit)
    axes[index, 1].set(title=f"{row['label']}: |E error|", xlabel="x/r0", ylabel="y/r0")
    axes[index, 2].scatter(row["generated_end"][:, 0], row["generated_end"][:, 1], s=2, alpha=0.35, label="generated/evaluated")
    axes[index, 2].scatter(row["stored_end"][:, 0], row["stored_end"][:, 1], s=8, color="black", label="stored")
    axes[index, 2].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", title=f"stored={row['stored']}, effective eval={row['evaluated']}", xlabel="x/r0", ylabel="y/r0")
    movement = row["stored_end"] - row["stored_start"]
    axes[index, 3].quiver(row["stored_start"][:, 0], row["stored_start"][:, 1], movement[:, 0], movement[:, 1], angles="xy", scale_units="xy", scale=1)
    axes[index, 3].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", title="independent Gaussian movement", xlabel="x/r0", ylabel="y/r0")

fig.colorbar(image_p, ax=axes[:, 0], label="normalized potential error")
fig.colorbar(image_e, ax=axes[:, 1], label="normalized field error")
fig.suptitle("Hypothesis: symmetry-tied Gaussian orbits improve physical accuracy per budget", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

print(f"reference q={q_reference:+.6e}")
for row in results:
    print(
        row["label"],
        f"q={row['q']:+.6e}",
        f"V_RMSE={np.sqrt(np.mean(row['potential_error']**2)):.3e}",
        f"E_RMSE={np.sqrt(np.mean(row['field_error']**2)):.3e}",
        f"stored={row['stored']} effective_eval={row['evaluated']}",
    )
print("CONCLUSION: reject this full signed-feature orbit at the tested budget; it worsens spatial errors at matched evaluated and stored counts, and 300 sparse-sample steps are insufficient for any model to pass the harmonic-sign gate.")
print("saved", OUTPUT)
