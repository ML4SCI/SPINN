import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn

from spinn.models.backends.decoders import DecoderMLP
from spinn.models.backends.pig import PigBackend
from spinn.models.backends.pixel import PixelBackend


OUTER_RADIUS = 5.0
MOMENT_LOSS = os.environ.get("MOMENT_LOSS", "0") == "1"
BACKEND = os.environ.get("SURROGATE_BACKEND", "pig").lower()
PIXEL_BOUND = float(os.environ.get("PIXEL_BOUND", str(OUTER_RADIUS)))
FIELD_LOSS = os.environ.get("FIELD_LOSS", "0") == "1"
if BACKEND == "pixel":
    OUTPUT = "exp/claude_tree_stress_test/results/29_field_supervised_central_pixel.png" if FIELD_LOSS else ("exp/claude_tree_stress_test/results/28_central_pixel_warm_start.png" if PIXEL_BOUND < OUTER_RADIUS else "exp/claude_tree_stress_test/results/27_moment_supervised_pixel_surrogate.png")
else:
    OUTPUT = "exp/claude_tree_stress_test/results/26_moment_supervised_pig_surrogate.png" if MOMENT_LOSS else "exp/claude_tree_stress_test/results/25_geometry_conditioned_pig_surrogate.png"
TRAIN_RADII = np.array([0.60, 0.64, 0.68, 0.72, 0.76])
TEST_RADII = np.array([0.62, 0.66, 0.70, 0.74])
COUNT = 56
ADAM_STEPS = 500
LBFGS_OUTER = 18
LBFGS_INNER = 15
SEED = 2501


def circle_points(center, radius, count, phase=0.0):
    theta = 2.0 * np.pi * np.arange(count) / count + phase
    return center + radius * np.exp(1j * theta)


def solve_circle(radius):
    distance = 1.0 + radius
    centers = distance * np.array([1.0, -1.0, 1.0j, -1.0j])
    voltages = np.array([0.5, 0.5, -0.5, -0.5])
    boundary = np.concatenate([circle_points(center, radius, 3 * COUNT) for center in centers])
    sources = np.concatenate([circle_points(center, 0.70 * radius, COUNT, 0.013) for center in centers])
    values = np.concatenate([np.full(3 * COUNT, voltage) for voltage in voltages])
    boundary = np.concatenate([boundary, circle_points(0.0, OUTER_RADIUS, 12 * COUNT)])
    sources = np.concatenate([sources, circle_points(0.0, 6.5, 4 * COUNT, 0.007)])
    values = np.concatenate([values, np.zeros(12 * COUNT)])
    matrix = np.column_stack([np.log(np.abs(boundary[:, None] - sources[None, :])), np.ones(len(boundary))])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=1e-12)
    return sources, coefficients


def reference_values(radius, points):
    sources, coefficients = solve_circle(radius)
    z = points[:, 0] + 1j * points[:, 1]
    delta = z[:, None] - sources[None, :]
    potential = np.log(np.abs(delta)) @ coefficients[:-1] + coefficients[-1]
    complex_gradient = np.sum(coefficients[None, :-1] / np.conj(delta), axis=1)
    field = -np.column_stack([complex_gradient.real, complex_gradient.imag])
    return potential, field


class RadiusConditionedPIG(nn.Module):
    def __init__(self):
        super().__init__()
        if BACKEND == "pixel":
            self.spatial = PixelBackend(outer_radius=PIXEL_BOUND, grid_resolution=16, feature_dim=4, num_grids=8, decoder_layers=1, decoder_width=12)
        else:
            self.spatial = PigBackend(outer_radius=OUTER_RADIUS, num_gaussians=80, feature_dim=4, init_sigma=0.10, decoder_layers=1, decoder_width=12)
        self.spatial.decoder = nn.Identity()
        self.heads = nn.ModuleList([DecoderMLP(4, layers=2, width=20) for _ in range(4)])

    def d4_head(self, head, points):
        reflected_x = points * points.new_tensor([-1.0, 1.0])
        reflected_y = points * points.new_tensor([1.0, -1.0])
        swapped = points[:, [1, 0]]
        variants = [points, reflected_x, reflected_y, -points, swapped, swapped * points.new_tensor([-1.0, 1.0]), swapped * points.new_tensor([1.0, -1.0]), -swapped]
        features = self.spatial.features(torch.cat(variants, dim=0))
        chunks = torch.chunk(head(features), 8, dim=0)
        return 0.125 * ((chunks[0] + chunks[1] + chunks[2] + chunks[3]) - (chunks[4] + chunks[5] + chunks[6] + chunks[7]))

    def forward(self, points, radius):
        t = (radius - 0.68) / 0.08
        polynomials = [torch.ones_like(t), t, 2.0 * t**2 - 1.0, 4.0 * t**3 - 3.0 * t]
        return sum(polynomial * self.d4_head(head, points) for polynomial, head in zip(polynomials, self.heads))


rng = np.random.default_rng(SEED)
angle = rng.uniform(0.0, 2.0 * np.pi, 800)
radial = 0.92 * np.sqrt(rng.uniform(0.0, 1.0, 800))
train_points_np = np.column_stack([radial * np.cos(angle), radial * np.sin(angle)])
train_points = torch.tensor(train_points_np, dtype=torch.float32)
train_targets = {radius: torch.tensor(reference_values(radius, train_points_np)[0][:, None], dtype=torch.float32) for radius in TRAIN_RADII}
field_points_np = train_points_np[:256]
field_targets = {radius: torch.tensor(reference_values(radius, field_points_np)[1], dtype=torch.float32) for radius in TRAIN_RADII}
theta = 2.0 * np.pi * np.arange(1024) / 1024
multipole_np = np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta)])
multipole_points = torch.tensor(multipole_np, dtype=torch.float32)
cosine2 = torch.tensor(np.cos(2.0 * theta), dtype=torch.float32)
cosine6 = torch.tensor(np.cos(6.0 * theta), dtype=torch.float32)


def raw_moments_from_tensor(values):
    a2 = 2.0 * torch.mean(values * cosine2) / 0.5**2
    a6 = 2.0 * torch.mean(values * cosine6) / 0.5**6
    return a2, a6


def moments_from_tensor(values):
    a2, a6 = raw_moments_from_tensor(values)
    return 2.0 * a2, a6 / a2


moment_targets = {}
for radius in TRAIN_RADII:
    values = reference_values(radius, multipole_np)[0]
    a2 = 2.0 * np.mean(values * np.cos(2.0 * theta)) / 0.5**2
    a6 = 2.0 * np.mean(values * np.cos(6.0 * theta)) / 0.5**6
    moment_targets[radius] = (torch.tensor(2.0 * a2, dtype=torch.float32), torch.tensor(a6, dtype=torch.float32))

torch.manual_seed(SEED)
model = RadiusConditionedPIG()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trajectory = []


def loss_function():
    loss = 0.0
    moment_loss = 0.0
    for radius in TRAIN_RADII:
        radius_tensor = torch.tensor(radius, dtype=torch.float32)
        prediction = model(train_points, radius_tensor)
        loss = loss + torch.mean((prediction - train_targets[radius]) ** 2)
        if FIELD_LOSS:
            field_points = torch.tensor(field_points_np, dtype=torch.float32, requires_grad=True)
            field_potential = model(field_points, radius_tensor).squeeze(-1)
            predicted_field = -torch.autograd.grad(field_potential.sum(), field_points, create_graph=True)[0]
            loss = loss + 0.1 * torch.mean((predicted_field - field_targets[radius]) ** 2)
        if MOMENT_LOSS:
            a2_prediction, a6_prediction = raw_moments_from_tensor(model(multipole_points, radius_tensor).squeeze(-1))
            eta_target, a6_target = moment_targets[radius]
            moment_loss = moment_loss + (2.0 * a2_prediction - eta_target) ** 2 + ((a6_prediction - a6_target) / (0.5 * eta_target)) ** 2
    return (loss + moment_loss) / len(TRAIN_RADII)


print("HYPOTHESIS:", ("a separately scaled electric-field snapshot loss will suppress the cell-scale oscillations of central-grid PIXEL without damaging moment or shape-derivative accuracy." if FIELD_LOSS else ("restricting PIXEL's grid to the declared central warm-start region will remove grid-band field error while the exact solve remains responsible for the full domain." if PIXEL_BOUND < OUTER_RADIUS else f"the moment-supervised geometry response will transfer from PIG to {BACKEND.upper()} without changing the exact solver or unseen gates.")) if BACKEND == "pixel" else ("separate eta/beta moment losses will repair the shape coordinate and its unseen derivative without sacrificing the full field." if MOMENT_LOSS else "shared PIG Gaussian features with a cubic Chebyshev geometry response can interpolate unseen circular-electrode fields and their beta derivative well enough to warm-start an exact solve."))
for step in range(ADAM_STEPS):
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step + 1 == ADAM_STEPS:
        trajectory.append((step, loss.item(), "Adam"))
        print("Adam", step, loss.item())

calls = 0
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=2 * LBFGS_INNER, history_size=50, line_search_fn="strong_wolfe")
for outer in range(LBFGS_OUTER):
    def closure():
        global calls
        optimizer.zero_grad()
        loss = loss_function()
        loss.backward()
        calls += 1
        return loss

    optimizer.step(closure)
    value = loss_function().item()
    trajectory.append((ADAM_STEPS + calls, value, "LBFGS"))
    print("LBFGS", outer, calls, value)

axis = np.linspace(-0.80, 0.80, 61)
xx, yy = np.meshgrid(axis, axis)
all_probe = np.column_stack([xx.ravel(), yy.ravel()])
probe_np = all_probe[np.hypot(all_probe[:, 0], all_probe[:, 1]) <= 0.80]


def predicted_field(radius, points_np):
    points = torch.tensor(points_np, dtype=torch.float32, requires_grad=True)
    value = model(points, torch.tensor(radius, dtype=torch.float32)).squeeze(-1)
    gradient = torch.autograd.grad(value.sum(), points)[0]
    return value.detach().numpy(), -gradient.detach().numpy()


def multipoles_from_values(values):
    a2 = 2.0 * np.mean(values * np.cos(2.0 * theta)) / 0.5**2
    a6 = 2.0 * np.mean(values * np.cos(6.0 * theta)) / 0.5**6
    return 2.0 * a2, a6 / a2


def predicted_eta_beta_torch(radius):
    values = model(multipole_points, radius).squeeze(-1)
    return moments_from_tensor(values)


evaluation_radii = np.sort(np.concatenate([TRAIN_RADII, TEST_RADII]))
records = {}
for radius in evaluation_radii:
    reference_potential, reference_field = reference_values(radius, probe_np)
    prediction, field = predicted_field(radius, probe_np)
    reference_probe = reference_values(radius, multipole_np)[0]
    exact_eta, exact_beta = multipoles_from_values(reference_probe)
    with torch.no_grad():
        pig_eta, pig_beta = predicted_eta_beta_torch(torch.tensor(radius, dtype=torch.float32))
    p_scale = np.max(np.abs(reference_potential))
    e_scale = np.max(np.linalg.norm(reference_field, axis=1))
    records[radius] = {
        "v": np.sqrt(np.mean(((prediction - reference_potential) / p_scale) ** 2)),
        "e": np.sqrt(np.mean((np.linalg.norm(field - reference_field, axis=1) / e_scale) ** 2)),
        "exact_eta": exact_eta,
        "pig_eta": pig_eta.item(),
        "exact_beta": exact_beta,
        "pig_beta": pig_beta.item(),
    }
    print("radius", radius, "unseen" if radius in TEST_RADII else "train", records[radius])

derivative_records = {}
for radius in TEST_RADII:
    variable = torch.tensor(radius, dtype=torch.float32, requires_grad=True)
    _, pig_beta = predicted_eta_beta_torch(variable)
    pig_derivative = torch.autograd.grad(pig_beta, variable)[0].item()
    h = 0.002
    beta_minus = multipoles_from_values(reference_values(radius - h, multipole_np)[0])[1]
    beta_plus = multipoles_from_values(reference_values(radius + h, multipole_np)[0])[1]
    exact_derivative = (beta_plus - beta_minus) / (2.0 * h)
    derivative_records[radius] = (exact_derivative, pig_derivative)
    print("derivative unseen radius", radius, "exact", exact_derivative, "PIG", pig_derivative)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
axes[0, 0].semilogy([row[0] for row in trajectory], [row[1] for row in trajectory], marker="o", markersize=3)
axes[0, 0].axvline(ADAM_STEPS, color="black", linestyle="--")
axes[0, 0].set(xlabel="optimizer evaluations", ylabel="snapshot potential MSE", title="Fit the geometry-conditioned field family")

ordered = evaluation_radii
axes[0, 1].plot(ordered, [records[r]["exact_beta"] for r in ordered], color="black", marker="o", label="exact")
axes[0, 1].plot(ordered, [records[r]["pig_beta"] for r in ordered], color="tab:red", marker="x", label=f"conditioned {BACKEND.upper()}")
axes[0, 1].scatter(TEST_RADII, [records[r]["pig_beta"] for r in TEST_RADII], facecolors="none", edgecolors="tab:blue", s=80, label="unseen radii")
axes[0, 1].set(xlabel="rod radius/r0", ylabel="beta=a6/a2", title=f"Does {BACKEND.upper()} recover the design coordinate?")
axes[0, 1].legend(fontsize=8)

axes[0, 2].plot(ordered, [records[r]["exact_eta"] for r in ordered], color="black", marker="o", label="exact")
axes[0, 2].plot(ordered, [records[r]["pig_eta"] for r in ordered], color="tab:red", marker="x", label=f"conditioned {BACKEND.upper()}")
axes[0, 2].set(xlabel="rod radius/r0", ylabel="eta", title="Efficiency response")
axes[0, 2].legend(fontsize=8)

positions = np.arange(len(ordered))
axes[1, 0].bar(positions - 0.2, [records[r]["v"] for r in ordered], 0.4, label="V RMSE")
axes[1, 0].bar(positions + 0.2, [records[r]["e"] for r in ordered], 0.4, label="E RMSE")
axes[1, 0].set_xticks(positions, [f"{r:.2f}" + ("*" if r in TEST_RADII else "") for r in ordered])
axes[1, 0].set(yscale="log", xlabel="radius (* unseen)", ylabel="normalized spatial RMSE", title="Independent full-field gate")
axes[1, 0].legend(fontsize=8)

axes[1, 1].bar(np.arange(len(TEST_RADII)) - 0.2, [derivative_records[r][0] for r in TEST_RADII], 0.4, label="exact finite difference")
axes[1, 1].bar(np.arange(len(TEST_RADII)) + 0.2, [derivative_records[r][1] for r in TEST_RADII], 0.4, label=f"{BACKEND.upper()} autograd")
axes[1, 1].set_xticks(np.arange(len(TEST_RADII)), [f"{r:.2f}" for r in TEST_RADII])
axes[1, 1].set(xlabel="unseen rod radius/r0", ylabel="d beta / d radius", title="Shape derivative is the warm-start gate")
axes[1, 1].legend(fontsize=8)

shown_radius = 0.70
reference_potential, _ = reference_values(shown_radius, probe_np)
prediction, _ = predicted_field(shown_radius, probe_np)
difference_grid = np.full(xx.shape, np.nan)
inside = np.hypot(all_probe[:, 0], all_probe[:, 1]) <= 0.80
difference_grid.ravel()[inside] = prediction - reference_potential
limit = np.nanmax(np.abs(difference_grid))
image = axes[1, 2].imshow(difference_grid, origin="lower", extent=[-0.8, 0.8, -0.8, 0.8], cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1, 2].set(aspect="equal", xlabel="x/r0", ylabel="y/r0", title=f"Unseen R=.70: {BACKEND.upper()} minus exact potential")
fig.colorbar(image, ax=axes[1, 2])

fig.suptitle(("Claude-tree stress test 29 — separate electric-field loss regularizes central PIXEL" if FIELD_LOSS else ("Claude-tree stress test 28 — central-grid PIXEL as an exact-solver warm start" if PIXEL_BOUND < OUTER_RADIUS else "Claude-tree stress test 27 — does the moment-supervised geometry response transfer to PIXEL?")) if BACKEND == "pixel" else ("Claude-tree stress test 26 — separate physical moment loss repairs conditioned PIG" if MOMENT_LOSS else "Claude-tree stress test 25 — can conditioned PIG warm-start an exact geometry solve?"), fontsize=14)
fig.savefig(OUTPUT, dpi=180)

print("CONCLUSION: accept this as a warm-start surrogate only if unseen V/E fields, beta/eta, and d beta/dR agree simultaneously. Snapshot and moment supervision do not replace the exact solve or independently prove Laplace physics.")
print("saved", OUTPUT)
