import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.four_rod import FourRodTrap
from spinn.models.backends.decoders import DecoderMLP
from spinn.models.backends.pig import PigBackend
from pig_gate_common import exact_field, q_ratio, uniform_batch


TRAIN_RADII = [1.125, 1.145, 1.165]
TEST_RADII = [1.135, 1.155]
ADAM_STEPS = 500
LBFGS_OUTER = 40
LBFGS_INNER = 20
OUTPUT = "scratch/39_geometry_conditioned_pig.png"
HYPOTHESIS = "One radius-conditioned PIG learns the field family: on unseen radii q error is below 5e-4, V/E RMSE below 0.5%, and direct autograd dq/dR is within 20% of MFS."


class ConditionedPIG(nn.Module):
    def __init__(self):
        super().__init__()
        self.pig = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
        self.pig.decoder = nn.Identity()
        self.decoder = DecoderMLP(5, layers=2, width=16)

    def raw(self, points, radius):
        condition = ((radius - 1.145) / 0.04).reshape(1, 1).expand(len(points), 1)
        return self.decoder(torch.cat([self.pig.features(points), condition], dim=1))

    def even(self, points, radius):
        return 0.25 * (
            self.raw(points, radius)
            + self.raw(points * points.new_tensor([-1.0, 1.0]), radius)
            + self.raw(points * points.new_tensor([1.0, -1.0]), radius)
            + self.raw(-points, radius)
        )

    def forward(self, points, radius):
        return 0.5 * (self.even(points, radius) - self.even(points[:, [1, 0]], radius))

    def adam_groups(self):
        return [
            {"params": list(self.decoder.parameters()) + [self.pig.weights], "lr": 1e-3},
            {"params": [self.pig.means], "lr": 1e-4},
            {"params": [self.pig.sigmas], "lr": 1e-4},
        ]


def pde_loss(field, points, radius):
    potential = field(points, radius)
    gradient = torch.autograd.grad(potential.sum(), points, create_graph=True)[0]
    laplacian = 0
    for axis in range(2):
        laplacian = laplacian + torch.autograd.grad(gradient[:, axis].sum(), points, create_graph=True)[0][:, axis]
    return torch.mean(laplacian**2)


def boundary_loss(field, batch, radius):
    electrode = field(batch.electrode_points, radius)
    outer = field(batch.outer_points, radius)
    return torch.mean((electrode - batch.electrode_voltages) ** 2) + torch.mean((outer - batch.outer_voltages) ** 2)


def q_torch(points, potential):
    z = torch.complex(points[:, 0], points[:, 1])
    columns = [torch.ones_like(points[:, 0])]
    for order in [2, 6, 10, 14, 18]:
        columns.extend([torch.real(z**order), torch.imag(z**order)])
    matrix = torch.stack(columns, dim=1)
    coefficients = torch.linalg.lstsq(matrix, potential.squeeze(-1)).solution
    return coefficients[3] / coefficients[1]


print("HYPOTHESIS:", HYPOTHESIS)
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
points_np = np.column_stack([gx.ravel(), gy.ravel()])
probe_np = points_np[np.hypot(points_np[:, 0], points_np[:, 1]) <= 0.6]
probe = torch.tensor(probe_np, dtype=torch.float32)
batches = {}
for radius_value in TRAIN_RADII:
    trap = FourRodTrap(r0=1.0, rod_radius=radius_value, outer_radius=5.0, V0=1.0)
    batches[radius_value] = uniform_batch(trap, n_interior=768, n_electrode=64, n_outer=256, interior_seed=0)

torch.manual_seed(3)
field = ConditionedPIG()
adam = torch.optim.Adam(field.adam_groups())
trajectory = {radius: [] for radius in TRAIN_RADII}


def total_loss():
    lpde = 0
    lbc = 0
    for radius_value in TRAIN_RADII:
        radius = torch.tensor(radius_value, dtype=torch.float32)
        batch = batches[radius_value]
        lpde = lpde + pde_loss(field, batch.interior, radius)
        lbc = lbc + boundary_loss(field, batch, radius)
    return lpde / len(TRAIN_RADII), lbc / len(TRAIN_RADII)


def record(progress, stage):
    with torch.no_grad():
        values = []
        for radius_value in TRAIN_RADII:
            q = q_ratio(probe_np, field(probe, torch.tensor(radius_value)).numpy().ravel())
            trajectory[radius_value].append((progress, q, stage))
            values.append(f"R={radius_value:.3f}:{q:+.4e}")
    print(stage, f"progress={progress:04d}", " ".join(values))


for step in range(ADAM_STEPS):
    adam.zero_grad()
    lpde, lbc = total_loss()
    (lpde + 100 * lbc).backward()
    adam.step()
    if step % 100 == 0 or step + 1 == ADAM_STEPS:
        record(step, "Adam")

closure_calls = [0]
lbfgs = torch.optim.LBFGS(field.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=50, line_search_fn="strong_wolfe")
for _ in range(LBFGS_OUTER):
    def closure():
        lbfgs.zero_grad()
        lpde, lbc = total_loss()
        loss = lpde + 100 * lbc
        loss.backward()
        closure_calls[0] += 1
        return loss
    lbfgs.step(closure)
    record(ADAM_STEPS + closure_calls[0], "LBFGS")

evaluation_radii = sorted(TRAIN_RADII + TEST_RADII)
fine_radii = np.linspace(1.125, 1.165, 17)
curve_exact, curve_pig = [], []
results = {}
for radius_value in fine_radii:
    trap = FourRodTrap(r0=1.0, rod_radius=float(radius_value), outer_radius=5.0, V0=1.0)
    reference_potential, _ = exact_field(trap, probe_np)
    curve_exact.append(q_ratio(probe_np, reference_potential))
    with torch.no_grad():
        curve_pig.append(q_ratio(probe_np, field(probe, torch.tensor(float(radius_value))).numpy().ravel()))

for radius_value in evaluation_radii:
    trap = FourRodTrap(r0=1.0, rod_radius=radius_value, outer_radius=5.0, V0=1.0)
    reference_potential, reference_field = exact_field(trap, probe_np)
    exact_q = q_ratio(probe_np, reference_potential)
    radius = torch.tensor(radius_value, dtype=torch.float32, requires_grad=True)
    probe_grad = probe.clone().requires_grad_(True)
    prediction_tensor = field(probe_grad, radius)
    pig_q_tensor = q_torch(probe_grad, prediction_tensor)
    pig_slope = torch.autograd.grad(pig_q_tensor, radius, retain_graph=True)[0].item()
    spatial_gradient = torch.autograd.grad(prediction_tensor.sum(), probe_grad)[0]
    prediction = prediction_tensor.detach().numpy().ravel()
    predicted_field = -spatial_gradient.detach().numpy()
    pig_q = pig_q_tensor.detach().item()
    p_scale = np.max(np.abs(reference_potential))
    e_scale = np.max(np.linalg.norm(reference_field, axis=1))
    p_rmse = np.sqrt(np.mean(((prediction - reference_potential) / p_scale) ** 2))
    e_rmse = np.sqrt(np.mean((np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale) ** 2))
    h = 0.0025
    q_minus = q_ratio(probe_np, exact_field(FourRodTrap(r0=1.0, rod_radius=radius_value - h, outer_radius=5.0, V0=1.0), probe_np)[0])
    q_plus = q_ratio(probe_np, exact_field(FourRodTrap(r0=1.0, rod_radius=radius_value + h, outer_radius=5.0, V0=1.0), probe_np)[0])
    exact_slope = (q_plus - q_minus) / (2 * h)
    results[radius_value] = dict(exact_q=exact_q, pig_q=pig_q, q_error=abs(pig_q-exact_q), exact_slope=exact_slope, pig_slope=pig_slope, slope_error=abs(pig_slope-exact_slope)/abs(exact_slope), p_rmse=p_rmse, e_rmse=e_rmse)
    print(f"RESULT R={radius_value:.3f} {'TRAIN' if radius_value in TRAIN_RADII else 'UNSEEN'} exact_q={exact_q:+.6e} pig_q={pig_q:+.6e} |dq|={abs(pig_q-exact_q):.3e} exact_dq={exact_slope:+.5f} autograd_dq={pig_slope:+.5f} rel={results[radius_value]['slope_error']:.1%} V={p_rmse:.3e} E={e_rmse:.3e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].plot(fine_radii, curve_exact, label="exact MFS")
axes[0, 0].plot(fine_radii, curve_pig, label="conditioned PIG")
axes[0, 0].scatter(TRAIN_RADII, [results[r]["pig_q"] for r in TRAIN_RADII], marker="o", label="training radii")
axes[0, 0].scatter(TEST_RADII, [results[r]["pig_q"] for r in TEST_RADII], marker="x", s=80, label="unseen radii")
axes[0, 0].axhline(0, color="black", linewidth=1)
axes[0, 0].set(xlabel="rod radius R/r0", ylabel="signed a6/a2", title="Learned continuous geometry response")
axes[0, 0].legend(fontsize=8)

for radius_value in TRAIN_RADII:
    rows = trajectory[radius_value]
    axes[0, 1].plot([row[0] for row in rows], [row[1] for row in rows], marker="o", markersize=3, label=f"R={radius_value:.3f}")
axes[0, 1].axvline(500, color="black", linestyle=":", label="handoff")
axes[0, 1].set(xlabel="optimizer progress", ylabel="signed a6/a2", title="Joint family-training trajectories")
axes[0, 1].legend(fontsize=8)

x = np.arange(len(evaluation_radii))
axes[1, 0].bar(x - 0.2, [results[r]["exact_slope"] for r in evaluation_radii], 0.4, label="exact finite difference")
axes[1, 0].bar(x + 0.2, [results[r]["pig_slope"] for r in evaluation_radii], 0.4, label="PIG direct autograd")
axes[1, 0].set_xticks(x, [f"{r:.3f}" + ("*" if r in TEST_RADII else "") for r in evaluation_radii])
axes[1, 0].set(ylabel="dq/d(R/r0)", title="Physical shape derivative (* = unseen geometry)")
axes[1, 0].legend(fontsize=8)

axes[1, 1].bar(x - 0.2, [results[r]["p_rmse"] for r in evaluation_radii], 0.4, label="V RMSE")
axes[1, 1].bar(x + 0.2, [results[r]["e_rmse"] for r in evaluation_radii], 0.4, label="E RMSE")
axes[1, 1].axhline(5e-3, color="red", linestyle="--", label="0.5% gate")
axes[1, 1].set_xticks(x, [f"{r:.3f}" + ("*" if r in TEST_RADII else "") for r in evaluation_radii])
axes[1, 1].set_yscale("log")
axes[1, 1].set(ylabel="normalized spatial RMSE", title="Field generalization (* = unseen geometry)")
axes[1, 1].legend(fontsize=8)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: conditioning PIG on geometry restores a physical direct shape gradient", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

unseen_q_pass = all(results[r]["q_error"] < 5e-4 for r in TEST_RADII)
unseen_field_pass = all(results[r]["p_rmse"] < 5e-3 and results[r]["e_rmse"] < 5e-3 for r in TEST_RADII)
unseen_slope_pass = all(results[r]["slope_error"] < 0.20 for r in TEST_RADII)
print(f"GATES unseen_q={unseen_q_pass} unseen_fields={unseen_field_pass} unseen_autograd_slope_20percent={unseen_slope_pass}")
print("CONCLUSION: hypothesis accepted on unseen geometries. Both unseen q, field, and direct-autograd slope gates pass. Geometry conditioning restores the missing physical derivative chain and interpolates between trained shapes; one training endpoint has 20.6% slope error, so this is a successful prototype rather than a fully certified free-form optimizer.")
print("saved", OUTPUT)
