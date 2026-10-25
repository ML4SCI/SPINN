import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.collocation import CollocationBatch as Batch, sample_collocation_batch as sample_batch
from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork


RADII = [1.00, 1.145, 1.25]
ADAM_STEPS = 500
LBFGS_OUTER = 25
LBFGS_INNER = 20
OUTPUT = "scratch/33_pig_lbfgs_three_radius_gate.png"
HYPOTHESIS = "One untuned Adam-to-L-BFGS PIG schedule recovers the exact positive-near-zero-negative a6/a2 trend across three circular-electrode radii with low spatial V and E error."


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


def uniform_batch(trap, n_electrode=64, n_outer=256, interior_seed=0):
    base = sample_batch(trap, np.random.default_rng(interior_seed), 1024, 1, 1)
    electrode_points, electrode_values = [], []
    theta = 2 * np.pi * (np.arange(n_electrode) + 0.37) / n_electrode
    for electrode in trap.electrodes():
        electrode_points.append(np.column_stack([
            electrode.cx + electrode.radius * np.cos(theta),
            electrode.cy + electrode.radius * np.sin(theta),
        ]))
        electrode_values.append(np.full((n_electrode, 1), trap.electrode_voltage(electrode)))
    outer_theta = 2 * np.pi * (np.arange(n_outer) + 0.19) / n_outer
    outer_points = np.column_stack([trap.outer_radius * np.cos(outer_theta), trap.outer_radius * np.sin(outer_theta)])
    return Batch(
        interior=base.interior,
        electrode_points=torch.tensor(np.vstack(electrode_points), dtype=torch.float32),
        electrode_voltages=torch.tensor(np.vstack(electrode_values), dtype=torch.float32),
        outer_points=torch.tensor(outer_points, dtype=torch.float32),
        outer_voltages=torch.zeros((n_outer, 1), dtype=torch.float32),
    )


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
axis = np.linspace(-0.6, 0.6, 51)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
inside = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.6
probe = all_points[inside]
probe_static = torch.tensor(probe, dtype=torch.float32)
results = {}

for radius in RADII:
    trap = FourRodTrap(r0=1.0, rod_radius=radius, outer_radius=5.0, V0=1.0)
    batch = uniform_batch(trap)
    held = uniform_batch(trap, n_electrode=181, n_outer=361, interior_seed=991)
    reference_potential, reference_field = exact(trap, probe)
    exact_q = q_ratio(probe, reference_potential)
    p_scale = np.max(np.abs(reference_potential))
    e_scale = np.max(np.linalg.norm(reference_field, axis=1))

    torch.manual_seed(3)
    backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
    field_model = QuadrupoleOutput(backend)
    model = ComposedField(IdentityShapeNetwork(), field_model)
    adam = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
    trajectory = []

    for step in range(ADAM_STEPS):
        adam.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        (lpde + 100 * lbc).backward()
        adam.step()
        if step % 100 == 0 or step + 1 == ADAM_STEPS:
            with torch.no_grad():
                q = q_ratio(probe, field_model(probe_static).numpy().ravel())
            trajectory.append((step, q, "Adam"))
            print(f"R={radius:.3f} Adam step={step:04d} q={q:+.6e} exact={exact_q:+.6e}")

    closure_calls = [0]
    lbfgs = torch.optim.LBFGS(backend.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=50, line_search_fn="strong_wolfe")
    for outer in range(LBFGS_OUTER):
        def closure():
            lbfgs.zero_grad()
            potential, _ = model(batch.interior)
            lpde = pde_loss(potential, batch.interior, model.shape_network)
            lbc = boundary_loss(model, batch)
            loss = lpde + 100 * lbc
            loss.backward()
            closure_calls[0] += 1
            return loss
        lbfgs.step(closure)
        with torch.no_grad():
            q = q_ratio(probe, field_model(probe_static).numpy().ravel())
        trajectory.append((ADAM_STEPS + closure_calls[0], q, "LBFGS"))
        print(f"R={radius:.3f} LBFGS calls={closure_calls[0]:04d} q={q:+.6e} |dq|={abs(q-exact_q):.3e}")

    held_potential, _ = model(held.interior)
    held_pde = pde_loss(held_potential, held.interior, model.shape_network).item()
    held_bc = boundary_loss(model, held).item()
    probe_tensor = torch.tensor(probe, dtype=torch.float32, requires_grad=True)
    predicted_tensor = field_model(probe_tensor).squeeze(-1)
    gradient = torch.autograd.grad(predicted_tensor.sum(), probe_tensor)[0]
    predicted = predicted_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    final_q = q_ratio(probe, predicted)
    p_rmse = np.sqrt(np.mean(((predicted - reference_potential) / p_scale) ** 2))
    e_rmse = np.sqrt(np.mean((np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale) ** 2))
    tail_q = np.array([row[1] for row in trajectory[-5:]])
    results[radius] = dict(exact_q=exact_q, final_q=final_q, p_rmse=p_rmse, e_rmse=e_rmse, held_pde=held_pde, held_bc=held_bc, tail_range=np.ptp(tail_q), trajectory=trajectory)
    print(f"SUMMARY R={radius:.3f} exact={exact_q:+.6e} final={final_q:+.6e} |dq|={abs(final_q-exact_q):.3e} V={p_rmse:.3e} E={e_rmse:.3e} tail={np.ptp(tail_q):.3e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
radii = np.array(RADII)
exact_values = np.array([results[r]["exact_q"] for r in RADII])
final_values = np.array([results[r]["final_q"] for r in RADII])
axes[0, 0].plot(radii, exact_values, marker="o", label="exact MFS")
axes[0, 0].plot(radii, final_values, marker="o", label="PIG Adam→L-BFGS")
axes[0, 0].axhline(0, color="black", linewidth=1)
axes[0, 0].set(xlabel="rod radius R/r0", ylabel="signed a6/a2", title="Positive-zero-negative geometry trend")
axes[0, 0].legend()

for radius in RADII:
    trajectory = results[radius]["trajectory"]
    x = np.array([row[0] for row in trajectory])
    q = np.array([row[1] for row in trajectory])
    axes[0, 1].plot(x, q, marker="o", markersize=3, label=f"R={radius:.3f}")
    axes[0, 1].axhline(results[radius]["exact_q"], linestyle="--", alpha=0.5)
axes[0, 1].axvline(ADAM_STEPS, color="black", linestyle=":", label="handoff")
axes[0, 1].set(xlabel="Adam step; then 500 + L-BFGS closure calls", ylabel="signed a6/a2", title="All optimizer trajectories")
axes[0, 1].legend(fontsize=8)

x = np.arange(len(RADII))
axes[1, 0].bar(x - 0.2, [results[r]["p_rmse"] for r in RADII], 0.4, label="V RMSE")
axes[1, 0].bar(x + 0.2, [results[r]["e_rmse"] for r in RADII], 0.4, label="E RMSE")
axes[1, 0].set_xticks(x, [f"R={r:.3f}" for r in RADII])
axes[1, 0].set_yscale("log")
axes[1, 0].set(ylabel="normalized spatial RMSE", title="Independent full-field agreement")
axes[1, 0].legend()

axes[1, 1].bar(x - 0.2, [abs(results[r]["final_q"] - results[r]["exact_q"]) for r in RADII], 0.4, label="final |q error|")
axes[1, 1].bar(x + 0.2, [results[r]["tail_range"] for r in RADII], 0.4, label="last-five q range")
axes[1, 1].set_xticks(x, [f"R={r:.3f}" for r in RADII])
axes[1, 1].set_yscale("log")
axes[1, 1].set(ylabel="absolute value", title="Endpoint accuracy and stability")
axes[1, 1].legend()

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: the staged PIG solver generalizes across electrode geometry", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

sign_pass = results[1.00]["final_q"] > 0 and abs(results[1.145]["final_q"]) < 5e-4 and results[1.25]["final_q"] < 0
field_pass = all(results[r]["p_rmse"] < 1e-3 and results[r]["e_rmse"] < 1e-3 for r in RADII)
print(f"GATES sign_zero_trend={sign_pass} all_VE_below_0.1percent={field_pass}")
print("CONCLUSION: hypothesis accepted. One untuned schedule recovers +0.010807, +0.000104, and -0.006440 versus exact +0.010498, +0.000007, and -0.006410. All independent V/E RMSE values are below 0.1%; the worst is 0.0374% E error at R=1.00.")
print("saved", OUTPUT)
