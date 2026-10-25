import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from pig_gate_common import exact_field, uniform_batch


MODES = ["none", "one_axis", "full_D4"]
ADAM_STEPS = 500
LBFGS_OUTER = 40
LBFGS_INNER = 20
OUTPUT = "scratch/40_pig_one_axis_physical_gate.png"
HYPOTHESIS = "On a trap with only y-axis mirror symmetry, the matching one-axis PIG improves field and displaced-null accuracy over no symmetry, while full D4 falsely pins the null to the origin."


class SymmetryField(nn.Module):
    def __init__(self, backend, mode):
        super().__init__()
        self.backend = backend
        self.mode = mode

    def y_axis_even(self, points):
        return 0.5 * (self.backend(points) + self.backend(points * points.new_tensor([-1.0, 1.0])))

    def d4_even(self, points):
        return 0.25 * (
            self.backend(points)
            + self.backend(points * points.new_tensor([-1.0, 1.0]))
            + self.backend(points * points.new_tensor([1.0, -1.0]))
            + self.backend(-points)
        )

    def forward(self, points):
        if self.mode == "none":
            return self.backend(points)
        if self.mode == "one_axis":
            return self.y_axis_even(points)
        return 0.5 * (self.d4_even(points) - self.d4_even(points[:, [1, 0]]))


def polynomial_fit(points, potential, degree=5):
    terms = [(i, j) for total in range(degree + 1) for i in range(total + 1) for j in [total - i]]
    matrix = np.column_stack([points[:, 0] ** i * points[:, 1] ** j for i, j in terms])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    return terms, coefficients


def gradient_hessian(terms, coefficients, point):
    x, y = point
    gx = gy = hxx = hxy = hyy = 0.0
    for (i, j), coefficient in zip(terms, coefficients):
        if i >= 1:
            gx += coefficient * i * x ** (i - 1) * y**j
        if j >= 1:
            gy += coefficient * j * x**i * y ** (j - 1)
        if i >= 2:
            hxx += coefficient * i * (i - 1) * x ** (i - 2) * y**j
        if i >= 1 and j >= 1:
            hxy += coefficient * i * j * x ** (i - 1) * y ** (j - 1)
        if j >= 2:
            hyy += coefficient * j * (j - 1) * x**i * y ** (j - 2)
    return np.array([gx, gy]), np.array([[hxx, hxy], [hxy, hyy]])


def null_and_axis(points, potential):
    terms, coefficients = polynomial_fit(points, potential)
    null = np.zeros(2)
    for _ in range(12):
        gradient, hessian = gradient_hessian(terms, coefficients, null)
        step = np.linalg.solve(hessian, gradient)
        null = null - step
        if np.linalg.norm(step) < 1e-12:
            break
    _, hessian = gradient_hessian(terms, coefficients, null)
    values, vectors = np.linalg.eigh(hessian)
    vector = vectors[:, np.argmax(np.abs(values))]
    angle = np.degrees(np.arctan2(vector[1], vector[0])) % 180
    return null, angle


def angle_error(first, second):
    return abs((first - second + 45) % 90 - 45)


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(
    r0=1.0,
    rod_radius=1.145,
    outer_radius=5.0,
    V0=1.0,
    center_perturb=(0.0, 0.0, 0.08, -0.04),
    radius_perturb=(0.0, 0.0, 0.05, -0.03),
)
batch = uniform_batch(trap, n_interior=1024, n_electrode=64, n_outer=256, interior_seed=0)
axis = np.linspace(-0.65, 0.65, 61)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
keep = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.65
probe_np = all_points[keep]
probe_static = torch.tensor(probe_np, dtype=torch.float32)
reference_potential, reference_field = exact_field(trap, probe_np)
p_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))
exact_null, exact_angle = null_and_axis(probe_np, reference_potential)

mirror_x_np = probe_np * np.array([-1.0, 1.0])
mirror_y_np = probe_np * np.array([1.0, -1.0])
reference_mirror_x = exact_field(trap, mirror_x_np)[0]
reference_mirror_y = exact_field(trap, mirror_y_np)[0]
exact_y_axis_residual = np.sqrt(np.mean((reference_potential - reference_mirror_x) ** 2)) / p_scale
exact_x_axis_residual = np.sqrt(np.mean((reference_potential - reference_mirror_y) ** 2)) / p_scale
print(f"EXACT null=({exact_null[0]:+.5f},{exact_null[1]:+.5f}) angle={exact_angle:.3f} y_axis_residual={exact_y_axis_residual:.3e} x_axis_residual={exact_x_axis_residual:.3e}")

results = {}
for mode in MODES:
    torch.manual_seed(3)
    backend = PigBackend(outer_radius=5.0, num_gaussians=100, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
    field_model = SymmetryField(backend, mode)
    model = ComposedField(IdentityShapeNetwork(), field_model)
    adam = torch.optim.Adam(backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4))
    history = []
    for step in range(ADAM_STEPS):
        adam.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        (lpde + 100 * lbc).backward()
        adam.step()
        if step % 100 == 0 or step + 1 == ADAM_STEPS:
            history.append((step, lpde.item(), lbc.item()))
            print(f"{mode:>8s} Adam step={step:04d} PDE={lpde.item():.3e} BC={lbc.item():.3e}")
    closure_calls = [0]
    lbfgs = torch.optim.LBFGS(backend.parameters(), lr=0.5, max_iter=LBFGS_INNER, max_eval=LBFGS_INNER * 2, history_size=50, line_search_fn="strong_wolfe")
    for _ in range(LBFGS_OUTER):
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
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        history.append((ADAM_STEPS + closure_calls[0], lpde.item(), lbc.item()))
        print(f"{mode:>8s} LBFGS calls={closure_calls[0]:04d} PDE={lpde.item():.3e} BC={lbc.item():.3e}")

    probe = probe_static.clone().requires_grad_(True)
    prediction_tensor = field_model(probe).squeeze(-1)
    gradient = torch.autograd.grad(prediction_tensor.sum(), probe)[0]
    prediction = prediction_tensor.detach().numpy()
    predicted_field = -gradient.detach().numpy()
    p_rmse = np.sqrt(np.mean(((prediction - reference_potential) / p_scale) ** 2))
    e_rmse = np.sqrt(np.mean((np.linalg.norm(predicted_field - reference_field, axis=1) / e_scale) ** 2))
    null, angle = null_and_axis(probe_np, prediction)
    with torch.no_grad():
        mirrored_x = field_model(torch.tensor(mirror_x_np, dtype=torch.float32)).numpy().ravel()
        mirrored_y = field_model(torch.tensor(mirror_y_np, dtype=torch.float32)).numpy().ravel()
    y_axis_residual = np.sqrt(np.mean((prediction - mirrored_x) ** 2)) / max(np.max(np.abs(prediction)), 1e-12)
    x_axis_residual = np.sqrt(np.mean((prediction - mirrored_y) ** 2)) / max(np.max(np.abs(prediction)), 1e-12)
    results[mode] = dict(prediction=prediction, p_rmse=p_rmse, e_rmse=e_rmse, null=null, null_error=np.linalg.norm(null-exact_null), angle=angle, angle_error=angle_error(angle, exact_angle), y_axis_residual=y_axis_residual, x_axis_residual=x_axis_residual, history=np.array(history))
    print(f"SUMMARY {mode} V={p_rmse:.3e} E={e_rmse:.3e} null=({null[0]:+.5f},{null[1]:+.5f}) nullErr={results[mode]['null_error']:.3e} angle={angle:.2f} yMirror={y_axis_residual:.3e} xMirror={x_axis_residual:.3e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
reference_map = np.full(gx.shape, np.nan)
reference_map.ravel()[keep] = reference_potential
image = axes[0, 0].imshow(reference_map, origin="lower", extent=(-0.65, 0.65, -0.65, 0.65), cmap="coolwarm")
axes[0, 0].scatter(*exact_null, marker="x", color="black", s=90, label="exact electrical null")
for mode, result in results.items():
    axes[0, 0].scatter(*result["null"], s=55, label=mode)
axes[0, 0].set(xlabel="x/r0", ylabel="y/r0", title="Exact asymmetric potential and predicted nulls")
axes[0, 0].legend(fontsize=8)
fig.colorbar(image, ax=axes[0, 0], label="V/V0")

x = np.arange(len(MODES))
axes[0, 1].bar(x - 0.2, [results[m]["p_rmse"] for m in MODES], 0.4, label="V RMSE")
axes[0, 1].bar(x + 0.2, [results[m]["e_rmse"] for m in MODES], 0.4, label="E RMSE")
axes[0, 1].set_xticks(x, MODES)
axes[0, 1].set_yscale("log")
axes[0, 1].set(ylabel="normalized spatial RMSE", title="Field accuracy under symmetry choices")
axes[0, 1].legend()

axes[1, 0].bar(x, [results[m]["null_error"] for m in MODES])
axes[1, 0].axhline(np.linalg.norm(exact_null), color="grey", linestyle="--", label="error if null forced to origin")
axes[1, 0].set_xticks(x, MODES)
axes[1, 0].set_yscale("log")
axes[1, 0].set(ylabel="electrical-null position error / r0", title=f"Exact null = ({exact_null[0]:+.3f}, {exact_null[1]:+.3f})")
axes[1, 0].legend(fontsize=8)

width = 0.35
axes[1, 1].bar(x - width / 2, [results[m]["y_axis_residual"] for m in MODES], width, label="required y-axis mirror")
axes[1, 1].bar(x + width / 2, [results[m]["x_axis_residual"] for m in MODES], width, label="broken x-axis mirror")
axes[1, 1].axhline(exact_x_axis_residual, color="black", linestyle="--", label="exact broken-mirror residual")
axes[1, 1].set_xticks(x, MODES)
axes[1, 1].set_yscale("log")
axes[1, 1].set(ylabel="normalized reflection residual", title="Does the toggle impose only true symmetry?")
axes[1, 1].legend(fontsize=8)

for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: one-axis symmetry helps only when it matches electrode physics", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

one_axis_better_field = results["one_axis"]["p_rmse"] < results["none"]["p_rmse"] and results["one_axis"]["e_rmse"] < results["none"]["e_rmse"]
one_axis_better_null = results["one_axis"]["null_error"] < results["none"]["null_error"]
d4_false_origin = np.linalg.norm(results["full_D4"]["null"]) < 1e-3 and np.linalg.norm(exact_null) > 1e-2
print(f"GATES one_axis_field_better={one_axis_better_field} one_axis_null_better={one_axis_better_null} full_D4_false_origin={d4_false_origin}")
print("CONCLUSION: accept only if matching one-axis improves field and null over none, and incompatible D4 demonstrably erases the physical null shift.")
print("saved", OUTPUT)
