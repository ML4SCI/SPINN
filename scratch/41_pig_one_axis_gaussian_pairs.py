import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spinn.geometry.four_rod import FourRodTrap
from spinn.losses.boundary import boundary_loss
from spinn.losses.pde import pde_loss
from spinn.models.backends.decoders import DecoderMLP
from spinn.models.backends.pig import PigBackend
from spinn.models.composed import ComposedField, IdentityShapeNetwork
from pig_gate_common import exact_field, uniform_batch


OUTPUT = "scratch/41_pig_one_axis_gaussian_pairs.png"
HYPOTHESIS = "One-axis Gaussian pairs may compress a matching mirror-symmetric PIG, but should be accepted only if they retain field and shifted-null accuracy at equal generated Gaussian count."


class OutputMirror(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward(self, points):
        reflected = points * points.new_tensor([-1.0, 1.0])
        return 0.5 * (self.backend(points) + self.backend(reflected))


class PairedPig(nn.Module):
    kind = "pig"

    def __init__(self, stored_gaussians):
        super().__init__()
        self.stored_gaussians = stored_gaussians
        self.raw_abs_x = nn.Parameter(torch.logit(torch.rand(4, stored_gaussians).clamp(0.02, 0.98)))
        self.raw_y = nn.Parameter(torch.logit(torch.rand(4, stored_gaussians).clamp(0.02, 0.98)))
        self.sigmas = nn.Parameter(torch.ones(4, stored_gaussians, 2) * 0.10)
        self.weights = nn.Parameter(torch.normal(0.0, 0.01, (4, stored_gaussians)))
        self.decoder = DecoderMLP(4, layers=2, width=16)

    def canonical_means(self):
        x = torch.sigmoid(self.raw_abs_x)
        y = 2.0 * torch.sigmoid(self.raw_y) - 1.0
        return torch.stack([x, y], dim=-1)

    def features(self, points):
        coords = points / 5.0
        means = self.canonical_means()
        reflected = means * means.new_tensor([-1.0, 1.0])
        pair_sum = 0.0
        for centers in (means, reflected):
            diff = coords.unsqueeze(0).unsqueeze(2) - centers.unsqueeze(1)
            exponent = torch.sum((diff / self.sigmas.unsqueeze(1)) ** 2, dim=-1)
            pair_sum = pair_sum + torch.exp(torch.clamp(-0.5 * exponent, min=-60.0, max=0.0))
        return (0.5 * pair_sum * self.weights.unsqueeze(1)).sum(dim=-1).T

    def forward(self, points):
        return self.decoder(self.features(points))

    def param_groups(self):
        return [
            {"params": list(self.decoder.parameters()) + [self.weights], "lr": 1e-3},
            {"params": [self.raw_abs_x, self.raw_y], "lr": 1e-4},
            {"params": [self.sigmas], "lr": 1e-4},
        ]

    def centers(self):
        means = self.canonical_means().detach()
        both = torch.stack([means, means * means.new_tensor([-1.0, 1.0])], dim=2)
        return (5.0 * both).reshape(-1, 2).cpu().numpy()


def fit_null(points, potential):
    x, y = points[:, 0], points[:, 1]
    terms = [(i, total - i) for total in range(6) for i in range(total + 1)]
    matrix = np.column_stack([x**i * y**j for i, j in terms])
    coefficients, *_ = np.linalg.lstsq(matrix, potential, rcond=None)
    point = np.zeros(2)
    for _ in range(12):
        px, py = point
        gradient = np.zeros(2)
        hessian = np.zeros((2, 2))
        for (i, j), coefficient in zip(terms, coefficients):
            if i:
                gradient[0] += coefficient * i * px ** (i - 1) * py**j
            if j:
                gradient[1] += coefficient * j * px**i * py ** (j - 1)
            if i >= 2:
                hessian[0, 0] += coefficient * i * (i - 1) * px ** (i - 2) * py**j
            if i and j:
                hessian[0, 1] += coefficient * i * j * px ** (i - 1) * py ** (j - 1)
            if j >= 2:
                hessian[1, 1] += coefficient * j * (j - 1) * px**i * py ** (j - 2)
        hessian[1, 0] = hessian[0, 1]
        step = np.linalg.solve(hessian, gradient)
        point -= step
        if np.linalg.norm(step) < 1e-12:
            break
    return point


def train(label, backend, field_model, groups, batch):
    model = ComposedField(IdentityShapeNetwork(), field_model)
    optimizer = torch.optim.Adam(groups)
    history = []
    for step in range(500):
        optimizer.zero_grad()
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        (lpde + 100.0 * lbc).backward()
        optimizer.step()
        if step % 25 == 0 or step == 499:
            history.append((step, lpde.item() + 100.0 * lbc.item()))
    calls = 0
    optimizer = torch.optim.LBFGS(backend.parameters(), lr=0.5, max_iter=20, max_eval=40, history_size=50, line_search_fn="strong_wolfe")
    for _ in range(25):
        def closure():
            nonlocal calls
            optimizer.zero_grad()
            potential, _ = model(batch.interior)
            lpde = pde_loss(potential, batch.interior, model.shape_network)
            lbc = boundary_loss(model, batch)
            loss = lpde + 100.0 * lbc
            loss.backward()
            calls += 1
            return loss
        optimizer.step(closure)
        potential, _ = model(batch.interior)
        lpde = pde_loss(potential, batch.interior, model.shape_network)
        lbc = boundary_loss(model, batch)
        history.append((500 + calls, lpde.item() + 100.0 * lbc.item()))
    print(label, "closure_calls", calls, "final_loss", history[-1][1])
    return np.array(history)


print("HYPOTHESIS:", HYPOTHESIS)
trap = FourRodTrap(r0=1.0, rod_radius=1.145, outer_radius=5.0, V0=1.0, center_perturb=(0.0, 0.0, 0.08, -0.04), radius_perturb=(0.0, 0.0, 0.05, -0.03))
batch = uniform_batch(trap, n_interior=1024, n_electrode=64, n_outer=256, interior_seed=0)
axis = np.linspace(-0.65, 0.65, 61)
gx, gy = np.meshgrid(axis, axis)
all_points = np.column_stack([gx.ravel(), gy.ravel()])
keep = np.hypot(all_points[:, 0], all_points[:, 1]) <= 0.65
probe_np = all_points[keep]
reference_potential, reference_field = exact_field(trap, probe_np)
exact_null = fit_null(probe_np, reference_potential)
p_scale = np.max(np.abs(reference_potential))
e_scale = np.max(np.linalg.norm(reference_field, axis=1))

specifications = [("output projection K=100", "output", 100), ("paired stored K=50", "paired", 50), ("paired stored K=100", "paired", 100)]
results = []
for label, mode, count in specifications:
    torch.manual_seed(3)
    if mode == "output":
        backend = PigBackend(outer_radius=5.0, num_gaussians=count, feature_dim=4, init_sigma=0.10, decoder_layers=2, decoder_width=16)
        field_model = OutputMirror(backend)
        groups = backend.param_groups(1e-3, lr_centers=1e-4, lr_covariances=1e-4)
        generated = 2 * count
        diagnostics = backend.gaussian_diagnostics()
        centers = np.vstack([np.column_stack([diagnostics["mu_x"], diagnostics["mu_y"]]), np.column_stack([-diagnostics["mu_x"], diagnostics["mu_y"]])])
    else:
        backend = PairedPig(count)
        field_model = backend
        groups = backend.param_groups()
        generated = 2 * count
        centers = None
    history = train(label, backend, field_model, groups, batch)
    points = torch.tensor(probe_np, dtype=torch.float32, requires_grad=True)
    potential_tensor = field_model(points).squeeze(-1)
    gradient = torch.autograd.grad(potential_tensor.sum(), points)[0]
    potential = potential_tensor.detach().numpy()
    field = -gradient.detach().numpy()
    if mode == "paired":
        centers = backend.centers()
    results.append({"label": label, "generated": generated, "stored": count, "history": history, "v": np.sqrt(np.mean(((potential-reference_potential)/p_scale)**2)), "e": np.sqrt(np.mean((np.linalg.norm(field-reference_field, axis=1)/e_scale)**2)), "null_error": np.linalg.norm(fit_null(probe_np, potential)-exact_null), "centers": centers})
    print(label, f"V={results[-1]['v']:.3e} E={results[-1]['e']:.3e} null={results[-1]['null_error']:.3e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for result in results:
    axes[0, 0].plot(result["history"][:, 0], result["history"][:, 1], label=result["label"])
axes[0, 0].set(yscale="log", xlabel="optimizer evaluations", ylabel="PDE + 100 BC", title="Optimization at identical schedules")
axes[0, 0].legend(fontsize=8)
x = np.arange(len(results))
axes[0, 1].bar(x-0.2, [r["v"] for r in results], 0.4, label="V RMSE")
axes[0, 1].bar(x+0.2, [r["e"] for r in results], 0.4, label="E RMSE")
axes[0, 1].set_xticks(x, [r["label"] for r in results], rotation=15, ha="right")
axes[0, 1].set(yscale="log", ylabel="normalized spatial RMSE", title="Independent MFS field gate")
axes[0, 1].legend()
axes[1, 0].bar(x, [r["null_error"] for r in results])
axes[1, 0].set_xticks(x, [r["label"] for r in results], rotation=15, ha="right")
axes[1, 0].set(yscale="log", ylabel="null-position error / r0", title="Displaced electrical-null gate")
colors = ["tab:blue", "tab:orange", "tab:green"]
for result, color in zip(results, colors):
    axes[1, 1].scatter(result["centers"][:, 0], result["centers"][:, 1], s=3, alpha=0.25, label=result["label"], color=color)
axes[1, 1].set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", xlabel="x/r0", ylabel="y/r0", title="Generated Gaussian centers")
axes[1, 1].legend(fontsize=7)
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.suptitle("Hypothesis: mirrored Gaussian pairs can compress one-axis PIG", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

baseline, compressed, expanded = results
compression_pass = compressed["v"] <= 1.25*baseline["v"] and compressed["e"] <= 1.25*baseline["e"] and compressed["null_error"] <= 1.25*baseline["null_error"]
expanded_better = expanded["v"] < baseline["v"] and expanded["e"] < baseline["e"] and expanded["null_error"] < baseline["null_error"]
print("GATES compressed_retains_accuracy=", compression_pass, "expanded_improves_all=", expanded_better)
print("CONCLUSION: accept Gaussian pairing only if the physical field and displaced-null gates pass; parameter compression alone is not evidence of a better solver.")
print("saved", OUTPUT)
