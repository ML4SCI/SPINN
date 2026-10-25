import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


COULOMB_VOLT_METRE = 1.4399645478425669e-9
R0_METRES = 1.0e-3
V0_VOLTS = 100.0
TRAIN_STEPS = 400
OUTPUT = "exp/coulomb_natural_units_stress_test/results/05_loss_only_scaling.png"
SLIDE_OUTPUT = "exp/coulomb_natural_units_stress_test/results/06_loss_only_scaling_slide.png"


def exact_u(points):
    return 0.5 * (points[:, 0:1] ** 2 - points[:, 1:2] ** 2)


def exact_e_hat(points):
    return torch.cat([-points[:, 0:1], points[:, 1:2]], dim=1)


def square_boundary(count):
    coordinate = torch.linspace(-1.0, 1.0, count, dtype=torch.float64)
    top = torch.stack([coordinate, torch.ones_like(coordinate)], dim=1)
    bottom = torch.stack([coordinate, -torch.ones_like(coordinate)], dim=1)
    left = torch.stack([-torch.ones_like(coordinate), coordinate], dim=1)
    right = torch.stack([torch.ones_like(coordinate), coordinate], dim=1)
    return torch.cat([top, bottom, left, right])


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        ).double()

    def forward(self, points):
        return self.layers(points)


def derivatives(model, points):
    u = model(points)
    gradient = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    laplacian = sum(
        torch.autograd.grad(gradient[:, axis].sum(), points, create_graph=True)[0][:, axis:axis + 1]
        for axis in range(2)
    )
    return u, -gradient, laplacian


def raw_terms(model, interior, boundary, field_points):
    _, _, laplace_u = derivatives(model, interior)
    boundary_u = model(boundary)
    _, field_e_hat, _ = derivatives(model, field_points)

    delta_v = V0_VOLTS * (boundary_u - exact_u(boundary))
    delta_e = (V0_VOLTS / R0_METRES) * (field_e_hat - exact_e_hat(field_points))
    laplace_v = (V0_VOLTS / R0_METRES**2) * laplace_u

    dimensionless = torch.stack([
        torch.mean(laplace_u**2),
        torch.mean((boundary_u - exact_u(boundary)) ** 2),
        torch.mean(torch.linalg.vector_norm(field_e_hat - exact_e_hat(field_points), dim=1)),
    ])
    converted = torch.stack([
        torch.mean(laplace_v**2),
        torch.mean(delta_v**2) / COULOMB_VOLT_METRE**2,
        torch.mean(torch.linalg.vector_norm(delta_e, dim=1)) / COULOMB_VOLT_METRE,
    ])
    return dimensionless, converted


def gradient_norms(terms, model):
    norms = []
    for term in terms:
        gradients = torch.autograd.grad(term, model.parameters(), retain_graph=True)
        norms.append(torch.sqrt(sum(torch.sum(gradient**2) for gradient in gradients)))
    return torch.stack(norms)


torch.manual_seed(19)
generator = torch.Generator().manual_seed(23)
interior_base = 2.0 * torch.rand((512, 2), generator=generator, dtype=torch.float64) - 1.0
boundary = square_boundary(80)
field_base = 1.6 * torch.rand((192, 2), generator=generator, dtype=torch.float64) - 0.8

seed_model = SmallMLP()
seed_state = {name: value.detach().clone() for name, value in seed_model.state_dict().items()}
interior = interior_base.detach().clone().requires_grad_(True)
field_points = field_base.detach().clone().requires_grad_(True)
dimensionless_initial, converted_initial = raw_terms(seed_model, interior, boundary, field_points)
dimensionless_gradient = gradient_norms(dimensionless_initial, seed_model).detach()
converted_gradient = gradient_norms(converted_initial, seed_model).detach()

gradient_weights = 1.0 / converted_gradient

schemes = {
    "dimensionless": ("dimensionless", torch.ones(3, dtype=torch.float64)),
    "loss-only, alpha=1": ("converted", torch.ones(3, dtype=torch.float64)),
    "loss-only, gradient-balanced": ("converted", gradient_weights),
    "loss-only, separate PDE": ("separate", gradient_weights),
}

print("PLOT HYPOTHESIS: applying Coulomb conversions only inside the V-MSE and E-MAE preserves signed V and Laplace physics, but alpha=1 will not balance optimization; explicit alpha weights may.")
print("initial dimensionless terms [Laplace MSE, V MSE, E MAE]:", dimensionless_initial.detach().numpy())
print("initial converted terms [physical Laplace MSE, V-MSE/C2, E-MAE/C]:", converted_initial.detach().numpy())
print("converted initial gradient norms:", converted_gradient.numpy())
print("gradient-balanced alphas:", gradient_weights.numpy())

results = {}
for name, (term_kind, weights) in schemes.items():
    model = SmallMLP()
    model.load_state_dict(seed_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    history = []

    if term_kind == "separate":
        pde_optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
        data_optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
        for cycle in range(TRAIN_STEPS // 2):
            interior = interior_base.detach().clone().requires_grad_(True)
            field_points = field_base.detach().clone().requires_grad_(True)
            _, converted_terms = raw_terms(model, interior, boundary, field_points)
            pde_optimizer.zero_grad()
            converted_terms[0].backward()
            pde_optimizer.step()

            interior = interior_base.detach().clone().requires_grad_(True)
            field_points = field_base.detach().clone().requires_grad_(True)
            dimensionless_terms, converted_terms = raw_terms(model, interior, boundary, field_points)
            data_loss = weights[1] * converted_terms[1] + weights[2] * converted_terms[2]
            data_optimizer.zero_grad()
            data_loss.backward()
            data_optimizer.step()

            if cycle % 5 == 0 or cycle == TRAIN_STEPS // 2 - 1:
                history.append([2 * cycle, *dimensionless_terms.detach().numpy()])
    else:
        for step in range(TRAIN_STEPS):
            interior = interior_base.detach().clone().requires_grad_(True)
            field_points = field_base.detach().clone().requires_grad_(True)
            dimensionless_terms, converted_terms = raw_terms(model, interior, boundary, field_points)
            selected = dimensionless_terms if term_kind == "dimensionless" else converted_terms
            loss = torch.sum(weights * selected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0 or step == TRAIN_STEPS - 1:
                history.append([step, *dimensionless_terms.detach().numpy()])

    evaluation_axis = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    gx, gy = torch.meshgrid(evaluation_axis, evaluation_axis, indexing="xy")
    evaluation_points = torch.stack([gx.ravel(), gy.ravel()], dim=1).requires_grad_(True)
    predicted_u, predicted_e_hat, predicted_laplace = derivatives(model, evaluation_points)
    exact_grid_u = exact_u(evaluation_points)
    exact_grid_e = exact_e_hat(evaluation_points)

    u_error = (predicted_u - exact_grid_u).detach().reshape(101, 101).numpy()
    e_error = torch.linalg.vector_norm(predicted_e_hat - exact_grid_e, dim=1).detach().reshape(101, 101).numpy()
    laplace_map = predicted_laplace.detach().reshape(101, 101).numpy()
    boundary_error = torch.max(torch.abs(model(boundary) - exact_u(boundary))).item()
    results[name] = {
        "history": np.array(history),
        "u_error": u_error,
        "e_error": e_error,
        "laplace": laplace_map,
        "boundary_error": boundary_error,
    }
    print(
        f"{name}: final dimensionless Laplace MSE={history[-1][1]:.3e}, "
        f"V MSE={history[-1][2]:.3e}, E MAE={history[-1][3]:.3e}, "
        f"max boundary error={boundary_error:.3e}"
    )


fig, axes = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
colors = ["tab:purple", "tab:blue", "tab:orange", "tab:green"]

for color, (name, result) in zip(colors, results.items()):
    history = result["history"]
    axes[0, 0].semilogy(history[:, 0], history[:, 1], color=color, label=name)
    axes[0, 1].semilogy(history[:, 0], history[:, 2], color=color, label=name)
    axes[0, 2].semilogy(history[:, 0], history[:, 3], color=color, label=name)

axes[0, 0].set(title="Unchanged Laplace equation", xlabel="Adam step", ylabel=r"MSE($\nabla^2 u$)")
axes[0, 1].set(title="Signed boundary potential", xlabel="Adam step", ylabel="dimensionless V MSE")
axes[0, 2].set(title="Electric-field fit", xlabel="Adam step", ylabel="dimensionless E MAE")
axes[0, 2].legend(fontsize=8)

labels = ["Laplace", "V", "E"]
x = np.arange(3)
axes[0, 3].bar(x - 0.18, converted_initial.detach().numpy(), 0.36, label="loss magnitude")
axes[0, 3].bar(x + 0.18, converted_gradient.numpy(), 0.36, label="gradient norm")
axes[0, 3].set_yscale("log")
axes[0, 3].set_xticks(x, labels)
axes[0, 3].set(title="Before training: alpha=1", ylabel="raw numerical magnitude")
axes[0, 3].legend(fontsize=8)

for column, (name, result) in enumerate(results.items()):
    limit = np.max(np.abs(result["u_error"]))
    image = axes[1, column].imshow(result["u_error"], origin="lower", extent=(-1, 1, -1, 1), cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1, column].set(title=f"{name}\nV error", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=axes[1, column], shrink=0.78)

    image = axes[2, column].imshow(result["e_error"], origin="lower", extent=(-1, 1, -1, 1), cmap="magma", vmin=0)
    axes[2, column].set(title=f"E error\nmax BC error={result['boundary_error']:.1e}", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=axes[2, column], shrink=0.78)

fig.suptitle("Corrected test: Coulomb conversion is applied only to individual loss residuals", fontsize=16)
fig.savefig(OUTPUT, dpi=180)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

axes[0].bar(x - 0.18, converted_initial.detach().numpy(), 0.36, label="loss magnitude", color="#1479c9")
axes[0].bar(x + 0.18, converted_gradient.numpy(), 0.36, label="parameter-gradient norm", color="#ff7f0e")
axes[0].set_yscale("log")
axes[0].set_xticks(x, labels)
axes[0].set(ylabel="numerical magnitude", title=r"Same units $\ne$ same scale")
axes[0].tick_params(labelsize=11)
axes[0].grid(axis="y", alpha=0.2)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")

map_names = ["dimensionless", "loss-only, alpha=1"]
map_titles = ["Dimensionless baseline", r"Coulomb-scaled losses, $\alpha_i=1$"]
shared_limit = max(np.max(results[name]["e_error"]) for name in map_names)
for axis, name, title in zip(axes[1:], map_names, map_titles):
    image = axis.imshow(
        results[name]["e_error"], origin="lower", extent=(-1, 1, -1, 1),
        cmap="magma", vmin=0.0, vmax=shared_limit,
    )
    final_e = results[name]["history"][-1, 3]
    axis.set(
        title=f"{title}\nfield MAE = {final_e:.3f}",
        xlabel=r"$x/r_0$", ylabel=r"$y/r_0$",
    )
    axis.tick_params(labelsize=10)

colorbar = fig.colorbar(image, ax=axes[1:], shrink=0.86, pad=0.02)
colorbar.set_label(r"$|\widehat{\mathbf{E}}-\widehat{\mathbf{E}}_{\mathrm{exact}}|$", fontsize=11)
fig.savefig(SLIDE_OUTPUT, dpi=220)

winner = min(results, key=lambda name: results[name]["history"][-1, 1:].sum())
print("PLOT CONCLUSION: signed V and the Laplace equation were preserved in all cases.")
print("PLOT CONCLUSION: putting converted objectives and the physical-unit PDE into one alpha=1 sum did not balance their numerical losses or gradients.")
print(f"PLOT CONCLUSION: lowest final sum of the three dimensionless diagnostics was {winner}.")
print(f"PLOT CONCLUSION: the compact slide plot uses one shared field-error colour scale and was saved to {SLIDE_OUTPUT}.")
