import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


COULOMB_VOLT_METRE = 1.4399645478425669e-9
R0_METRES = 1.0e-3
V0_VOLTS = 100.0
TRAIN_STEPS = 400
OUTPUT_DIR = "exp/coulomb_natural_units_stress_test/results"


def save_point_charge_identity():
    print("\nPLOT 1 HYPOTHESIS: for one elementary point charge, E/C, (V/C)^2, and D^-2 coincide exactly.")
    radius = np.logspace(-9, -2, 300)
    potential = COULOMB_VOLT_METRE / radius
    electric_field = COULOMB_VOLT_METRE / radius**2
    e_scaled = electric_field / COULOMB_VOLT_METRE
    v_scaled = (potential / COULOMB_VOLT_METRE) ** 2
    d_scaled = radius**-2

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.loglog(radius, e_scaled, linewidth=3, label=r"$E/C$")
    ax.loglog(radius, v_scaled, "--", linewidth=2.5, label=r"$(V/C)^2$")
    ax.loglog(radius, d_scaled, ":", linewidth=2.5, label=r"$D^{-2}$")
    ax.set(xlabel="distance D [m]", ylabel=r"scaled value [$m^{-2}$]", title="The proposed collapse is exact for a Coulomb point charge")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(f"{OUTPUT_DIR}/01_point_charge_identity.png", dpi=180)
    maximum_relative_difference = np.max(np.abs(e_scaled - v_scaled) / d_scaled)
    print(f"PLOT 1 CONCLUSION: accepted for the point-charge law; maximum relative mismatch = {maximum_relative_difference:.3e}.")


def save_trap_scale_conditioning():
    print("\nPLOT 2 HYPOTHESIS: equal dimensions do not make the three quantities numerically comparable for a driven Paul trap.")
    characteristic_e = V0_VOLTS / R0_METRES
    values = np.array([
        characteristic_e / COULOMB_VOLT_METRE,
        (V0_VOLTS / COULOMB_VOLT_METRE) ** 2,
        R0_METRES**-2,
    ])

    radii = np.logspace(-6, -2, 121)
    voltages = np.logspace(0, 3, 121)
    rr, vv = np.meshgrid(radii, voltages)
    stack = np.stack([
        vv / (COULOMB_VOLT_METRE * rr),
        (vv / COULOMB_VOLT_METRE) ** 2,
        rr**-2,
    ])
    spread = np.log10(np.max(stack, axis=0) / np.min(stack, axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [r"$E_0/C$", r"$(V_0/C)^2$", r"$r_0^{-2}$"]
    axes[0].bar(labels, values, color=["tab:orange", "tab:blue", "tab:green"])
    axes[0].set_yscale("log")
    axes[0].set(ylabel=r"characteristic value [$m^{-2}$]", title=f"Example trap: V0={V0_VOLTS:g} V, r0={R0_METRES*1e3:g} mm")
    for index, value in enumerate(values):
        axes[0].text(index, value * 1.6, f"{value:.1e}", ha="center", fontsize=9)
    image = axes[1].pcolormesh(radii, voltages, spread, shading="auto", cmap="magma")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].scatter([R0_METRES], [V0_VOLTS], marker="x", s=80, color="cyan", label="example trap")
    axes[1].set(xlabel="r0 [m]", ylabel="V0 [V]", title="Orders-of-magnitude spread despite equal units")
    axes[1].legend()
    fig.colorbar(image, ax=axes[1], label="log10(max/min)")
    fig.savefig(f"{OUTPUT_DIR}/02_trap_scale_conditioning.png", dpi=180)

    print("PLOT 2 VALUES:")
    for label, value in zip(labels, values):
        print(f"  {label}: {value:.6e} m^-2")
    print(f"PLOT 2 CONCLUSION: rejected as a numerical normalization; the example spans {np.log10(values.max()/values.min()):.2f} orders of magnitude.")


def ideal_dimensionless_potential(x, y):
    return 0.5 * (x**2 - y**2)


def save_pde_change():
    print("\nPLOT 3 HYPOTHESIS: squaring potential does not preserve the source-free Laplace equation and removes voltage sign.")
    axis = np.linspace(-1, 1, 241)
    x, y = np.meshgrid(axis, axis)
    u = ideal_dimensionless_potential(x, y)
    w = u**2
    laplace_u = np.zeros_like(u)
    laplace_w = 2.0 * (x**2 + y**2)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    panels = [
        (u, "coolwarm", r"signed $u=V/V_0$"),
        (w, "viridis", r"squared $w=u^2$"),
        (laplace_u, "magma", r"$\nabla^2 u=0$"),
        (laplace_w, "magma", r"$\nabla^2 w=2|\nabla u|^2$"),
    ]
    for ax, (field, cmap, title) in zip(axes, panels):
        image = ax.imshow(field, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap)
        ax.set(xlabel="x/r0", ylabel="y/r0", title=title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Squaring the potential changes both the information and the governing PDE", fontsize=15)
    fig.savefig(f"{OUTPUT_DIR}/03_squared_potential_changes_pde.png", dpi=180)
    print("PLOT 3 CONCLUSION: rejected as a drop-in potential variable; +V and -V become identical after squaring, and the squared field has a nonzero source term.")


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, points):
        return self.layers(points)


def laplacian(values, points):
    gradient = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
    return sum(
        torch.autograd.grad(gradient[:, axis].sum(), points, create_graph=True)[0][:, axis:axis+1]
        for axis in range(2)
    )


def square_boundary(count=96):
    coordinate = torch.linspace(-1, 1, count)
    top = torch.stack([coordinate, torch.ones_like(coordinate)], dim=1)
    bottom = torch.stack([coordinate, -torch.ones_like(coordinate)], dim=1)
    left = torch.stack([-torch.ones_like(coordinate), coordinate], dim=1)
    right = torch.stack([torch.ones_like(coordinate), coordinate], dim=1)
    return torch.cat([top, bottom, left, right])


def train_representation(squared):
    torch.manual_seed(7)
    model = SmallMLP()
    generator = torch.Generator().manual_seed(12)
    interior = 2.0 * torch.rand((1024, 2), generator=generator) - 1.0
    interior.requires_grad_(True)
    boundary = square_boundary()
    exact_boundary = ideal_dimensionless_potential(boundary[:, 0:1], boundary[:, 1:2])
    if squared:
        exact_boundary = exact_boundary**2
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = []
    for step in range(TRAIN_STEPS):
        optimizer.zero_grad()
        interior_values = model(interior)
        pde = torch.mean(laplacian(interior_values, interior)**2)
        bc = torch.mean((model(boundary) - exact_boundary)**2)
        loss = pde + 20.0 * bc
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == TRAIN_STEPS - 1:
            history.append((step, pde.item(), bc.item()))
    return model, np.array(history)


def save_training_stress_test():
    print("\nPLOT 4 HYPOTHESIS: signed nondimensional potential can fit Laplace and boundary data together; squared potential with a naive Laplace loss cannot.")
    signed_model, signed_history = train_representation(False)
    squared_model, squared_history = train_representation(True)
    axis = np.linspace(-1, 1, 101)
    gx, gy = np.meshgrid(axis, axis)
    points_np = np.column_stack([gx.ravel(), gy.ravel()])
    points = torch.tensor(points_np, dtype=torch.float32, requires_grad=True)
    signed_prediction = signed_model(points)
    squared_prediction = squared_model(points)
    signed_residual = laplacian(signed_prediction, points).detach().numpy().reshape(gx.shape)
    squared_residual = laplacian(squared_prediction, points).detach().numpy().reshape(gx.shape)
    signed_map = signed_prediction.detach().numpy().reshape(gx.shape)
    squared_map = squared_prediction.detach().numpy().reshape(gx.shape)
    exact_signed = ideal_dimensionless_potential(gx, gy)
    exact_squared = exact_signed**2
    signed_rmse = np.sqrt(np.mean((signed_map - exact_signed)**2))
    squared_rmse = np.sqrt(np.mean((squared_map - exact_squared)**2))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].semilogy(signed_history[:, 0], signed_history[:, 1], label="PDE")
    axes[0, 0].semilogy(signed_history[:, 0], signed_history[:, 2], label="boundary")
    axes[0, 0].set(title=f"Signed potential training\nfield RMSE={signed_rmse:.2e}", xlabel="Adam step", ylabel="loss")
    axes[0, 0].legend()
    image = axes[0, 1].imshow(signed_map - exact_signed, origin="lower", extent=(-1, 1, -1, 1), cmap="coolwarm")
    axes[0, 1].set(title="signed model - exact", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=axes[0, 1])
    image = axes[0, 2].imshow(signed_residual, origin="lower", extent=(-1, 1, -1, 1), cmap="coolwarm")
    axes[0, 2].set(title="signed Laplace residual", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=axes[0, 2])

    axes[1, 0].semilogy(squared_history[:, 0], squared_history[:, 1], label="naive PDE")
    axes[1, 0].semilogy(squared_history[:, 0], squared_history[:, 2], label="boundary")
    axes[1, 0].set(title=f"Squared potential + wrong PDE\nfield RMSE={squared_rmse:.2e}", xlabel="Adam step", ylabel="loss")
    axes[1, 0].legend()
    image = axes[1, 1].imshow(squared_map - exact_squared, origin="lower", extent=(-1, 1, -1, 1), cmap="coolwarm")
    axes[1, 1].set(title="squared model - exact", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=axes[1, 1])
    image = axes[1, 2].imshow(squared_residual, origin="lower", extent=(-1, 1, -1, 1), cmap="coolwarm")
    axes[1, 2].set(title="naive squared Laplace residual", xlabel="x/r0", ylabel="y/r0")
    fig.colorbar(image, ax=axes[1, 2])
    for ax in axes.ravel():
        ax.grid(alpha=0.15)
    fig.suptitle("Identical network and optimizer: the variable transformation changes the physics problem", fontsize=15)
    fig.savefig(f"{OUTPUT_DIR}/04_short_training_stress_test.png", dpi=180)
    print(f"PLOT 4 RESULTS: signed RMSE={signed_rmse:.4e}; squared RMSE={squared_rmse:.4e}.")
    print("PLOT 4 CONCLUSION: compare the maps, not loss alone; the squared representation is solving an incompatible boundary/PDE problem.")


print("C = k_e e =", COULOMB_VOLT_METRE, "V m")
print("Correct common-unit proposal: E/C, (V/C)^2, D^-2 all have units m^-2")
print("Incorrect variant: (V/C)^-2 has units m^2")
save_point_charge_identity()
save_trap_scale_conditioning()
save_pde_change()
save_training_stress_test()
print("\nOVERALL: the identity is dimensionally correct and exact for a point charge, but it is not a safe drop-in normalization for a Paul-trap PINN.")
