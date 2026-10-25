import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

# -----------------------------
# Config
# -----------------------------
config = {
    "seed": 0,

    "output_root": "runs",

    "physics": {
        "e_charge": 1.602e-19, # electron charge [C]
        "amu": 1.66054e-27, # atomic mass unit [kg]
        "ion_mass_amu": 40.0, # assume Ca40 ion
        "rf_frequency_hz": 10.2e6, # RF frequency [hz]
        "geom_unit_to_m": 1e-2,   # geometry units [cm]
    },

    "geometry": {
        "world_size_x": 0.5,
        "world_size_y": 0.5,
        "radius": 0.5e-1,
        "N_distance": 1.25e-1,
        "S_distance": 1.25e-1,
        "W_distance": 1.25e-1,
        "E_distance": 1.25e-1,
    },

    "boundary_conditions": {
        # Outer world boundary condition.
        # type = "dirichlet" enforces Phi = value.
        # type = "neumann" enforces dPhi/dn = value, where n is the outward normal.
        # The Neumann value is in volts per geometry unit. Here geometry units are cm.
        "world": {
            "type": "neumann",
            "value": 0.0,
        },
        "rf_voltages": {
            "North": 0.0,
            "South": 0.0,
            "West": 1.0,
            "East": 1.0,
        },
    },

    "model": {
        "n_gaussians": 50,
        "feature_dim": 16,
        "mlp_hidden": 16,
        # Initial std of each Gaussian, in normalized [0, 1] domain units.
        "sigma_init": 0.1,
        # Initial std of the learnable per-Gaussian feature values.
        "feature_init_std": 0.01,
    },

    "training": {
        "steps": 20000,
        "lr": 1e-3,
        "n_int": 4096,
        "n_world": 1024,
        "n_electrode": 2048,
        "bc_weight": 100.0,
        "print_every": 100,

        # If True, compute spatial derivatives using closed-form Gaussian
        "use_analytic_derivatives": True,

        "use_lr_scheduler": False,
        "lr_scheduler": {
            "factor": 0.5,
            "patience": 300,
            "min_lr": 1e-6,
        },

    "use_grad_clip": True,
    "grad_clip_max_norm": 100.0,
    },

    "evaluation": {
        "nx": 1000,
        "ny": 800,
        # Grid points are evaluated in chunks to bound the memory used by the
        # (points x gaussians) distance tensors.
        "chunk_size": 8192,
    },
}


# -----------------------------
# Run directory / saving
# -----------------------------
def make_run_dir(config):
    root = Path(config["output_root"])
    root.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"paul_trap_pig_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def save_config(config, run_dir):
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_script_copy(run_dir):
    try:
        script_path = Path(__file__).resolve()
        shutil.copy(script_path, run_dir / script_path.name)
    except NameError:
        print("Could not save script copy because __file__ is not defined.")
        print("This usually happens in notebooks or interactive sessions.")


# -----------------------------
# Physical constants
# -----------------------------
p = config["physics"]

e_charge = p["e_charge"]
amu = p["amu"]
M = p["ion_mass_amu"] * amu

f = p["rf_frequency_hz"]
Omega = 2 * np.pi * f
scale = e_charge**2 / (4 * M * Omega**2)

geom_unit_to_m = p["geom_unit_to_m"]


# -----------------------------
# Geometry
# -----------------------------
g = config["geometry"]

world_size_x = g["world_size_x"]
world_size_y = g["world_size_y"]
radius = g["radius"]

N_distance = g["N_distance"]
S_distance = g["S_distance"]
W_distance = g["W_distance"]
E_distance = g["E_distance"]

xmin, xmax = -world_size_x / 2, world_size_x / 2
ymin, ymax = -world_size_y / 2, world_size_y / 2

electrodes = {
    "North": (0.0,  N_distance, radius),
    "South": (0.0, -S_distance, radius),
    "West":  (-W_distance, 0.0, radius),
    "East":  ( E_distance, 0.0, radius),
}

bc = config["boundary_conditions"]
rf_voltages = bc["rf_voltages"]
world_bc = bc.get("world", {
    "type": "dirichlet",
    "value": bc.get("world_voltage", 0.0),
})
world_bc_type = world_bc["type"].lower()
world_bc_value = float(world_bc["value"])

if world_bc_type not in {"dirichlet", "neumann"}:
    raise ValueError(
        "boundary_conditions['world']['type'] must be either "
        "'dirichlet' or 'neumann'."
    )


# -----------------------------
# Sampling
# -----------------------------
def inside_any_electrode(xy):
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)

    for cx, cy, r in electrodes.values():
        inside |= (x - cx)**2 + (y - cy)**2 <= r**2

    return inside


def sample_interior(n):
    pts = []

    while len(pts) < n:
        xy = np.column_stack([
            np.random.uniform(xmin, xmax, n),
            np.random.uniform(ymin, ymax, n),
        ])
        xy = xy[~inside_any_electrode(xy)]
        pts.extend(xy.tolist())

    return np.array(pts[:n], dtype=np.float32)


def sample_world_boundary(n):
    n_side = [n // 4] * 4
    for i in range(n % 4):
        n_side[i] += 1

    n_bottom, n_top, n_left, n_right = n_side

    xb = np.random.uniform(xmin, xmax, n_bottom)
    xt = np.random.uniform(xmin, xmax, n_top)
    yl = np.random.uniform(ymin, ymax, n_left)
    yr = np.random.uniform(ymin, ymax, n_right)

    bottom = np.column_stack([xb, np.full(n_bottom, ymin)])
    top = np.column_stack([xt, np.full(n_top, ymax)])
    left = np.column_stack([np.full(n_left, xmin), yl])
    right = np.column_stack([np.full(n_right, xmax), yr])

    xy = np.vstack([bottom, top, left, right]).astype(np.float32)

    normals = np.vstack([
        np.tile([0.0, -1.0], (n_bottom, 1)),
        np.tile([0.0,  1.0], (n_top, 1)),
        np.tile([-1.0, 0.0], (n_left, 1)),
        np.tile([ 1.0, 0.0], (n_right, 1)),
    ]).astype(np.float32)

    values = np.full((len(xy), 1), world_bc_value, dtype=np.float32)

    return xy, normals, values


def sample_circle_boundary(cx, cy, r, n):
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    return np.column_stack([x, y]).astype(np.float32)


def sample_electrode_boundaries(n_per_electrode):
    xs, vs = [], []

    for name, (cx, cy, r) in electrodes.items():
        xy = sample_circle_boundary(cx, cy, r, n_per_electrode)
        v = np.full((n_per_electrode, 1), rf_voltages[name], dtype=np.float32)

        xs.append(xy)
        vs.append(v)

    return np.vstack(xs), np.vstack(vs)


def to_tensor(x, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


# -----------------------------
# PIG model
# -----------------------------
class PIG(nn.Module):

    def __init__(self, n_gaussians, feature_dim, mlp_hidden,
                 sigma_init, feature_init_std):
        super().__init__()

        self.register_buffer(
            "domain_lo", torch.tensor([xmin, ymin], dtype=torch.float32)
        )
        self.register_buffer(
            "domain_size",
            torch.tensor([xmax - xmin, ymax - ymin], dtype=torch.float32),
        )

        N, k = n_gaussians, feature_dim

        self.mu = nn.Parameter(torch.rand(N, k, 2))
        self.log_sigma = nn.Parameter(
            torch.full((N, k, 2), float(np.log(sigma_init)))
        )
        self.features = nn.Parameter(feature_init_std * torch.randn(N, k))

        self.mlp = nn.Sequential(
            nn.Linear(k, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 1),
        )

    def gaussian_features(self, xy):
        u = (xy - self.domain_lo) / self.domain_size

        # (batch, N, k, 2) -> Gaussian weights (batch, N, k)
        diff = u[:, None, None, :] - self.mu
        sigma = torch.exp(self.log_sigma)
        G = torch.exp(-0.5 * ((diff / sigma)**2).sum(dim=-1))

        # Weighted sum over Gaussians -> feature vector (batch, k)
        return (self.features * G).sum(dim=1)

    def forward(self, xy):
        return self.mlp(self.gaussian_features(xy))

    def value_grad_laplace_analytic(self, xy):

        if (
            len(self.mlp) != 3
            or not isinstance(self.mlp[0], nn.Linear)
            or not isinstance(self.mlp[1], nn.Tanh)
            or not isinstance(self.mlp[2], nn.Linear)
            or self.mlp[2].out_features != 1
        ):
            raise RuntimeError(
                "Analytic derivatives require the current PIG MLP form: "
                "Linear -> Tanh -> Linear with one scalar output."
            )

        u = (xy - self.domain_lo) / self.domain_size

        # diff: (batch, n_gaussians, feature_dim, xy_dim)
        diff = u[:, None, None, :] - self.mu
        sigma = torch.exp(self.log_sigma)
        inv_s2 = 1.0 / sigma**2
        inv_s4 = inv_s2**2

        q = torch.sum(diff**2 * inv_s2[None, :, :, :], dim=-1)
        G = torch.exp(-0.5 * q)

        # Gaussian feature vector F: (batch, feature_dim)
        F = torch.sum(self.features[None, :, :] * G, dim=1)

        # First and second derivatives of each Gaussian with respect to
        # normalized coordinates u.
        dG_du = -G[:, :, :, None] * diff * inv_s2[None, :, :, :]
        d2G_du2 = G[:, :, :, None] * (
            diff**2 * inv_s4[None, :, :, :] - inv_s2[None, :, :, :]
        )

        dF_du = torch.sum(
            self.features[None, :, :, None] * dG_du,
            dim=1,
        )
        d2F_du2 = torch.sum(
            self.features[None, :, :, None] * d2G_du2,
            dim=1,
        )

        # Convert derivatives from normalized coordinates u to geometry
        # coordinates xy.
        grad_F = dF_du / self.domain_size[None, None, :]
        lap_F = torch.sum(
            d2F_du2 / self.domain_size[None, None, :]**2,
            dim=-1,
        )

        # MLP value, gradient, and Hessian with respect to F.
        lin1 = self.mlp[0]
        lin2 = self.mlp[2]
        W1 = lin1.weight
        b1 = lin1.bias
        W2 = lin2.weight[0]
        b2 = lin2.bias[0]

        z = F @ W1.T + b1[None, :]
        h = torch.tanh(z)
        phi = h @ W2[:, None] + b2

        dh_dz = 1.0 - h**2
        d2h_dz2 = -2.0 * h * dh_dz

        # m_grad[b, j] = d Phi / d F_j
        m_grad = (dh_dz * W2[None, :]) @ W1

        # m_hess[b, j, l] = d^2 Phi / d F_j d F_l
        weighted_second = d2h_dz2 * W2[None, :]
        m_hess = torch.einsum(
            "bh,hj,hl->bjl",
            weighted_second,
            W1,
            W1,
        )

        grad_phi = torch.einsum("bk,bka->ba", m_grad, grad_F)

        lap_term_1 = torch.sum(m_grad * lap_F, dim=1, keepdim=True)
        lap_term_2 = torch.einsum(
            "bjl,bja,bla->b",
            m_hess,
            grad_F,
            grad_F,
        ).unsqueeze(1)

        lap_phi = lap_term_1 + lap_term_2

        return phi, grad_phi, lap_phi

    def gaussian_params_numpy(self):
        """Gaussian centers, stds and features in physical (geometry) units."""
        lo = self.domain_lo.numpy()
        size = self.domain_size.numpy()

        mu = lo + self.mu.detach().numpy() * size
        sigma = np.exp(self.log_sigma.detach().numpy()) * size
        features = self.features.detach().numpy()

        return mu, sigma, features


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def laplacian(phi, xy):
    grad_phi = torch.autograd.grad(
        phi,
        xy,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
    )[0]

    phi_x = grad_phi[:, 0:1]
    phi_y = grad_phi[:, 1:2]

    phi_xx = torch.autograd.grad(
        phi_x,
        xy,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True,
    )[0][:, 0:1]

    phi_yy = torch.autograd.grad(
        phi_y,
        xy,
        grad_outputs=torch.ones_like(phi_y),
        create_graph=True,
    )[0][:, 1:2]

    return phi_xx + phi_yy


def world_boundary_loss(model, xy_world, n_world, values_world, use_analytic):
    if world_bc_type == "dirichlet":
        phi_world = model(xy_world)
        return torch.mean((phi_world - values_world)**2)

    if world_bc_type == "neumann":
        if use_analytic:
            _, grad_world, _ = model.value_grad_laplace_analytic(xy_world)
        else:
            phi_world = model(xy_world)
            grad_world = torch.autograd.grad(
                phi_world,
                xy_world,
                grad_outputs=torch.ones_like(phi_world),
                create_graph=True,
            )[0]

        dphi_dn = torch.sum(grad_world * n_world, dim=1, keepdim=True)
        return torch.mean((dphi_dn - values_world)**2)

    raise RuntimeError(f"Unknown world boundary condition type: {world_bc_type}")


def pinn_loss(model):
    train_cfg = config["training"]
    use_analytic = train_cfg.get("use_analytic_derivatives", False)

    xy_int = to_tensor(
        sample_interior(train_cfg["n_int"]),
        requires_grad=not use_analytic,
    )

    if use_analytic:
        _, _, lap_phi = model.value_grad_laplace_analytic(xy_int)
    else:
        phi_int = model(xy_int)
        lap_phi = laplacian(phi_int, xy_int)

    loss_pde = torch.mean(lap_phi**2)

    xy_world, n_world, values_world = sample_world_boundary(train_cfg["n_world"])
    xy_world = to_tensor(
        xy_world,
        requires_grad=(world_bc_type == "neumann" and not use_analytic),
    )
    n_world = to_tensor(n_world)
    values_world = to_tensor(values_world)
    loss_world = world_boundary_loss(
        model,
        xy_world,
        n_world,
        values_world,
        use_analytic=use_analytic,
    )

    xy_el, v_el = sample_electrode_boundaries(train_cfg["n_electrode"])
    xy_el = to_tensor(xy_el)
    v_el = to_tensor(v_el)
    loss_electrode = torch.mean((model(xy_el) - v_el)**2)

    loss_bc = loss_world + loss_electrode
    loss = loss_pde + train_cfg["bc_weight"] * loss_bc

    return loss, loss_pde.detach(), loss_bc.detach()

def grad_norm(model):
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2

    return total ** 0.5


def train(model, run_dir):
    train_cfg = config["training"]

    print(
        f"World BC: {world_bc_type}, "
        f"value={world_bc_value:g}"
    )
    print(
        "Derivative mode: "
        f"{'analytic' if train_cfg.get('use_analytic_derivatives', False) else 'autograd'}"
    )

    opt = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    scheduler = None
    if train_cfg.get("use_lr_scheduler", False):
        sched_cfg = train_cfg["lr_scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=sched_cfg["factor"],
            patience=sched_cfg["patience"],
            min_lr=sched_cfg["min_lr"],
        )

    hist = []

    for step in range(train_cfg["steps"]):
        opt.zero_grad()

        loss, loss_pde, loss_bc = pinn_loss(model)

        loss.backward()

        grad_pre_clip = grad_norm(model)

        if train_cfg.get("use_grad_clip", False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=train_cfg["grad_clip_max_norm"],
            )

        grad_post_clip = grad_norm(model)

        opt.step()

        current_lr = opt.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(loss.item())

        row = [
            step,
            loss.item(),
            loss_pde.item(),
            loss_bc.item(),
            current_lr,
            grad_pre_clip,
            grad_post_clip,
        ]

        if step % train_cfg["print_every"] == 0:
            hist.append(row)

            print(
                f"{step:6d} | "
                f"loss={loss.item():.3e} | "
                f"pde={loss_pde.item():.3e} | "
                f"bc={loss_bc.item():.3e} | "
                f"lr={current_lr:.3e} | "
                f"grad={grad_pre_clip:.3e}->{grad_post_clip:.3e}"
            )

    hist = np.array(hist)

    np.savetxt(
        run_dir / "training_history.csv",
        hist,
        delimiter=",",
        header="step,total_loss,pde_loss,bc_loss,lr,grad_pre_clip,grad_post_clip",
        comments="",
    )

    torch.save(model.state_dict(), run_dir / "model_state_dict.pt")

    return hist


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_on_grid(model):
    eval_cfg = config["evaluation"]

    nx = eval_cfg["nx"]
    ny = eval_cfg["ny"]
    chunk_size = eval_cfg["chunk_size"]

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    mask = inside_any_electrode(xy).reshape(ny, nx)

    phi_chunks = []
    grad2_chunks = []

    use_analytic = config["training"].get("use_analytic_derivatives", False)

    for start in range(0, len(xy), chunk_size):
        xy_t = to_tensor(
            xy[start:start + chunk_size],
            requires_grad=not use_analytic,
        )

        if use_analytic:
            phi, grad_phi, _ = model.value_grad_laplace_analytic(xy_t)
        else:
            phi = model(xy_t)
            grad_phi = torch.autograd.grad(
                phi,
                xy_t,
                grad_outputs=torch.ones_like(phi),
                create_graph=False,
            )[0]

        grad_phi_m = grad_phi / geom_unit_to_m
        grad2 = torch.sum(grad_phi_m**2, dim=1, keepdim=True)

        phi_chunks.append(phi.detach())
        grad2_chunks.append(grad2.detach())

    phi = torch.cat(phi_chunks)
    grad2 = torch.cat(grad2_chunks)

    psi_J = scale * grad2
    psi_eV = psi_J / e_charge

    phi_np = phi.numpy().reshape(ny, nx)
    psi_eV_np = psi_eV.numpy().reshape(ny, nx)

    phi_np = np.where(mask, np.nan, phi_np)
    psi_eV_np = np.where(mask, np.nan, psi_eV_np)

    return X, Y, phi_np, psi_eV_np


def save_fields(run_dir, X, Y, phi, psi_eV):
    np.savez(
        run_dir / "fields.npz",
        X=X,
        Y=Y,
        phi_rf=phi,
        psi_rf_eV=psi_eV,
        scale=scale,
        e_charge=e_charge,
        M=M,
        Omega=Omega,
        geom_unit_to_m=geom_unit_to_m,
    )


def save_gaussian_params(run_dir, mu, sigma, features, tag):
    np.savez(
        run_dir / f"gaussians_{tag}.npz",
        mu=mu,
        sigma=sigma,
        features=features,
    )



def add_electrode_patches(ax, facecolor="white", edgecolor="black", linewidth=1.5):

    for name, (cx, cy, r) in electrodes.items():

        circle = Circle(
            (cx, cy),
            r,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=10,
        )

        ax.add_patch(circle)

        # ax.text(
        #     cx,
        #     cy,
        #     name[0],
        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     zorder=11,
        # )

def plot_field(run_dir, X, Y, field, filename, colorbar_label, title=None):

    fig, ax = plt.subplots()

    field_masked = np.ma.masked_invalid(field)

    c = ax.contourf(
        X,
        Y,
        field_masked,
        levels=100,
        cmap="jet",
    )

    fig.colorbar(c, ax=ax, label=colorbar_label)
    add_electrode_patches(ax)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=300)

    return fig, ax


def plot_gaussians(run_dir, mu, sigma, filename, title, background=None):

    fig, ax = plt.subplots()

    if background is not None:
        X, Y, field = background
        c = ax.contourf(
            X,
            Y,
            np.ma.masked_invalid(field),
            levels=100,
            cmap="jet",
        )
        fig.colorbar(c, ax=ax, label=r"$\psi_{\rm RF}$ [eV]")

    n_gaussians, k = mu.shape[0], mu.shape[1]
    colors = plt.cm.tab10.colors

    for j in range(k):
        color = colors[j % len(colors)]

        ax.scatter(
            mu[:, j, 0],
            mu[:, j, 1],
            s=4,
            color=color,
            label=f"feature dim {j}",
            zorder=6,
        )

        for i in range(n_gaussians):
            ellipse = Ellipse(
                (mu[i, j, 0], mu[i, j, 1]),
                width=2 * sigma[i, j, 0],
                height=2 * sigma[i, j, 1],
                facecolor="none",
                edgecolor=color,
                linewidth=0.5,
                alpha=0.15,
                zorder=5,
            )
            ax.add_patch(ellipse)

    add_electrode_patches(
        ax,
        facecolor="white" if background is not None else "none",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=300)

    return fig, ax


def plot_loss_history(run_dir, hist):
    fig, ax1 = plt.subplots()

    ax1.semilogy(hist[:, 0], hist[:, 1], label="total")
    ax1.semilogy(hist[:, 0], hist[:, 2], label="pde")
    ax1.semilogy(hist[:, 0], hist[:, 3], label="bc")

    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()

    if hist.shape[1] > 4:
        ax2.semilogy(hist[:, 0], hist[:, 4], linestyle="--", label="lr")

    if hist.shape[1] > 6:
        ax2.semilogy(hist[:, 0], hist[:, 5], linestyle=":", label="grad pre")
        ax2.semilogy(hist[:, 0], hist[:, 6], linestyle="-.", label="grad post")

    ax2.set_ylabel("lr / gradient norm")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(run_dir / "loss_history.png", dpi=300)

    return fig, ax1

def plot_and_save_results(run_dir, X, Y, phi, psi_eV, hist,
                          gaussians_initial, gaussians_final):
    plot_loss_history(run_dir, hist)

    plot_field(
        run_dir,
        X,
        Y,
        phi,
        filename="phi_rf.png",
        colorbar_label=r"$\Phi_{\rm RF}$ [V]",
        title=r"RF Potential Amplitude $\Phi_{\rm RF}$",
    )

    plot_field(
        run_dir,
        X,
        Y,
        psi_eV,
        filename="psi_rf_eV.png",
        colorbar_label=r"$\psi_{\rm RF}$ [eV]",
        title=r"RF Pseudopotential $\psi_{\rm RF}$",
    )

    mu0, sigma0, _ = gaussians_initial
    plot_gaussians(
        run_dir,
        mu0,
        sigma0,
        filename="gaussians_initial.png",
        title="Gaussians at initialization (centers + 1$\\sigma$ ellipses)",
    )

    mu1, sigma1, _ = gaussians_final
    plot_gaussians(
        run_dir,
        mu1,
        sigma1,
        filename="gaussians_final.png",
        title="Gaussians after training (centers + 1$\\sigma$ ellipses)",
    )

    plot_gaussians(
        run_dir,
        mu1,
        sigma1,
        filename="gaussians_on_psi.png",
        title="Trained Gaussians over $\\psi_{\\rm RF}$",
        background=(X, Y, psi_eV),
    )

    plt.show()

# -----------------------------
# Runtime logging
# -----------------------------
def save_runtime_summary(run_dir, runtime):
    with open(run_dir / "runtime_summary.json", "w") as f:
        json.dump(runtime, f, indent=2)

    keys = [
        "derivative_mode",
        "training_seconds",
        "evaluation_seconds",
        "plotting_seconds",
        "total_seconds",
    ]

    with open(run_dir / "runtime_summary.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(runtime[k]) for k in keys) + "\n")


# -----------------------------
# Run
# -----------------------------
total_timer_start = perf_counter()

run_dir = make_run_dir(config)
save_config(config, run_dir)
save_script_copy(run_dir)

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

model_cfg = config["model"]
model = PIG(
    n_gaussians=model_cfg["n_gaussians"],
    feature_dim=model_cfg["feature_dim"],
    mlp_hidden=model_cfg["mlp_hidden"],
    sigma_init=model_cfg["sigma_init"],
    feature_init_std=model_cfg["feature_init_std"],
)

print(
    f"PIG model: {model_cfg['n_gaussians']} Gaussians x "
    f"{model_cfg['feature_dim']} feature dims, "
    f"{count_parameters(model)} trainable parameters"
)

gaussians_initial = model.gaussian_params_numpy()
save_gaussian_params(run_dir, *gaussians_initial, tag="initial")

train_timer_start = perf_counter()
hist = train(model, run_dir)
training_seconds = perf_counter() - train_timer_start

gaussians_final = model.gaussian_params_numpy()
save_gaussian_params(run_dir, *gaussians_final, tag="final")

eval_timer_start = perf_counter()
X, Y, phi, psi_eV = evaluate_on_grid(model)
evaluation_seconds = perf_counter() - eval_timer_start

save_fields(run_dir, X, Y, phi, psi_eV)

plot_timer_start = perf_counter()
plot_and_save_results(
    run_dir, X, Y, phi, psi_eV, hist,
    gaussians_initial, gaussians_final,
)
plotting_seconds = perf_counter() - plot_timer_start
total_seconds = perf_counter() - total_timer_start

runtime = {
    "derivative_mode": (
        "analytic"
        if config["training"].get("use_analytic_derivatives", False)
        else "autograd"
    ),
    "training_seconds": training_seconds,
    "evaluation_seconds": evaluation_seconds,
    "plotting_seconds": plotting_seconds,
    "total_seconds": total_seconds,
    "training_minutes": training_seconds / 60.0,
    "evaluation_minutes": evaluation_seconds / 60.0,
    "plotting_minutes": plotting_seconds / 60.0,
    "total_minutes": total_seconds / 60.0,
}
save_runtime_summary(run_dir, runtime)

print(f"Runtime summary saved to: {run_dir / 'runtime_summary.json'}")
print(f"Total runtime: {total_seconds:.2f} s ({total_seconds / 60.0:.2f} min)")
print(f"\nSaved run to: {run_dir.resolve()}")
