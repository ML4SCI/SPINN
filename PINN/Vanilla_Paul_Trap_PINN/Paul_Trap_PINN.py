import os
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
        "world_size_x": 2.5,
        "world_size_y": 2.0,
        "radius": 0.5e-1,
        "N_distance": 1.25e-1,
        "S_distance": 1.25e-1,
        "W_distance": 1.25e-1,
        "E_distance": 1.25e-1,
    },

    "boundary_conditions": {
        "world_voltage": 0.0,
        "rf_voltages": {
            "North": 0.0,
            "South": 0.0,
            "West": 300.0,
            "East": 300.0,
        },
    },

    "model": {
        "width": 128,
        "depth": 4,
    },

    "training": {
        "steps": 20000,
        "lr": 1e-3,
        "n_int": 4096,
        "n_world": 1024,
        "n_electrode": 2048,
        "bc_weight": 100.0,
        "print_every": 100,

        "use_lr_scheduler": True,
        "lr_scheduler": {
            "factor": 0.5,
            "patience": 300,
            "min_lr": 1e-6,
        },

    "use_grad_clip": True,
    "grad_clip_max_norm": 1.0,
    },

    "evaluation": {
        "nx": 1000,
        "ny": 800,
    },
}


# -----------------------------
# Run directory / saving
# -----------------------------
def make_run_dir(config):
    root = Path(config["output_root"])
    root.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"paul_trap_pinn_{timestamp}"
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
world_voltage = bc["world_voltage"]


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
    n_each = n // 4

    xb = np.random.uniform(xmin, xmax, n_each)
    xt = np.random.uniform(xmin, xmax, n_each)
    yl = np.random.uniform(ymin, ymax, n_each)
    yr = np.random.uniform(ymin, ymax, n_each)

    bottom = np.column_stack([xb, np.full(n_each, ymin)])
    top = np.column_stack([xt, np.full(n_each, ymax)])
    left = np.column_stack([np.full(n_each, xmin), yl])
    right = np.column_stack([np.full(n_each, xmax), yr])

    xy = np.vstack([bottom, top, left, right]).astype(np.float32)
    v = np.full((len(xy), 1), world_voltage, dtype=np.float32)

    return xy, v


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
# PINN
# -----------------------------
class MLP(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()

        layers = [nn.Linear(2, width), nn.Tanh()]

        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]

        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


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


def pinn_loss(model):
    train_cfg = config["training"]

    xy_int = to_tensor(sample_interior(train_cfg["n_int"]), requires_grad=True)
    phi_int = model(xy_int)
    loss_pde = torch.mean(laplacian(phi_int, xy_int)**2)

    xy_world, v_world = sample_world_boundary(train_cfg["n_world"])
    xy_world = to_tensor(xy_world)
    v_world = to_tensor(v_world)
    loss_world = torch.mean((model(xy_world) - v_world)**2)

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

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    mask = inside_any_electrode(xy).reshape(ny, nx)

    xy_t = to_tensor(xy, requires_grad=True)
    phi = model(xy_t)

    grad_phi = torch.autograd.grad(
        phi,
        xy_t,
        grad_outputs=torch.ones_like(phi),
        create_graph=False,
    )[0]

    grad_phi_m = grad_phi / geom_unit_to_m
    grad2 = torch.sum(grad_phi_m**2, dim=1, keepdim=True)

    psi_J = scale * grad2
    psi_eV = psi_J / e_charge

    phi_np = phi.detach().numpy().reshape(ny, nx)
    psi_eV_np = psi_eV.detach().numpy().reshape(ny, nx)

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

def plot_and_save_results(run_dir, X, Y, phi, psi_eV, hist):
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

    plt.show()

# -----------------------------
# Run
# -----------------------------
run_dir = make_run_dir(config)
save_config(config, run_dir)
save_script_copy(run_dir)

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

model_cfg = config["model"]
model = MLP(width=model_cfg["width"], depth=model_cfg["depth"])

hist = train(model, run_dir)

X, Y, phi, psi_eV = evaluate_on_grid(model)

save_fields(run_dir, X, Y, phi, psi_eV)
plot_and_save_results(run_dir, X, Y, phi, psi_eV, hist)

print(f"\nSaved run to: {run_dir.resolve()}")
