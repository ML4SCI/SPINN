import os
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# -----------------------------
# Config
# -----------------------------
config = {
    "seed": 0,
    "output_root": "runs",

    "geometry": {
        "world_size_x": 0.5,
        "world_size_y": 0.5,
        "radius": 0.5e-1,
        "N_distance": 1.25e-1,
        "S_distance": 1.25e-1,
        "W_distance": 1.25e-1,
        "E_distance": 1.25e-1,
    },

    "model": {
        "width": 50,
        "depth": 6,
        "zero_last_layer": False,
    },

    "training": {
        "steps": 10000,
        "lr": 1e-3,
        "n_int": 4096,
        "n_world": 1024,
        "n_electrode": 2048,
        "print_every": 100,

        "use_lr_scheduler": True,
        "lr_scheduler": {
            "factor": 0.5,
            "patience": 500,
            "min_lr": 1e-6,
        },

        "use_grad_clip": True,
        "grad_clip_max_norm": 1.0,
    },

    "evaluation": {
        "nx": 250,
        "ny": 200,
        "plot_n_points": 5000,
        "quiver_stride": 12,
    },
}


# -----------------------------
# Run directory / saving
# -----------------------------
def make_run_dir(config):
    root = Path(config["output_root"])
    root.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"coordinate_projection_identity_{timestamp}"
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


# -----------------------------
# Sampling
# -----------------------------
def inside_any_electrode(xy):
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)

    for cx, cy, r in electrodes.values():
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

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

    return np.vstack([bottom, top, left, right]).astype(np.float32)


def sample_circle_boundary(cx, cy, r, n):
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    return np.column_stack([x, y]).astype(np.float32)


def sample_electrode_boundaries(n_per_electrode):
    pts = []

    for cx, cy, r in electrodes.values():
        pts.append(sample_circle_boundary(cx, cy, r, n_per_electrode))

    return np.vstack(pts).astype(np.float32)


def sample_training_points():
    train_cfg = config["training"]

    z_int = sample_interior(train_cfg["n_int"])
    z_world = sample_world_boundary(train_cfg["n_world"])
    z_electrode = sample_electrode_boundaries(train_cfg["n_electrode"])

    z = np.vstack([z_int, z_world, z_electrode]).astype(np.float32)

    return z


def to_tensor(x, requires_grad=False, device="cpu"):
    return torch.tensor(
        x,
        dtype=torch.float32,
        requires_grad=requires_grad,
        device=device,
    )


# -----------------------------
# Coordinate Projection Network
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, width=50, depth=6):
        super().__init__()

        layers = [nn.Linear(in_dim, width), nn.Tanh()]

        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]

        layers += [nn.Linear(width, out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CoordinateProjectionNet(nn.Module):
    """
    Eq. 13 style:

        NN_phi(z) = z + NNhat_phi(z)

    The network learns a displacement field.
    For identity initialization, we want NNhat_phi(z) -> 0.
    """
    def __init__(self, width=50, depth=6, zero_last_layer=False):
        super().__init__()
        self.displacement = MLP(2, 2, width=width, depth=depth)

        if zero_last_layer:
            last = self.displacement.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z):
        return z + self.displacement(z)


# -----------------------------
# Loss / training
# -----------------------------
def identity_loss(model, z):
    x_pred = model(z)
    return torch.mean(torch.sum((x_pred - z) ** 2, dim=1))


def mean_identity_distance_loss(model, z, eps=1e-12):
    x_pred = model(z)
    dist = torch.sqrt(torch.sum((x_pred - z) ** 2, dim=1) + eps)
    return torch.mean(dist)


def rms_identity_error(model, z):
    with torch.no_grad():
        x_pred = model(z)
        return torch.sqrt(torch.mean(torch.sum((x_pred - z) ** 2, dim=1)))


def grad_norm(model):
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2

    return total ** 0.5


def train(model, run_dir, device):
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
        z_np = sample_training_points()
        z = to_tensor(z_np, device=device)

        opt.zero_grad()

        loss = mean_identity_distance_loss(model, z)
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

        rms = rms_identity_error(model, z).item()

        row = [
            step,
            loss.item(),
            rms,
            current_lr,
            grad_pre_clip,
            grad_post_clip,
        ]

        if step % train_cfg["print_every"] == 0:
            hist.append(row)

            print(
                f"{step:6d} | "
                f"loss={loss.item():.3e} | "
                f"rms={rms:.3e} | "
                f"lr={current_lr:.3e} | "
                f"grad={grad_pre_clip:.3e}->{grad_post_clip:.3e}"
            )

    hist = np.array(hist)

    np.savetxt(
        run_dir / "training_history.csv",
        hist,
        delimiter=",",
        header="step,mean_identity_distance_loss,rms_identity_error,lr,grad_pre_clip,grad_post_clip",
        comments="",
    )

    torch.save(model.state_dict(), run_dir / "model_state_dict.pt")

    return hist


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_on_grid(model, device):
    eval_cfg = config["evaluation"]

    nx = eval_cfg["nx"]
    ny = eval_cfg["ny"]

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    z = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    mask = inside_any_electrode(z).reshape(ny, nx)

    z_t = to_tensor(z, device=device)

    with torch.no_grad():
        x_pred = model(z_t).cpu().numpy()

    U = (x_pred[:, 0] - z[:, 0]).reshape(ny, nx)
    V = (x_pred[:, 1] - z[:, 1]).reshape(ny, nx)
    D = np.sqrt(U ** 2 + V ** 2)

    U = np.where(mask, np.nan, U)
    V = np.where(mask, np.nan, V)
    D = np.where(mask, np.nan, D)

    return X, Y, U, V, D


def save_projection_data(run_dir, model, device):
    z = sample_training_points()

    with torch.no_grad():
        x_pred = model(to_tensor(z, device=device)).cpu().numpy()

    np.savez(
        run_dir / "projection_data.npz",
        z_reference=z,
        x_pred=x_pred,
        displacement=x_pred - z,
    )


# -----------------------------
# Plotting
# -----------------------------
def add_geometry(ax, linewidth=1.5):
    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        linewidth=linewidth,
    )
    ax.add_patch(rect)

    for name, (cx, cy, r) in electrodes.items():
        circle = Circle(
            (cx, cy),
            r,
            fill=False,
            linewidth=linewidth,
        )
        ax.add_patch(circle)


def predicted_circle(model, cx, cy, r, device, n=400):
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.column_stack([
        cx + r * np.cos(theta),
        cy + r * np.sin(theta),
    ]).astype(np.float32)

    with torch.no_grad():
        x_pred = model(to_tensor(z, device=device)).cpu().numpy()

    return z, x_pred


def predicted_world_boundary(model, device, n=400):
    bottom_x = np.linspace(xmin, xmax, n)
    top_x = np.linspace(xmin, xmax, n)
    left_y = np.linspace(ymin, ymax, n)
    right_y = np.linspace(ymin, ymax, n)

    z = np.vstack([
        np.column_stack([bottom_x, np.full(n, ymin)]),
        np.column_stack([np.full(n, xmax), right_y]),
        np.column_stack([top_x[::-1], np.full(n, ymax)]),
        np.column_stack([np.full(n, xmin), left_y[::-1]]),
    ]).astype(np.float32)

    with torch.no_grad():
        x_pred = model(to_tensor(z, device=device)).cpu().numpy()

    return z, x_pred


def plot_loss_history(run_dir, hist):
    fig, ax1 = plt.subplots()

    ax1.semilogy(hist[:, 0], hist[:, 1], label="identity loss")
    ax1.semilogy(hist[:, 0], hist[:, 2], label="RMS coordinate error")

    ax1.set_xlabel("step")
    ax1.set_ylabel("loss / error")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.semilogy(hist[:, 0], hist[:, 3], linestyle="--", label="lr")
    ax2.semilogy(hist[:, 0], hist[:, 4], linestyle=":", label="grad pre")
    ax2.semilogy(hist[:, 0], hist[:, 5], linestyle="-.", label="grad post")
    ax2.set_ylabel("lr / gradient norm")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(run_dir / "loss_history.png", dpi=300)

    return fig, ax1


def plot_displacement_field(run_dir, X, Y, U, V, D):
    stride = config["evaluation"]["quiver_stride"]

    fig, ax = plt.subplots()

    field_masked = np.ma.masked_invalid(D)

    c = ax.contourf(X, Y, field_masked, levels=100)
    fig.colorbar(c, ax=ax, label=r"$|NN_\phi(z)-z|$")

    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        U[::stride, ::stride],
        V[::stride, ::stride],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002,
    )

    add_geometry(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Residual Displacement Field")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(run_dir / "displacement_field.png", dpi=300)

    return fig, ax


def plot_geometry_overlay(run_dir, model, device):
    fig, ax = plt.subplots()

    z_world, x_world = predicted_world_boundary(model, device)
    ax.plot(z_world[:, 0], z_world[:, 1], linestyle="--", label="reference world")
    ax.plot(x_world[:, 0], x_world[:, 1], label="projected world")

    first = True
    for name, (cx, cy, r) in electrodes.items():
        z_el, x_el = predicted_circle(model, cx, cy, r, device)

        ax.plot(
            z_el[:, 0],
            z_el[:, 1],
            linestyle="--",
            label="reference electrodes" if first else None,
        )

        ax.plot(
            x_el[:, 0],
            x_el[:, 1],
            label="projected electrodes" if first else None,
        )

        first = False

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Reference Geometry vs Projected Geometry")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    fig.tight_layout()
    fig.savefig(run_dir / "geometry_overlay.png", dpi=300)

    return fig, ax


def plot_point_cloud_projection(run_dir, model, device):
    n = config["evaluation"]["plot_n_points"]

    z = sample_training_points()
    if len(z) > n:
        idx = np.random.choice(len(z), n, replace=False)
        z = z[idx]

    with torch.no_grad():
        x_pred = model(to_tensor(z, device=device)).cpu().numpy()

    # One random color value per point.
    # The same color array is used for z and NN_phi(z), so corresponding
    # points have matching colors in both panels.
    point_colors = np.random.rand(len(z))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    axes[0].scatter(
        z[:, 0],
        z[:, 1],
        c=point_colors,
        s=2,
        alpha=0.7,
        cmap="hsv",
        edgecolors="none",
    )
    add_geometry(axes[0])
    axes[0].set_title("Reference points z")

    axes[1].scatter(
        x_pred[:, 0],
        x_pred[:, 1],
        c=point_colors,
        s=2,
        alpha=0.7,
        cmap="hsv",
        edgecolors="none",
    )
    add_geometry(axes[1])
    axes[1].set_title(r"Projected points $NN_\phi(z)$")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(run_dir / "point_cloud_projection.png", dpi=300)

    return fig, axes


def plot_and_save_results(run_dir, model, hist, device):
    plot_loss_history(run_dir, hist)

    X, Y, U, V, D = evaluate_on_grid(model, device)
    plot_displacement_field(run_dir, X, Y, U, V, D)

    plot_geometry_overlay(run_dir, model, device)
    plot_point_cloud_projection(run_dir, model, device)

    plt.show()


# -----------------------------
# Run
# -----------------------------
run_dir = make_run_dir(config)
save_config(config, run_dir)
save_script_copy(run_dir)

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_cfg = config["model"]
model = CoordinateProjectionNet(
    width=model_cfg["width"],
    depth=model_cfg["depth"],
    zero_last_layer=model_cfg["zero_last_layer"],
).to(device)

hist = train(model, run_dir, device)

save_projection_data(run_dir, model, device)
plot_and_save_results(run_dir, model, hist, device)

print(f"\nSaved run to: {run_dir.resolve()}")