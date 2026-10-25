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

    "pixel": {
        "grid_spacing_x": 0.01,
        "grid_spacing_y": 0.01,
        "num_grids": 16,
        "offset_x": "auto",
        "offset_y": "auto",
        "channels": 32,
        # Interpolation kernel:
        #   "cosine"   -> k(x) = 0.5*(1 - cos(pi*x)); infinitely differentiable,
        #                 required for 2nd-order operators such as the Laplacian.
        #   "bilinear" -> k(x) = x; only valid for 1st-order PDEs.
        "kernel": "cosine",
        # Std of the normal initialization of the cell representations.
        "init_std": 0.01,
    },

    "model": {
        "width": 128,
        "depth": 2,
    },

    "hard_electrode_bc": {
        "enabled": False,
        # Width in geometry units (cm) controlling how quickly the trainable
        # correction turns on away from the electrode surfaces.
        "envelope_width": 0.1, #0.025
        # Small stabilizer used only in the denominator of the smooth
        # product-distance boundary extension.
        "distance_eps": 1e-18,
    },

    "training": {
        # Optimizer options: "adam", "lbfgs", or "adam_then_lbfgs".
        "optimizer": "adam_then_lbfgs",
        "adam_steps": 1000,
        "lbfgs_steps": 2500,
        "steps": 5000,
        "lr": 1e-3,
        "lbfgs_lr": 1.e-3,
        "lbfgs_max_iter_per_step": 5,
        "lbfgs_history_size": 50,
        "n_int": 4096,
        "n_world": 1024,
        "n_electrode": 2048,
        # With hard electrode BCs, bc_weight/world_bc_weight applies only to the
        # outer boundary condition. The electrode BC diagnostic is not included
        # in the optimization loss.
        "bc_weight": 100.0,
        "world_bc_weight": 100.0,
        "pde_weight": 1.0,
        "print_every": 100,

        # Reference voltage used to nondimensionalize the potential during
        # training (the network solves for u = Phi / potential_scale). "auto"
        # uses the largest boundary voltage magnitude. Keeping the training
        # targets O(1) is essential: with raw volts (e.g. 300 V) the BC loss and
        # its gradient blow up, gradient clipping throttles the step, and the PDE
        # residual drifts up while the BC loss stalls.
        "potential_scale": "auto",

        "use_lr_scheduler": True,
        "lr_scheduler": {
            "factor": 0.5,
            "patience": 300,
            "min_lr": 1e-6,
        },

    "use_grad_clip": True,
    "grad_clip_max_norm": 1.e3,
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
# Potential normalization
# -----------------------------
_candidate_scales = [abs(v) for v in rf_voltages.values()]
if world_bc_type == "dirichlet":
    _candidate_scales.append(abs(world_bc_value))

v_ref = config["training"].get("potential_scale", "auto")
if v_ref == "auto":
    v_ref = max(_candidate_scales + [1.0])
v_ref = float(v_ref)

# Normalized boundary targets actually fed to the training losses.
rf_voltages_n = {name: v / v_ref for name, v in rf_voltages.items()}
world_bc_value_n = world_bc_value / v_ref  # V/cm for Neumann, V for Dirichlet


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

    values = np.full((len(xy), 1), world_bc_value_n, dtype=np.float32)

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
        v = np.full((n_per_electrode, 1), rf_voltages_n[name], dtype=np.float32)

        xs.append(xy)
        vs.append(v)

    return np.vstack(xs), np.vstack(vs)


def to_tensor(x, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


# -----------------------------
# PINN
# -----------------------------
class PIXEL(nn.Module):

    def __init__(self, domain, pixel_cfg, model_cfg):
        super().__init__()

        xmin, xmax, ymin, ymax = domain

        self.dx = float(pixel_cfg["grid_spacing_x"])
        self.dy = float(pixel_cfg["grid_spacing_y"])
        self.M = int(pixel_cfg["num_grids"])

        offx = pixel_cfg["offset_x"]
        offy = pixel_cfg["offset_y"]
        self.offx = self.dx / self.M if offx == "auto" else float(offx)
        self.offy = self.dy / self.M if offy == "auto" else float(offy)
        self.c = int(pixel_cfg["channels"])
        self.kernel = pixel_cfg.get("kernel", "cosine").lower()

        if self.kernel not in {"cosine", "bilinear"}:
            raise ValueError("pixel['kernel'] must be 'cosine' or 'bilinear'.")

        self.x0 = xmin - self.dx
        self.y0 = ymin - self.dy

        max_x0 = self.x0 + (self.M - 1) * self.offx
        max_y0 = self.y0 + (self.M - 1) * self.offy
        min_x0 = self.x0 + min(0.0, (self.M - 1) * self.offx)
        min_y0 = self.y0 + min(0.0, (self.M - 1) * self.offy)

        self.Nx = int(np.ceil((xmax - min_x0) / self.dx)) + 3
        self.Ny = int(np.ceil((ymax - min_y0) / self.dy)) + 3

        # If offsets are positive, the most shifted origin is largest. The +3
        # above leaves enough nodes for both the base and most-shifted grids.
        while max_x0 + (self.Nx - 1) * self.dx < xmax + self.dx:
            self.Nx += 1
        while max_y0 + (self.Ny - 1) * self.dy < ymax + self.dy:
            self.Ny += 1

        if self.Nx < 2 or self.Ny < 2:
            raise ValueError(
                "PIXEL grid has fewer than 2 nodes per axis inside the domain; "
                "decrease grid_spacing / offset or the number of grids."
            )

        # Per-grid origins (node 0 of grid g), stored as buffers.
        g_idx = torch.arange(self.M, dtype=torch.float32)
        self.register_buffer("gx0", self.x0 + self.offx * g_idx)
        self.register_buffer("gy0", self.y0 + self.offy * g_idx)

        # Trainable cell representations: (M, c, Ny, Nx).
        C = torch.empty(self.M, self.c, self.Ny, self.Nx)
        nn.init.normal_(C, std=float(pixel_cfg["init_std"]))
        self.C = nn.Parameter(C)

        # Small MLP: summed features (c) -> potential (1).
        width = model_cfg["width"]
        depth = model_cfg["depth"]

        layers = [nn.Linear(self.c, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.mlp = nn.Sequential(*layers)

    def _weights(self, frac):
        # Weight assigned to the "far" node (index i0 + 1); the "near" node
        # (index i0) receives 1 - w since the kernels form a partition of unity.
        if self.kernel == "cosine":
            return 0.5 * (1.0 - torch.cos(np.pi * frac))
        return frac  # bilinear

    def interpolate(self, xy):
        """Return the summed multigrid feature (N, c) for query points xy."""
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        feat = xy.new_zeros(xy.shape[0], self.c)

        for g in range(self.M):
            fx = (x - self.gx0[g]) / self.dx
            fy = (y - self.gy0[g]) / self.dy

            # Integer node indices (piecewise constant -> detached from autograd).
            i0 = torch.floor(fx).detach().long().clamp(0, self.Nx - 2)
            j0 = torch.floor(fy).detach().long().clamp(0, self.Ny - 2)

            fracx = fx - i0.to(fx.dtype)
            fracy = fy - j0.to(fy.dtype)

            wx1 = self._weights(fracx)
            wx0 = 1.0 - wx1
            wy1 = self._weights(fracy)
            wy0 = 1.0 - wy1

            i0 = i0.squeeze(1)
            j0 = j0.squeeze(1)

            Cg = self.C[g].reshape(self.c, -1)  # (c, Ny*Nx)

            def node(jj, ii):
                idx = (jj * self.Nx + ii)
                return Cg.index_select(1, idx).t()  # (N, c)

            c00 = node(j0, i0)
            c01 = node(j0, i0 + 1)
            c10 = node(j0 + 1, i0)
            c11 = node(j0 + 1, i0 + 1)

            feat = feat + (
                wy0 * wx0 * c00
                + wy0 * wx1 * c01
                + wy1 * wx0 * c10
                + wy1 * wx1 * c11
            )

        return feat

    def grid_points(self, grid_index):
        """Node coordinates (X, Y meshgrids) of grid `grid_index` for plotting."""
        xs = self.x0 + self.offx * grid_index + np.arange(self.Nx) * self.dx
        ys = self.y0 + self.offy * grid_index + np.arange(self.Ny) * self.dy
        return np.meshgrid(xs, ys)

    def forward(self, xy):
        return self.mlp(self.interpolate(xy))


def electrode_signed_distances_torch(xy):
    """Signed distances d_i = distance_to_center - radius for each electrode.

    The physical PDE domain is outside the electrodes, so d_i >= 0 for valid
    interior collocation points. On electrode i, d_i = 0.
    """
    ds = []
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    for name in electrodes:
        cx, cy, r = electrodes[name]
        d = torch.sqrt((x - cx)**2 + (y - cy)**2 + 1e-24) - r
        ds.append(d)

    return torch.cat(ds, dim=1)


def hard_bc_extension(xy, eps):
    """Smooth extension of electrode voltages into the domain."""
    d = electrode_signed_distances_torch(xy)
    voltages = torch.tensor(
        [rf_voltages_n[name] for name in electrodes],
        dtype=xy.dtype,
        device=xy.device,
    ).view(1, -1)

    d2 = d**2
    products = []
    for i in range(d2.shape[1]):
        mask = [j for j in range(d2.shape[1]) if j != i]
        products.append(torch.prod(d2[:, mask], dim=1, keepdim=True))
    P = torch.cat(products, dim=1)

    lam = P / (torch.sum(P, dim=1, keepdim=True) + eps)
    return torch.sum(lam * voltages, dim=1, keepdim=True)


def hard_bc_envelope(xy, width):
    """Multiplicative envelope that is exactly zero on all electrodes."""
    d = electrode_signed_distances_torch(xy)
    return torch.prod(torch.tanh(d / width), dim=1, keepdim=True)


class HardElectrodeBCPIXEL(nn.Module):
    """PIXEL model with exact electrode Dirichlet boundary conditions."""

    def __init__(self, core, hard_cfg):
        super().__init__()
        self.core = core
        self.envelope_width = float(hard_cfg["envelope_width"])
        self.distance_eps = float(hard_cfg["distance_eps"])

    def forward(self, xy):
        u_bc = hard_bc_extension(xy, self.distance_eps)
        g_env = hard_bc_envelope(xy, self.envelope_width)
        return u_bc + g_env * self.core(xy)

    def raw_correction(self, xy):
        return self.core(xy)


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


def world_boundary_loss(model, xy_world, n_world, values_world):
    if world_bc_type == "dirichlet":
        phi_world = model(xy_world)
        return torch.mean((phi_world - values_world)**2)

    if world_bc_type == "neumann":
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


def sample_training_batch():
    train_cfg = config["training"]

    xy_int = to_tensor(sample_interior(train_cfg["n_int"]), requires_grad=True)

    xy_world, n_world, values_world = sample_world_boundary(train_cfg["n_world"])
    xy_world = to_tensor(
        xy_world,
        requires_grad=(world_bc_type == "neumann"),
    )
    n_world = to_tensor(n_world)
    values_world = to_tensor(values_world)

    xy_el, v_el = sample_electrode_boundaries(train_cfg["n_electrode"])
    xy_el = to_tensor(xy_el)
    v_el = to_tensor(v_el)

    return xy_int, xy_world, n_world, values_world, xy_el, v_el


def pinn_loss(model, batch=None):
    train_cfg = config["training"]

    if batch is None:
        batch = sample_training_batch()

    xy_int, xy_world, n_world, values_world, xy_el, v_el = batch

    phi_int = model(xy_int)
    loss_pde = torch.mean(laplacian(phi_int, xy_int)**2)

    # Electrode Dirichlet BCs are enforced by construction, not by penalty.
    # loss_electrode is only a diagnostic and should be near floating point zero
    # when sampled exactly on the circular boundaries.
    loss_world = world_boundary_loss(model, xy_world, n_world, values_world)
    loss_electrode = torch.mean((model(xy_el) - v_el)**2)

    pde_weight = train_cfg.get("pde_weight", 1.0)
    world_weight = train_cfg.get("world_bc_weight", train_cfg.get("bc_weight", 100.0))
    loss = pde_weight * loss_pde + world_weight * loss_world

    return loss, loss_pde.detach(), loss_world.detach(), loss_electrode.detach()


def grad_norm(model):
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2

    return total ** 0.5


def _log_row(hist, step, loss, loss_pde, loss_world, loss_electrode, lr, grad_pre, grad_post, print_every):
    row = [step, loss, loss_pde, loss_world, loss_electrode, lr, grad_pre, grad_post]

    if step % print_every == 0:
        hist.append(row)
        print(
            f"{step:6d} | "
            f"loss={loss:.3e} | "
            f"pde={loss_pde:.3e} | "
            f"world={loss_world:.3e} | "
            f"electrode={loss_electrode:.3e} | "
            f"lr={lr:.3e} | "
            f"grad={grad_pre:.3e}->{grad_post:.3e}"
        )


def _adam_train(model, hist, start_step, n_steps):
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

    for k in range(n_steps):
        step = start_step + k
        opt.zero_grad()
        loss, loss_pde, loss_world, loss_electrode = pinn_loss(model)
        loss.backward()

        grad_pre = grad_norm(model)
        if train_cfg.get("use_grad_clip", False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=train_cfg["grad_clip_max_norm"],
            )
        grad_post = grad_norm(model)

        opt.step()
        lr = opt.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(loss.item())

        _log_row(
            hist, step, loss.item(), loss_pde.item(), loss_world.item(), loss_electrode.item(),
            lr, grad_pre, grad_post, train_cfg["print_every"]
        )

    return start_step + n_steps


def _lbfgs_train(model, hist, start_step, n_steps):
    train_cfg = config["training"]
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=train_cfg["lbfgs_lr"],
        max_iter=train_cfg["lbfgs_max_iter_per_step"],
        history_size=train_cfg["lbfgs_history_size"],
        line_search_fn="strong_wolfe",
    )

    for k in range(n_steps):
        step = start_step + k
        batch = sample_training_batch()
        metrics = {}

        def closure():
            opt.zero_grad()
            loss, loss_pde, loss_world, loss_electrode = pinn_loss(model, batch=batch)
            loss.backward()

            grad_pre = grad_norm(model)
            grad_post = grad_pre
            if train_cfg.get("use_grad_clip", False):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=train_cfg["grad_clip_max_norm"],
                )
                grad_post = grad_norm(model)

            metrics["loss"] = loss.item()
            metrics["pde"] = loss_pde.item()
            metrics["world"] = loss_world.item()
            metrics["electrode"] = loss_electrode.item()
            metrics["grad_pre"] = grad_pre
            metrics["grad_post"] = grad_post
            return loss

        opt.step(closure)
        lr = opt.param_groups[0]["lr"]

        _log_row(
            hist, step, metrics["loss"], metrics["pde"], metrics["world"], metrics["electrode"],
            lr, metrics["grad_pre"], metrics["grad_post"],
            train_cfg["print_every"]
        )

    return start_step + n_steps


def train(model, run_dir):
    train_cfg = config["training"]
    optimizer_name = train_cfg.get("optimizer", "adam").lower()

    print(
        f"World BC: {world_bc_type}, "
        f"value={world_bc_value:g}"
    )
    print(f"Optimizer: {optimizer_name}")

    hist = []
    step = 0

    if optimizer_name == "adam":
        step = _adam_train(model, hist, step, train_cfg["steps"])
    elif optimizer_name == "lbfgs":
        step = _lbfgs_train(model, hist, step, train_cfg["lbfgs_steps"])
    elif optimizer_name == "adam_then_lbfgs":
        step = _adam_train(model, hist, step, train_cfg["adam_steps"])
        step = _lbfgs_train(model, hist, step, train_cfg["lbfgs_steps"])
    else:
        raise ValueError("training['optimizer'] must be 'adam', 'lbfgs', or 'adam_then_lbfgs'.")

    hist = np.array(hist)

    np.savetxt(
        run_dir / "training_history.csv",
        hist,
        delimiter=",",
        header="step,total_loss,pde_loss,world_bc_loss,electrode_bc_diagnostic,lr,grad_pre_clip,grad_post_clip",
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
    # The model predicts the normalized potential u = Phi / v_ref; convert back
    # to physical volts here so downstream fields/plots are in real units.
    phi = v_ref * model(xy_t)

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

def _draw_domain_box(ax):
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        color="black",
        linewidth=1.5,
        zorder=5,
    )


def plot_grid_points(run_dir, model):

    M = model.M
    colors = plt.cm.viridis(np.linspace(0, 1, M))

    # --- Full domain -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    for g in range(M):
        GX, GY = model.grid_points(g)
        ax.scatter(
            GX,
            GY,
            s=4,
            color=colors[g],
            label=f"grid {g}",
            zorder=3,
        )

    _draw_domain_box(ax)
    add_electrode_patches(ax)

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(
        f"PIXEL grid nodes "
        f"({M} grids, spacing=({model.dx:g}, {model.dy:g}), "
        f"offset=({model.offx:g}, {model.offy:g}))"
    )
    ax.set_aspect("equal", adjustable="box")
    if M <= 12:
        ax.legend(loc="upper right", markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "grid_points.png", dpi=300)

    # --- Zoom on a few cells near the base origin --------------------------
    fig_z, ax_z = plt.subplots(figsize=(7, 7))

    zx0 = xmin
    zy0 = ymin
    zoom_w = 3.0 * model.dx
    zoom_h = 3.0 * model.dy

    for g in range(M):
        GX, GY = model.grid_points(g)
        ax_z.scatter(
            GX,
            GY,
            s=40,
            color=colors[g],
            label=f"grid {g}",
            zorder=3,
        )

    ax_z.set_xlim(zx0 - 0.5 * model.dx, zx0 + zoom_w)
    ax_z.set_ylim(zy0 - 0.5 * model.dy, zy0 + zoom_h)
    ax_z.set_xlabel("x [cm]")
    ax_z.set_ylabel("y [cm]")
    ax_z.set_title("PIXEL grid nodes (zoom: per-grid offset)")
    ax_z.set_aspect("equal", adjustable="box")
    if M <= 12:
        ax_z.legend(loc="upper right", markerscale=1.2, fontsize=8)
    fig_z.tight_layout()
    fig_z.savefig(run_dir / "grid_points_zoom.png", dpi=300)

    return fig, ax


def plot_loss_history(run_dir, hist):
    fig, ax1 = plt.subplots()

    ax1.semilogy(hist[:, 0], hist[:, 1], label="total")
    ax1.semilogy(hist[:, 0], hist[:, 2], label="pde")
    ax1.semilogy(hist[:, 0], hist[:, 3], label="world bc")
    ax1.semilogy(hist[:, 0], hist[:, 4], label="electrode bc diagnostic")

    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()

    if hist.shape[1] > 5:
        ax2.semilogy(hist[:, 0], hist[:, 5], linestyle="--", label="lr")

    if hist.shape[1] > 7:
        ax2.semilogy(hist[:, 0], hist[:, 6], linestyle=":", label="grad pre")
        ax2.semilogy(hist[:, 0], hist[:, 7], linestyle="-.", label="grad post")

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
pixel_cfg = config["pixel"]
domain = (xmin, xmax, ymin, ymax)
core_model = PIXEL(domain, pixel_cfg, model_cfg)

if config.get("hard_electrode_bc", {}).get("enabled", True):
    model = HardElectrodeBCPIXEL(core_model, config["hard_electrode_bc"])
else:
    model = core_model

print(
    f"PIXEL grid: {core_model.M} grids of {core_model.Ny}x{core_model.Nx} nodes, "
    f"channels={core_model.c}, kernel={core_model.kernel}"
)
print("Electrode BC mode: hard ansatz" if isinstance(model, HardElectrodeBCPIXEL) else "Electrode BC mode: soft penalty")
print(f"Potential normalization: v_ref = {v_ref:g} V (network solves u = Phi/v_ref)")

# Visualize the grid-node locations before training.
plot_grid_points(run_dir, core_model)

hist = train(model, run_dir)

X, Y, phi, psi_eV = evaluate_on_grid(model)

save_fields(run_dir, X, Y, phi, psi_eV)
plot_and_save_results(run_dir, X, Y, phi, psi_eV, hist)

print(f"\nSaved run to: {run_dir.resolve()}")
