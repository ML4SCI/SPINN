import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.path import Path as MplPath


# -----------------------------
# Config
# -----------------------------
config = {
    "seed": 0,
    "output_root": "runs",

    # Set these to your already-trained model files.
    # These files are only loaded. They are never overwritten.
    "input_models": {
        "use_existing_PINN": True,
        "use_existing_CPN": True,
        "pinn_state_dict": "../Vanilla_Paul_Trap_PINN/runs/paul_trap_pinn_20260618_170346/model_state_dict.pt",
        "cpn_state_dict": "../Coordinate_Projection_Network/runs/coordinate_projection_identity_20260618_155116/model_state_dict.pt",
    },

    "physics": {
        "e_charge": 1.602e-19,      # elementary charge [C]
        "amu": 1.66054e-27,         # atomic mass unit [kg]
        "ion_mass_amu": 40.0,       # Ca40 ion
        "rf_frequency_hz": 10.2e6,   # RF frequency [Hz]
        "geom_unit_to_m": 1e-2,      # geometry units are cm
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
        # For Neumann, value has units of volts per geometry unit. Here geometry units are cm.
        "world": {
            "type": "neumann",
            "value": 0.0,
        },
        "rf_voltages": {
            "North": 0.0,
            "South": 0.0,
            "West": 300.0,
            "East": 300.0,
        },
    },

    "pinn_model": {
        "width": 128,
        "depth": 4,
    },

    "cpn_model": {
        "width": 50,
        "depth": 6,
        "zero_last_layer": False,
    },

    "coupled_training": {
        "steps": 10000,
        "lr": 1e-3,
        "n_int": 4096,
        "n_world": 1024,
        "n_electrode": 2048,
        "n_query": 2048,
        "query_half_width": 0.04,

        "pde_weight": 1.0,
        "bc_weight": 100.0,
        "pseudo_weight": 100.0,

        # Electrode geometry regularization for the CPN. These terms discourage
        # the projected electrodes from solving the pseudopotential objective by
        # simply translating or uniformly growing/shrinking.
        "electrode_geometry_penalty": {
            # Per-electrode geometric penalties for the CPN.
            #
            # translation:
            #   Keeps each projected electrode centered near its known reference
            #   center from the electrodes dictionary.
            #
            # scale:
            #   Keeps each projected electrode's pointwise radius near its
            #   original radius. This discourages pure growth/shrinkage.
            #
            # smoothness:
            #   Keeps neighboring boundary points from developing very different
            #   displacements. This helps suppress twisting/folding artifacts.
            "use_translation_penalty": True,
            "translation_weight": 1.0e6,

            "use_scale_penalty": True,
            "scale_weight": 1.0e2,

            "use_smoothness_penalty": False,
            "smoothness_weight": 1.0e3,

            "n_points_per_electrode": 256,
        },

        "pseudo_roi_radius": 0.07,
        "pseudo_roi_transition": 0.01,

        # Leave as "auto" to use the ideal hyperbolic trap estimate from the
        # electrode voltages and inner electrode distance. Units: eV / cm^2.
        "pseudo_K_eV_per_geom_unit2": 3e3, # "auto" or float

        # Usually useful options:
        #   train both networks together: True, True
        #   freeze the PINN and optimize only geometry: False, True
        #   freeze geometry and re-fit field only: True, False
        "train_field_net": True,
        "train_shape_net": True,

        "print_every": 100,

        # Save the same diagnostic plots normally made at the end of training
        # every N epochs. Set to None or 0 to disable periodic plotting.
        "plot_every": 1000,
        "plot_at_epoch_zero": False,
        "plot_final": True,

        "use_lr_scheduler": True,
        "lr_scheduler": {
            "factor": 0.9,
            "patience": 1000,
            "min_lr": 1e-6,
        },

        "use_grad_clip": True,
        "grad_clip": {
            # Set either value to None to disable clipping for that network.
            "field_max_norm": 10.0,
            "shape_max_norm": 100, #Set to None to disable entirely
        },
    },

    "evaluation": {
        "nx": 1000,
        "ny": 800,
        "chunk_size": 50000,
        "plot_n_points": 5000,
        "quiver_stride": 12,
        "pseudo_vmax_eV": 3e2,
    },
}


# -----------------------------
# Run directory / saving
# -----------------------------
def make_run_dir(config):
    root = Path(config["output_root"])
    root.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"paul_trap_cpn_pinn_coupled_{timestamp}"
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
# Physical constants
# -----------------------------
p = config["physics"]
e_charge = p["e_charge"]
amu = p["amu"]
M = p["ion_mass_amu"] * amu
Omega = 2 * np.pi * p["rf_frequency_hz"]
scale_J_per_E2 = e_charge**2 / (4 * M * Omega**2)
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
world_bc = bc.get("world", {"type": "dirichlet", "value": bc.get("world_voltage", 0.0)})


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


def sample_query_region(n):
    h = config["coupled_training"]["query_half_width"]
    pts = []
    while len(pts) < n:
        xy = np.column_stack([
            np.random.uniform(-h, h, n),
            np.random.uniform(-h, h, n),
        ])
        xy = xy[~inside_any_electrode(xy)]
        pts.extend(xy.tolist())
    return np.array(pts[:n], dtype=np.float32)


def sample_world_boundary(n):
    n_each = n // 4

    # Avoid exact corners. For Neumann BCs the outward normal is ambiguous
    # exactly at a rectangular corner.
    eps = 1e-6
    xb = np.random.uniform(xmin + eps, xmax - eps, n_each)
    xt = np.random.uniform(xmin + eps, xmax - eps, n_each)
    yl = np.random.uniform(ymin + eps, ymax - eps, n_each)
    yr = np.random.uniform(ymin + eps, ymax - eps, n_each)

    bottom = np.column_stack([xb, np.full(n_each, ymin)])
    top = np.column_stack([xt, np.full(n_each, ymax)])
    left = np.column_stack([np.full(n_each, xmin), yl])
    right = np.column_stack([np.full(n_each, xmax), yr])

    xy = np.vstack([bottom, top, left, right]).astype(np.float32)
    normals = np.vstack([
        np.tile([0.0, -1.0], (n_each, 1)),
        np.tile([0.0,  1.0], (n_each, 1)),
        np.tile([-1.0, 0.0], (n_each, 1)),
        np.tile([ 1.0, 0.0], (n_each, 1)),
    ]).astype(np.float32)
    values = np.full((len(xy), 1), float(world_bc["value"]), dtype=np.float32)
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


def sample_cpn_plot_points():
    eval_cfg = config["evaluation"]
    n = eval_cfg["plot_n_points"]
    z_int = sample_interior(config["coupled_training"]["n_int"])
    z_world, _, _ = sample_world_boundary(config["coupled_training"]["n_world"])
    z_el, _ = sample_electrode_boundaries(config["coupled_training"]["n_electrode"])
    z = np.vstack([z_int, z_world, z_el]).astype(np.float32)
    if len(z) > n:
        z = z[np.random.choice(len(z), n, replace=False)]
    return z


def to_tensor(x, device, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, device=device, requires_grad=requires_grad)


# -----------------------------
# PINN model
# -----------------------------
class FieldMLP(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


# -----------------------------
# Coordinate Projection Network
# -----------------------------
class CPNMLP(nn.Module):
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
    def __init__(self, width=50, depth=6, zero_last_layer=False):
        super().__init__()
        self.displacement = CPNMLP(2, 2, width=width, depth=depth)
        if zero_last_layer:
            last = self.displacement.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z):
        return z + self.displacement(z)


# -----------------------------
# Loading helpers
# -----------------------------
def load_state_dict_safely(model, path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model


# -----------------------------
# Weighting constraints
# -----------------------------
def roi_soft_weight(xy, r_cut=0.05, transition=0.01):
    r = torch.sqrt(torch.sum(xy**2, dim=1, keepdim=True))
    return torch.sigmoid((r_cut - r) / transition)

def weighted_pseudopotential_loss(
    psi_model,
    psi_target,
    xy,
    r_cut=0.05,
    transition=0.01,
):
    w = roi_soft_weight(xy, r_cut=r_cut, transition=transition)
    err2 = (psi_model - psi_target) ** 2
    return torch.sum(w * err2) / (torch.sum(w) + 1e-12)


def deterministic_electrode_boundary_points(n_per_electrode):
    theta = np.linspace(0.0, 2.0 * np.pi, n_per_electrode, endpoint=False)
    pts = []
    names = []

    for name, (cx, cy, r) in electrodes.items():
        xy = np.column_stack([
            cx + r * np.cos(theta),
            cy + r * np.sin(theta),
        ]).astype(np.float32)
        pts.append(xy)
        names.append(name)

    return np.stack(pts, axis=0), names


def deterministic_single_electrode_points(name, n_points):
    cx, cy, r = electrodes[name]
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    xy = np.column_stack([
        cx + r * np.cos(theta),
        cy + r * np.sin(theta),
    ]).astype(np.float32)

    center = np.array([cx, cy], dtype=np.float32)

    return xy, center


def electrode_geometry_regularization_loss(shape_net, device):
    geom_cfg = config["coupled_training"].get("electrode_geometry_penalty", {})

    use_translation = geom_cfg.get("use_translation_penalty", False)
    use_scale = geom_cfg.get("use_scale_penalty", False)
    use_smoothness = geom_cfg.get("use_smoothness_penalty", False)

    zero = torch.zeros((), device=device)

    if not use_translation and not use_scale and not use_smoothness:
        return zero, zero, zero

    n_points = int(geom_cfg.get("n_points_per_electrode", 256))

    translation_losses = []
    scale_losses = []
    smoothness_losses = []

    for name in electrodes:
        z_np, center_np = deterministic_single_electrode_points(name, n_points)

        z = to_tensor(z_np, device=device, requires_grad=False)
        center = to_tensor(center_np[None, :], device=device, requires_grad=False)

        x = shape_net(z)

        # Translation penalty:
        # Keep this electrode's projected center near its known reference center.
        # The projected center is approximated as the mean of projected boundary
        # collocation points.
        x_center = torch.mean(x, dim=0, keepdim=True)
        loss_translation_e = torch.mean((x_center - center) ** 2)

        # Scale penalty:
        # Keep each projected point's radius from the projected center close to
        # the original electrode radius. This discourages pure growth/shrinkage
        # without forcing the electrode to remain perfectly circular.
        _, _, r = electrodes[name]
        r_ref = torch.tensor(float(r), device=device)

        x_rel = x - x_center
        x_r = torch.sqrt(torch.sum(x_rel**2, dim=1) + 1e-12)
        loss_scale_e = torch.mean(((x_r - r_ref) / (r_ref + 1e-12)) ** 2)

        # Smoothness penalty:
        # Neighboring points around one electrode should have similar
        # displacements. This suppresses twisting/folding loopholes that can
        # still satisfy centroid and scale constraints.
        displacement = x - z
        displacement_next = torch.roll(displacement, shifts=-1, dims=0)
        loss_smoothness_e = torch.mean(
            torch.sum((displacement_next - displacement) ** 2, dim=1)
        )

        translation_losses.append(loss_translation_e)
        scale_losses.append(loss_scale_e)
        smoothness_losses.append(loss_smoothness_e)

    loss_translation = torch.stack(translation_losses).mean()
    loss_scale = torch.stack(scale_losses).mean()
    loss_smoothness = torch.stack(smoothness_losses).mean()

    if not use_translation:
        loss_translation = zero
    if not use_scale:
        loss_scale = zero
    if not use_smoothness:
        loss_smoothness = zero

    return loss_translation, loss_scale, loss_smoothness

# -----------------------------
# Autograd helpers
# -----------------------------
def gradient(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplacian(phi, xy):
    grad_phi = gradient(phi, xy)
    phi_x = grad_phi[:, 0:1]
    phi_y = grad_phi[:, 1:2]
    phi_xx = gradient(phi_x, xy)[:, 0:1]
    phi_yy = gradient(phi_y, xy)[:, 1:2]
    return phi_xx + phi_yy


def electric_field(phi, xy):
    grad_phi = gradient(phi, xy)
    Ex = -grad_phi[:, 0:1]
    Ey = -grad_phi[:, 1:2]
    return Ex, Ey


def pseudopotential_eV(phi, xy):
    Ex, Ey = electric_field(phi, xy)
    Ex_m = Ex / geom_unit_to_m
    Ey_m = Ey / geom_unit_to_m
    E2 = Ex_m**2 + Ey_m**2
    psi_J = scale_J_per_E2 * E2
    return psi_J / e_charge


def automatic_ideal_pseudo_K():
    Vx = 0.5 * (rf_voltages["West"] + rf_voltages["East"])
    Vy = 0.5 * (rf_voltages["North"] + rf_voltages["South"])
    r0 = 0.25 * (
        E_distance - radius
        + W_distance - radius
        + N_distance - radius
        + S_distance - radius
    )
    A = 0.5 * (Vx - Vy) / (r0**2)
    return (scale_J_per_E2 / e_charge) * (2 * A / geom_unit_to_m)**2


def ideal_pseudopotential_eV(xy):
    cfg_val = config["coupled_training"]["pseudo_K_eV_per_geom_unit2"]
    K = automatic_ideal_pseudo_K() if cfg_val == "auto" else float(cfg_val)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return K * (x**2 + y**2)


# -----------------------------
# Coupled loss / training
# -----------------------------
def coupled_loss(field_net, shape_net, device):
    train_cfg = config["coupled_training"]

    z_int = to_tensor(sample_interior(train_cfg["n_int"]), device=device, requires_grad=True)
    x_int = shape_net(z_int)
    phi_int = field_net(x_int)
    loss_pde = torch.mean(laplacian(phi_int, x_int)**2)

    z_world_np, n_world_np, world_value_np = sample_world_boundary(train_cfg["n_world"])
    z_world = to_tensor(z_world_np, device=device)
    n_world = to_tensor(n_world_np, device=device)
    world_value = to_tensor(world_value_np, device=device)
    x_world = shape_net(z_world)
    phi_world = field_net(x_world)

    bc_type = world_bc["type"].lower()
    if bc_type == "dirichlet":
        loss_world = torch.mean((phi_world - world_value)**2)
    elif bc_type == "neumann":
        # Enforce dPhi/dn = value on the projected world boundary.
        # The derivative is taken with respect to the projected physical
        # coordinates. The outward normals correspond to the original
        # rectangular sides, which is appropriate while the world projection
        # remains close to identity.
        grad_world = gradient(phi_world, x_world)
        dphi_dn = torch.sum(grad_world * n_world, dim=1, keepdim=True)
        loss_world = torch.mean((dphi_dn - world_value)**2)
    else:
        raise ValueError(f"Unknown world BC type: {world_bc['type']}")

    z_el_np, v_el_np = sample_electrode_boundaries(train_cfg["n_electrode"])
    z_el = to_tensor(z_el_np, device=device)
    v_el = to_tensor(v_el_np, device=device)
    loss_electrode = torch.mean((field_net(shape_net(z_el)) - v_el)**2)
    loss_bc = loss_world + loss_electrode

    z_q = to_tensor(sample_query_region(train_cfg["n_query"]), device=device, requires_grad=True)
    x_q = shape_net(z_q)
    phi_q = field_net(x_q)
    # The field derivatives must be taken with respect to the projected
    # physical coordinates x_q. However, the optimization target is defined
    # on the reference design coordinates z_q. This is what allows the CPN to
    # learn a useful coordinate warp instead of simply moving both the model
    # and target together.
    psi_model = pseudopotential_eV(phi_q, x_q)
    psi_target = ideal_pseudopotential_eV(z_q)

    loss_pseudo = weighted_pseudopotential_loss(
        psi_model,
        psi_target,
        z_q,
        r_cut=train_cfg["pseudo_roi_radius"],
        transition=train_cfg["pseudo_roi_transition"],
    )

    loss_translation, loss_scale, loss_smoothness = electrode_geometry_regularization_loss(shape_net, device)
    geom_cfg = train_cfg.get("electrode_geometry_penalty", {})

    loss = (
        train_cfg["pde_weight"] * loss_pde
        + train_cfg["bc_weight"] * loss_bc
        + train_cfg["pseudo_weight"] * loss_pseudo
        + geom_cfg.get("translation_weight", 0.0) * loss_translation
        + geom_cfg.get("scale_weight", 0.0) * loss_scale
        + geom_cfg.get("smoothness_weight", 0.0) * loss_smoothness
    )

    return (
        loss,
        loss_pde.detach(),
        loss_bc.detach(),
        loss_pseudo.detach(),
        loss_translation.detach(),
        loss_scale.detach(),
        loss_smoothness.detach(),
    )


def grad_norm(models):
    total = 0.0
    for model in models:
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def model_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def history_array(hist):
    if len(hist) == 0:
        return np.empty((0, 15), dtype=np.float64)
    return np.array(hist, dtype=np.float64)


def save_training_history(run_dir, hist):
    hist_np = history_array(hist)
    np.savetxt(
        run_dir / "training_history.csv",
        hist_np,
        delimiter=",",
        header="step,total_loss,pde_loss,bc_loss,pseudo_loss,translation_loss,scale_loss,smoothness_loss,lr,grad_pre_clip,grad_post_clip,field_grad_pre_clip,field_grad_post_clip,shape_grad_pre_clip,shape_grad_post_clip",
        comments="",
    )


def maybe_make_epoch_plot(
    step,
    field_net,
    shape_net,
    run_dir,
    hist,
    device,
    force=False,
):
    train_cfg = config["coupled_training"]
    plot_every = train_cfg.get("plot_every", None)

    if not force:
        if plot_every is None or int(plot_every) <= 0:
            return
        if step % int(plot_every) != 0:
            return

    plot_dir = run_dir / f"epoch_{step:06d}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving diagnostic plots for epoch {step} -> {plot_dir}")
    plot_and_save_results(
        plot_dir,
        field_net,
        shape_net,
        history_array(hist),
        device,
        show=False,
    )
    plt.close("all")


def train_coupled(field_net, shape_net, run_dir, device):
    train_cfg = config["coupled_training"]

    for p in field_net.parameters():
        p.requires_grad_(train_cfg["train_field_net"])
    for p in shape_net.parameters():
        p.requires_grad_(train_cfg["train_shape_net"])

    params = [
        p for p in list(field_net.parameters()) + list(shape_net.parameters())
        if p.requires_grad
    ]
    if not params:
        raise ValueError("No trainable parameters. Enable train_field_net or train_shape_net.")

    opt = torch.optim.Adam(params, lr=train_cfg["lr"])

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

    if train_cfg.get("plot_at_epoch_zero", False):
        maybe_make_epoch_plot(
            0,
            field_net,
            shape_net,
            run_dir,
            hist,
            device,
            force=True,
        )

    for step in range(train_cfg["steps"]):
        opt.zero_grad()
        loss, loss_pde, loss_bc, loss_pseudo, loss_translation, loss_scale, loss_smoothness = coupled_loss(field_net, shape_net, device)
        loss.backward()

        grad_pre_clip = grad_norm([field_net, shape_net])
        field_grad_pre_clip = model_grad_norm(field_net)
        shape_grad_pre_clip = model_grad_norm(shape_net)

        if train_cfg.get("use_grad_clip", False):
            clip_cfg = train_cfg.get("grad_clip", {})
            field_max_norm = clip_cfg.get("field_max_norm", None)
            shape_max_norm = clip_cfg.get("shape_max_norm", None)

            if field_max_norm is not None and train_cfg["train_field_net"]:
                torch.nn.utils.clip_grad_norm_(field_net.parameters(), max_norm=float(field_max_norm))
            if shape_max_norm is not None and train_cfg["train_shape_net"]:
                torch.nn.utils.clip_grad_norm_(shape_net.parameters(), max_norm=float(shape_max_norm))

        field_grad_post_clip = model_grad_norm(field_net)
        shape_grad_post_clip = model_grad_norm(shape_net)
        grad_post_clip = grad_norm([field_net, shape_net])

        opt.step()
        current_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(loss.item())

        row = [
            step,
            loss.item(),
            loss_pde.item(),
            loss_bc.item(),
            loss_pseudo.item(),
            loss_translation.item(),
            loss_scale.item(),
            loss_smoothness.item(),
            current_lr,
            grad_pre_clip,
            grad_post_clip,
            field_grad_pre_clip,
            field_grad_post_clip,
            shape_grad_pre_clip,
            shape_grad_post_clip,
        ]

        hist.append(row)

        epoch = step + 1

        if step % train_cfg["print_every"] == 0:
            print(
                f"{step:6d} | "
                f"loss={loss.item():.3e} | "
                f"pde={loss_pde.item():.3e} | "
                f"bc={loss_bc.item():.3e} | "
                f"pseudo={loss_pseudo.item():.3e} | "
                f"trans={loss_translation.item():.3e} | "
                f"scale={loss_scale.item():.3e} | "
                f"smooth={loss_smoothness.item():.3e} | "
                f"lr={current_lr:.3e} | "
                f"grad={grad_pre_clip:.3e}->{grad_post_clip:.3e} | "
                f"field_grad={field_grad_pre_clip:.3e}->{field_grad_post_clip:.3e} | "
                f"shape_grad={shape_grad_pre_clip:.3e}->{shape_grad_post_clip:.3e}"
            )

        maybe_make_epoch_plot(
            epoch,
            field_net,
            shape_net,
            run_dir,
            hist,
            device,
        )

    if train_cfg.get("plot_final", True):
        final_epoch = train_cfg["steps"]
        plot_every = train_cfg.get("plot_every", None)
        final_already_saved = (
            plot_every is not None
            and int(plot_every) > 0
            and final_epoch % int(plot_every) == 0
        )
        if not final_already_saved:
            maybe_make_epoch_plot(
                final_epoch,
                field_net,
                shape_net,
                run_dir,
                hist,
                device,
                force=True,
            )

    save_training_history(run_dir, hist)
    hist = history_array(hist)

    # New files only. Input model files are not touched.
    torch.save(field_net.state_dict(), run_dir / "field_net_coupled_state_dict.pt")
    torch.save(shape_net.state_dict(), run_dir / "shape_net_coupled_state_dict.pt")

    return hist


# -----------------------------
# Evaluation / saving
# -----------------------------
def projected_electrode_mask(xy_projected, shape_net, device, n_boundary=800):
    mask = np.zeros(len(xy_projected), dtype=bool)
    for cx, cy, r in electrodes.values():
        _, x_el = predicted_circle(shape_net, cx, cy, r, device, n=n_boundary)
        mask |= MplPath(x_el).contains_points(xy_projected)
    return mask


def evaluate_coupled_on_grid(field_net, shape_net, device):
    eval_cfg = config["evaluation"]
    nx, ny = eval_cfg["nx"], eval_cfg["ny"]
    chunk_size = eval_cfg["chunk_size"]

    z_x = np.linspace(xmin, xmax, nx)
    z_y = np.linspace(ymin, ymax, ny)
    Zx, Zy = np.meshgrid(z_x, z_y)
    z_np = np.column_stack([Zx.ravel(), Zy.ravel()]).astype(np.float32)

    phi_out = []
    psi_out = []
    xproj_out = []

    for start in range(0, len(z_np), chunk_size):
        z_chunk = to_tensor(z_np[start:start + chunk_size], device=device, requires_grad=True)
        x_chunk = shape_net(z_chunk)
        phi = field_net(x_chunk)
        psi = pseudopotential_eV(phi, x_chunk)

        phi_out.append(phi.detach().cpu().numpy())
        psi_out.append(psi.detach().cpu().numpy())
        xproj_out.append(x_chunk.detach().cpu().numpy())

    phi_np = np.vstack(phi_out).reshape(ny, nx)
    psi_np = np.vstack(psi_out).reshape(ny, nx)
    xproj_np = np.vstack(xproj_out).reshape(ny, nx, 2)

    # Mask the plotted fields using the projected electrode regions, not the
    # original circular electrode regions.
    mask = projected_electrode_mask(xproj_np.reshape(-1, 2), shape_net, device).reshape(ny, nx)
    phi_np = np.where(mask, np.nan, phi_np)
    psi_np = np.where(mask, np.nan, psi_np)

    Xp = xproj_np[:, :, 0]
    Yp = xproj_np[:, :, 1]
    return Xp, Yp, phi_np, psi_np, xproj_np

def evaluate_cpn_on_grid(shape_net, device):
    eval_cfg = config["evaluation"]
    nx, ny = eval_cfg["nx"], eval_cfg["ny"]
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    z = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    mask = inside_any_electrode(z).reshape(ny, nx)

    with torch.no_grad():
        x_pred = shape_net(to_tensor(z, device=device)).cpu().numpy()

    U = (x_pred[:, 0] - z[:, 0]).reshape(ny, nx)
    V = (x_pred[:, 1] - z[:, 1]).reshape(ny, nx)
    D = np.sqrt(U**2 + V**2)

    U = np.where(mask, np.nan, U)
    V = np.where(mask, np.nan, V)
    D = np.where(mask, np.nan, D)
    return X, Y, U, V, D


def save_fields(run_dir, X, Y, phi, psi_eV, xproj):
    np.savez(
        run_dir / "fields.npz",
        X=X,
        Y=Y,
        phi_rf=phi,
        psi_rf_eV=psi_eV,
        x_projected_grid=xproj,
        scale_J_per_E2=scale_J_per_E2,
        e_charge=e_charge,
        M=M,
        Omega=Omega,
        geom_unit_to_m=geom_unit_to_m,
        ideal_pseudo_K_eV_per_geom_unit2=automatic_ideal_pseudo_K(),
    )


def save_projection_data(run_dir, shape_net, device):
    z = sample_cpn_plot_points()
    with torch.no_grad():
        x_pred = shape_net(to_tensor(z, device=device)).cpu().numpy()
    np.savez(run_dir / "projection_data.npz", z_reference=z, x_pred=x_pred, displacement=x_pred - z)


# -----------------------------
# Plotting: PINN-style
# -----------------------------
def predicted_circle(shape_net, cx, cy, r, device, n=400):
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]).astype(np.float32)
    with torch.no_grad():
        x_pred = shape_net(to_tensor(z, device=device)).cpu().numpy()
    return z, x_pred


def predicted_world_boundary(shape_net, device, n=400):
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
        x_pred = shape_net(to_tensor(z, device=device)).cpu().numpy()
    return z, x_pred


def add_reference_geometry(ax, linewidth=1.0, linestyle="--"):
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=linewidth, linestyle=linestyle))
    for cx, cy, r in electrodes.values():
        ax.add_patch(Circle((cx, cy), r, fill=False, linewidth=linewidth, linestyle=linestyle))


def add_projected_geometry(ax, shape_net, device, linewidth=1.5):
    _, x_world = predicted_world_boundary(shape_net, device)
    ax.plot(x_world[:, 0], x_world[:, 1], color="black", linewidth=linewidth)
    for cx, cy, r in electrodes.values():
        _, x_el = predicted_circle(shape_net, cx, cy, r, device)
        ax.plot(x_el[:, 0], x_el[:, 1], color="black", linewidth=linewidth)


def plot_field(run_dir, X, Y, field, filename, colorbar_label, title, shape_net, device):
    fig, ax = plt.subplots()
    field_masked = np.ma.masked_invalid(field)
    c = ax.contourf(X, Y, field_masked, levels=100, cmap="jet")
    fig.colorbar(c, ax=ax, label=colorbar_label)
    add_projected_geometry(ax, shape_net, device)
    ax.set_xlabel("reference x [cm]")
    ax.set_ylabel("reference y [cm]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=300)
    return fig, ax


def plot_loss_history(run_dir, hist):
    if hist is None or len(hist) == 0:
        return None, None

    hist = np.asarray(hist)
    fig, ax1 = plt.subplots()
    ax1.semilogy(hist[:, 0], hist[:, 1], label="total")
    ax1.semilogy(hist[:, 0], hist[:, 2], label="pde")
    ax1.semilogy(hist[:, 0], hist[:, 3], label="bc")
    ax1.semilogy(hist[:, 0], hist[:, 4], label="pseudo")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")

    ax2 = ax1.twinx()
    if hist.shape[1] > 12:
        ax1.semilogy(hist[:, 0], hist[:, 5], label="translation")
        ax1.semilogy(hist[:, 0], hist[:, 6], label="scale")
        lr_idx = 7
        grad_idx = 8
    else:
        lr_idx = 5
        grad_idx = 6

    ax2.semilogy(hist[:, 0], hist[:, lr_idx], linestyle="--", label="lr")
    ax2.semilogy(hist[:, 0], hist[:, grad_idx], linestyle=":", label="grad pre")
    ax2.semilogy(hist[:, 0], hist[:, grad_idx + 1], linestyle="-.", label="grad post")
    if hist.shape[1] > grad_idx + 5:
        ax2.semilogy(hist[:, 0], hist[:, grad_idx + 2], linestyle="--", label="field grad pre")
        ax2.semilogy(hist[:, 0], hist[:, grad_idx + 3], linestyle="-.", label="field grad post")
        ax2.semilogy(hist[:, 0], hist[:, grad_idx + 4], linestyle=":", label="shape grad pre")
        ax2.semilogy(hist[:, 0], hist[:, grad_idx + 5], linestyle="-", label="shape grad post")
    ax1.legend(loc="upper right")
    ax2.set_ylabel("lr / gradient norm")
    ax2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_history.png", dpi=300)
    return fig, ax1

def plot_pseudo_roi_weight(run_dir, shape_net, device):
    train_cfg = config["coupled_training"]
    eval_cfg = config["evaluation"]

    r_cut = train_cfg["pseudo_roi_radius"]
    transition = train_cfg["pseudo_roi_transition"]

    nx = eval_cfg["nx"]
    ny = eval_cfg["ny"]

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    xy_t = to_tensor(xy, device=device, requires_grad=False)

    with torch.no_grad():
        w = roi_soft_weight(
            xy_t,
            r_cut=r_cut,
            transition=transition,
        ).cpu().numpy()

    W = w.reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(7, 6))

    c = ax.contourf(X, Y, W, levels=100, cmap="jet")
    fig.colorbar(c, ax=ax, label="pseudopotential ROI weight")

    # Dashed circle marks r = r_cut, where the soft weight is 0.5.
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        r_cut * np.cos(theta),
        r_cut * np.sin(theta),
        linestyle="--",
        linewidth=1.5,
        label=f"r_cut = {r_cut:g}",
    )

    # Overlay projected electrode/world geometry.
    add_projected_geometry(ax, shape_net, device)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Pseudopotential ROI Weight")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(run_dir / "pseudo_roi_weight.png", dpi=300)

    return fig, ax

def plot_ideal_pseudopotential(run_dir, shape_net, device, vmax=None):
    train_cfg = config["coupled_training"]
    eval_cfg = config["evaluation"]
    vmax = eval_cfg.get("pseudo_vmax_eV", None)

    nx = eval_cfg["nx"]
    ny = eval_cfg["ny"]

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    xy_t = to_tensor(xy, device=device, requires_grad=False)

    with torch.no_grad():
        psi = ideal_pseudopotential_eV(xy_t).cpu().numpy()

    PSI = psi.reshape(ny, nx)

    if vmax is None:
        vmax = np.nanmax(PSI)

    fig, ax = plt.subplots(figsize=(7, 6))

    c = ax.contourf(
        X,
        Y,
        PSI,
        levels=100,
        vmin=0.0,
        vmax=vmax,
        cmap="jet"
    )
    fig.colorbar(c, ax=ax, label="ideal pseudopotential [eV]")

    add_projected_geometry(ax, shape_net, device)

    if "pseudo_roi_radius" in train_cfg:
        r_cut = train_cfg["pseudo_roi_radius"]
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(
            r_cut * np.cos(theta),
            r_cut * np.sin(theta),
            linestyle="--",
            linewidth=1.5,
            label=f"r_cut = {r_cut:g}",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ideal Pseudopotential")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(run_dir / "ideal_pseudopotential.png", dpi=300)

    return fig, ax

def plot_pseudo_xeqy_cross_section(
    run_dir,
    psi_eV,
    device,
    filename="pseudo_xeqy_cross_section.png",
    n_line=500,
    subtract_center=True,
):
    eval_cfg = config["evaluation"]

    nx = eval_cfg["nx"]
    ny = eval_cfg["ny"]

    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)

    if torch.is_tensor(psi_eV):
        PSI = psi_eV.detach().cpu().numpy()
    else:
        PSI = np.asarray(psi_eV)

    PSI = PSI.reshape(ny, nx)

    # x = y line must lie inside both the x and y plotting domains.
    t_min = max(xmin, ymin)
    t_max = min(xmax, ymax)
    t = np.linspace(t_min, t_max, n_line)

    # Signed distance along the x = y line from the trap center.
    # Point on line is (t, t), so distance from origin is sqrt(2) * t.
    s = np.sqrt(2.0) * t

    # Bilinear interpolation of PSI(x, y) onto points (t, t).
    xi = (t - xmin) / (xmax - xmin) * (nx - 1)
    yi = (t - ymin) / (ymax - ymin) * (ny - 1)

    i0 = np.floor(xi).astype(int)
    j0 = np.floor(yi).astype(int)

    i0 = np.clip(i0, 0, nx - 2)
    j0 = np.clip(j0, 0, ny - 2)

    i1 = i0 + 1
    j1 = j0 + 1

    wx = xi - i0
    wy = yi - j0

    psi_actual = (
        (1.0 - wx) * (1.0 - wy) * PSI[j0, i0]
        + wx * (1.0 - wy) * PSI[j0, i1]
        + (1.0 - wx) * wy * PSI[j1, i0]
        + wx * wy * PSI[j1, i1]
    )

    xy_line = np.column_stack([t, t]).astype(np.float32)
    xy_line_t = to_tensor(xy_line, device=device, requires_grad=False)

    with torch.no_grad():
        psi_ideal = ideal_pseudopotential_eV(xy_line_t).detach().cpu().numpy().ravel()

    if subtract_center:
        center_xy = np.array([[0.0, 0.0]], dtype=np.float32)
        center_t = to_tensor(center_xy, device=device, requires_grad=False)

        with torch.no_grad():
            psi_ideal_center = ideal_pseudopotential_eV(center_t).detach().cpu().numpy().item()

        # For the actual sampled field, use the interpolated value closest to s = 0.
        center_idx = np.argmin(np.abs(s))
        psi_actual = psi_actual - psi_actual[center_idx]
        psi_ideal = psi_ideal - psi_ideal_center

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        s,
        psi_actual,
        linewidth=2.0,
        label="actual pseudopotential",
    )

    ax.plot(
        s,
        psi_ideal,
        linestyle="--",
        linewidth=2.0,
        label="ideal pseudopotential",
    )

    ax.set_xlabel("signed distance along x = y")
    ax.set_ylabel("trap depth [eV]" if subtract_center else "pseudopotential [eV]")
    ax.set_title("Pseudopotential Cross Section Along x = y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=300)

    return fig, ax

# -----------------------------
# Plotting: CPN-style
# -----------------------------
def plot_displacement_field(run_dir, X, Y, U, V, D):
    stride = config["evaluation"]["quiver_stride"]
    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, np.ma.masked_invalid(D), levels=100)
    fig.colorbar(c, ax=ax, label=r"$|NN_\phi(z)-z|$")
    ax.quiver(
        X[::stride, ::stride], Y[::stride, ::stride],
        U[::stride, ::stride], V[::stride, ::stride],
        angles="xy", scale_units="xy", scale=1, width=0.002,
    )
    add_reference_geometry(ax)
    ax.set_xlabel("reference x [cm]")
    ax.set_ylabel("reference y [cm]")
    ax.set_title("CPN Residual Displacement Field")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(run_dir / "cpn_displacement_field.png", dpi=300)
    return fig, ax


def plot_geometry_overlay(run_dir, shape_net, device):
    fig, ax = plt.subplots()
    z_world, x_world = predicted_world_boundary(shape_net, device)
    ax.plot(z_world[:, 0], z_world[:, 1], linestyle="--", label="reference world")
    ax.plot(x_world[:, 0], x_world[:, 1], label="projected world")

    first = True
    for cx, cy, r in electrodes.values():
        z_el, x_el = predicted_circle(shape_net, cx, cy, r, device)
        ax.plot(z_el[:, 0], z_el[:, 1], linestyle="--", label="reference electrodes" if first else None)
        ax.plot(x_el[:, 0], x_el[:, 1], label="projected electrodes" if first else None)
        first = False

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title("Reference Geometry vs Final CPN Projection")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "cpn_geometry_overlay.png", dpi=300)
    return fig, ax


def plot_point_cloud_projection(run_dir, shape_net, device):
    z = sample_cpn_plot_points()
    with torch.no_grad():
        x_pred = shape_net(to_tensor(z, device=device)).cpu().numpy()

    point_colors = np.random.rand(len(z))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    axes[0].scatter(z[:, 0], z[:, 1], c=point_colors, s=2, alpha=0.7, cmap="hsv", edgecolors="none")
    add_reference_geometry(axes[0])
    axes[0].set_title("Reference points z")

    axes[1].scatter(x_pred[:, 0], x_pred[:, 1], c=point_colors, s=2, alpha=0.7, cmap="hsv", edgecolors="none")
    add_projected_geometry(axes[1], shape_net, device)
    axes[1].set_title(r"Projected points $NN_\phi(z)$")

    for ax in axes:
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(run_dir / "cpn_point_cloud_projection.png", dpi=300)
    return fig, axes


def plot_and_save_results(run_dir, field_net, shape_net, hist, device, show=False):
    plot_loss_history(run_dir, hist)

    X, Y, phi, psi_eV, xproj = evaluate_coupled_on_grid(field_net, shape_net, device)
    save_fields(run_dir, X, Y, phi, psi_eV, xproj)

    plot_field(
        run_dir, X, Y, phi,
        filename="phi_rf.png",
        colorbar_label=r"$\Phi_{\rm RF}$ [V]",
        title=r"Coupled RF Potential Amplitude $\Phi_{\rm RF}$",
        shape_net=shape_net,
        device=device,
    )
    plot_field(
        run_dir, X, Y, psi_eV,
        filename="psi_rf_eV.png",
        colorbar_label=r"$\psi_{\rm RF}$ [eV]",
        title=r"Coupled RF Pseudopotential $\psi_{\rm RF}$",
        shape_net=shape_net,
        device=device,
    )

    Xc, Yc, U, V, D = evaluate_cpn_on_grid(shape_net, device)
    plot_displacement_field(run_dir, Xc, Yc, U, V, D)
    plot_geometry_overlay(run_dir, shape_net, device)
    plot_point_cloud_projection(run_dir, shape_net, device)
    plot_pseudo_roi_weight(run_dir, shape_net, device)
    plot_ideal_pseudopotential(run_dir, shape_net, device)
    plot_pseudo_xeqy_cross_section(run_dir, psi_eV=psi_eV, device=device)
    save_projection_data(run_dir, shape_net, device)

    if show:
        plt.show()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_dir = make_run_dir(config)
    save_config(config, run_dir)
    save_script_copy(run_dir)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Ideal pseudo K [eV/cm^2]: {automatic_ideal_pseudo_K():.6e}")
    print(f"World BC: {world_bc['type']} with value {world_bc['value']}")
    if config["coupled_training"].get("use_grad_clip", False):
        clip_cfg = config["coupled_training"].get("grad_clip", {})
        print(
            "Gradient clipping: "
            f"field_max_norm={clip_cfg.get('field_max_norm')}, "
            f"shape_max_norm={clip_cfg.get('shape_max_norm')}"
        )
    else:
        print("Gradient clipping: disabled")

    geom_penalty_cfg = config["coupled_training"].get("electrode_geometry_penalty", {})
    print(
        "Electrode geometry penalties: "
        f"translation={geom_penalty_cfg.get('use_translation_penalty', False)} "
        f"(weight={geom_penalty_cfg.get('translation_weight', 0.0)}), "
        f"scale={geom_penalty_cfg.get('use_scale_penalty', False)} "
        f"(weight={geom_penalty_cfg.get('scale_weight', 0.0)})"
    )

    field_cfg = config["pinn_model"]
    shape_cfg = config["cpn_model"]

    field_net = FieldMLP(width=field_cfg["width"], depth=field_cfg["depth"]).to(device)
    shape_net = CoordinateProjectionNet(
        width=shape_cfg["width"],
        depth=shape_cfg["depth"],
        zero_last_layer=shape_cfg["zero_last_layer"],
    ).to(device)

    if config["input_models"]["use_existing_PINN"]:
        print(f"Using PINN: {config['input_models']['pinn_state_dict']}")
        field_net = load_state_dict_safely(field_net, config["input_models"]["pinn_state_dict"], device)
    else:
        print("*********Creating new PINN*********")
    if config["input_models"]["use_existing_CPN"]:
        print(f"Using CPN: {config['input_models']['cpn_state_dict']}")
        shape_net = load_state_dict_safely(shape_net, config["input_models"]["cpn_state_dict"], device)
    else:
        print("*********Creating new CPN*********")

    hist = train_coupled(field_net, shape_net, run_dir, device)

    print(f"\nLoaded initial PINN from: {config['input_models']['pinn_state_dict']}")
    print(f"Loaded initial CPN from:  {config['input_models']['cpn_state_dict']}")
    print("Saved updated models as new files:")
    print(f"  {run_dir / 'field_net_coupled_state_dict.pt'}")
    print(f"  {run_dir / 'shape_net_coupled_state_dict.pt'}")
    print(f"\nSaved run to: {run_dir.resolve()}")
