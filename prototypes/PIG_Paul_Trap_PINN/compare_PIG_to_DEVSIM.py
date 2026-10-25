import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.tri as mtri
import pyvista as pv

# -----------------------------
# User settings
# -----------------------------
# Put this script in the directory containing paul_trap_basis_RF.dat, or set DEVSIM_DIR.
DEVSIM_DIR = Path("/Users/dalejulson/Desktop/DEVSIM/SPINN/DEVSIM/Circular_trap/Neumann_BC/")
RF_MESH_FILE = "paul_trap_basis_RF.dat"

# Set this to a specific PIG run directory, or leave as None to use the newest
# runs/paul_trap_pig_* directory containing model_state_dict.pt and config.json.
PIG_RUN_DIR = "runs/paul_trap_pig_20260706_165733/"

# Output folder for comparison plots. Leave as None to save into PIG_RUN_DIR.
OUT_DIR = None

# PyVista multiblock index used by your existing visualization script.
BLOCK_INDEX = 4

# Inference batch size. Lower this if you run out of memory.
BATCH_SIZE = 8192

# Extra scale applied to the PIG potential after loading the model.
# Example: PIG trained with 1 V electrodes but DEVSIM uses 300 V electrodes -> set 300.0.
# The RF potential is multiplied by this factor, and the pseudopotential changes by this factor squared.
PIG_OUTPUT_SCALE = 300.0

# Pseudopotential constants.
e_charge = 1.602e-19
amu = 1.66054e-27
ion_mass_amu = 40.0
rf_frequency_hz = 10.2e6
geom_unit_to_m = 1e-2  # geometry unit is cm

# Plotting
CMAP = "jet"
N_LEVELS = 120


# -----------------------------
# Model reconstruction
# -----------------------------
class PIG(nn.Module):

    def __init__(self, domain, n_gaussians, feature_dim, mlp_hidden,
                 sigma_init, feature_init_std):
        super().__init__()
        xmin, xmax, ymin, ymax = domain

        self.register_buffer("domain_lo", torch.tensor([xmin, ymin], dtype=torch.float32))
        self.register_buffer("domain_size", torch.tensor([xmax - xmin, ymax - ymin], dtype=torch.float32))

        N = int(n_gaussians)
        k = int(feature_dim)

        self.mu = nn.Parameter(torch.rand(N, k, 2))
        self.log_sigma = nn.Parameter(torch.full((N, k, 2), float(np.log(sigma_init))))
        self.features = nn.Parameter(float(feature_init_std) * torch.randn(N, k))

        self.mlp = nn.Sequential(
            nn.Linear(k, int(mlp_hidden)),
            nn.Tanh(),
            nn.Linear(int(mlp_hidden), 1),
        )

    def gaussian_features(self, xy):
        u = (xy - self.domain_lo) / self.domain_size
        diff = u[:, None, None, :] - self.mu
        sigma = torch.exp(self.log_sigma)
        G = torch.exp(-0.5 * ((diff / sigma) ** 2).sum(dim=-1))
        return (self.features * G).sum(dim=1)

    def forward(self, xy):
        return self.mlp(self.gaussian_features(xy))

    def gaussian_params_numpy(self):
        lo = self.domain_lo.detach().cpu().numpy()
        size = self.domain_size.detach().cpu().numpy()
        mu = lo + self.mu.detach().cpu().numpy() * size
        sigma = np.exp(self.log_sigma.detach().cpu().numpy()) * size
        features = self.features.detach().cpu().numpy()
        return mu, sigma, features


def electrodes_from_config(config):
    g = config["geometry"]
    return {
        "North": (0.0, float(g["N_distance"]), float(g["radius"])),
        "South": (0.0, -float(g["S_distance"]), float(g["radius"])),
        "West": (-float(g["W_distance"]), 0.0, float(g["radius"])),
        "East": (float(g["E_distance"]), 0.0, float(g["radius"])),
    }


def reference_scale_from_config(config):

    train_cfg = config.get("training", {})
    potential_scale = train_cfg.get("potential_scale", 1.0)
    if potential_scale != "auto":
        return float(potential_scale)

    bc = config["boundary_conditions"]
    vals = [abs(float(v)) for v in bc["rf_voltages"].values()]
    world = bc.get("world", {})
    if world.get("type", "").lower() == "dirichlet":
        vals.append(abs(float(world.get("value", 0.0))))
    return max(vals + [1.0])


def load_pig_model(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    g = config["geometry"]
    xmin, xmax = -float(g["world_size_x"]) / 2, float(g["world_size_x"]) / 2
    ymin, ymax = -float(g["world_size_y"]) / 2, float(g["world_size_y"]) / 2
    domain = (xmin, xmax, ymin, ymax)

    m = config["model"]
    model = PIG(
        domain=domain,
        n_gaussians=m["n_gaussians"],
        feature_dim=m["feature_dim"],
        mlp_hidden=m["mlp_hidden"],
        sigma_init=m["sigma_init"],
        feature_init_std=m["feature_init_std"],
    )

    state = torch.load(run_dir / "model_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    electrodes = electrodes_from_config(config)
    reference_scale = reference_scale_from_config(config)
    return model, config, electrodes, reference_scale


# -----------------------------
# DEVSIM loading and field calculations
# -----------------------------
def newest_pig_run():
    candidates = sorted(Path("runs").glob("paul_trap_pig_*"))
    candidates = [p for p in candidates if (p / "model_state_dict.pt").exists() and (p / "config.json").exists()]
    if not candidates:
        raise FileNotFoundError("No runs/paul_trap_pig_* directory with model_state_dict.pt and config.json was found.")
    return candidates[-1]


def read_devsim_mesh(filename, block_index=4):
    data = pv.get_reader(str(filename)).read()
    if isinstance(data, pv.MultiBlock):
        mesh = data[block_index]
    else:
        mesh = data
    return mesh.cell_data_to_point_data()


def get_point_array(mesh, name):
    if name in mesh.point_data:
        return np.asarray(mesh.point_data[name])
    if name in mesh.cell_data:
        mesh2 = mesh.cell_data_to_point_data()
        return np.asarray(mesh2.point_data[name])
    raise KeyError(
        f"Could not find '{name}' in mesh point_data or cell_data. "
        f"Available point_data: {list(mesh.point_data.keys())}"
    )


def inside_any_electrode_np(xy, electrodes):
    inside = np.zeros(len(xy), dtype=bool)
    x, y = xy[:, 0], xy[:, 1]
    for cx, cy, r in electrodes.values():
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def devsim_fields(mesh):
    phi = get_point_array(mesh, "Potential").astype(float)
    gx = get_point_array(mesh, "Potential_gradx").astype(float)  # V/cm
    gy = get_point_array(mesh, "Potential_grady").astype(float)  # V/cm

    grad2_vm2 = (gx ** 2 + gy ** 2) / (geom_unit_to_m ** 2)
    M = ion_mass_amu * amu
    Omega = 2 * np.pi * rf_frequency_hz
    scale = e_charge ** 2 / (4 * M * Omega ** 2)
    psi_eV = scale * grad2_vm2 / e_charge
    return phi, psi_eV


def pig_fields_at_points(model, xy, reference_scale=1.0, output_scale=1.0, batch_size=BATCH_SIZE):
    phi_all = []
    gx_all = []
    gy_all = []

    for i in range(0, len(xy), batch_size):
        xb = torch.tensor(xy[i:i + batch_size], dtype=torch.float32, requires_grad=True)
        u = model(xb)
        phi = output_scale * reference_scale * u
        grad = torch.autograd.grad(
            phi,
            xb,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
        )[0]
        phi_all.append(phi.detach().cpu().numpy().ravel())
        gx_all.append(grad[:, 0].detach().cpu().numpy())  # V/cm
        gy_all.append(grad[:, 1].detach().cpu().numpy())  # V/cm

    phi = np.concatenate(phi_all)
    gx = np.concatenate(gx_all)
    gy = np.concatenate(gy_all)

    grad2_vm2 = (gx ** 2 + gy ** 2) / (geom_unit_to_m ** 2)
    M = ion_mass_amu * amu
    Omega = 2 * np.pi * rf_frequency_hz
    scale = e_charge ** 2 / (4 * M * Omega ** 2)
    psi_eV = scale * grad2_vm2 / e_charge
    return phi, psi_eV


# -----------------------------
# Plotting and metrics
# -----------------------------
def add_electrodes(ax, electrodes):
    for _, (cx, cy, r) in electrodes.items():
        ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor="black", linewidth=1.2, zorder=10))


def masked_triangulation(x, y, electrodes):
    tri = mtri.Triangulation(x, y)
    tx = x[tri.triangles].mean(axis=1)
    ty = y[tri.triangles].mean(axis=1)
    mask = inside_any_electrode_np(np.column_stack([tx, ty]), electrodes)
    tri.set_mask(mask)
    return tri


def field_metrics(a, b):
    diff = a - b
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    denom = np.sqrt(np.mean(b ** 2))
    nrmse = rmse / denom if denom > 0 else np.nan
    max_abs = np.max(np.abs(diff))
    return diff, diff ** 2, mse, rmse, nrmse, max_abs


def save_map(out_dir, x, y, values, electrodes, filename, title, cbar_label, symmetric=False, log10=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    tri = masked_triangulation(x, y, electrodes)

    plot_values = values.copy()
    if log10:
        positive = plot_values[np.isfinite(plot_values) & (plot_values > 0)]
        floor = positive.min() if len(positive) else 1e-300
        plot_values = np.log10(np.maximum(plot_values, floor))
        cbar_label = "log10(" + cbar_label + ")"

    kwargs = {}
    if symmetric:
        vmax = np.nanmax(np.abs(plot_values))
        kwargs.update(vmin=-vmax, vmax=vmax)

    tpc = ax.tricontourf(tri, plot_values, levels=N_LEVELS, cmap=CMAP, **kwargs)
    fig.colorbar(tpc, ax=ax, label=cbar_label)
    add_electrodes(ax, electrodes)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=300)
    plt.close(fig)


def save_side_by_side(out_dir, x, y, devsim, pig, electrodes, filename, title, cbar_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    tri = masked_triangulation(x, y, electrodes)
    vmin = np.nanmin([np.nanmin(devsim), np.nanmin(pig)])
    vmax = np.nanmax([np.nanmax(devsim), np.nanmax(pig)])

    for ax, vals, label in zip(axes, [devsim, pig], ["DEVSIM", "PIG"]):
        tpc = ax.tricontourf(tri, vals, levels=N_LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax)
        add_electrodes(ax, electrodes)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_title(label)

    fig.colorbar(tpc, ax=axes.ravel().tolist(), label=cbar_label)
    fig.suptitle(title)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_one_quantity(out_dir, x, y, devsim, pig, electrodes, name, units):
    diff, sqerr, mse, rmse, nrmse, max_abs = field_metrics(pig, devsim)

    save_side_by_side(
        out_dir, x, y, devsim, pig, electrodes,
        f"{name}_devsim_vs_pig.png",
        f"{name}: DEVSIM vs PIG",
        units,
    )

    save_map(
        out_dir, x, y, diff, electrodes,
        f"{name}_difference_pig_minus_devsim.png",
        f"{name}: PIG - DEVSIM\nRMSE={rmse:.4e} {units}, NRMSE={nrmse:.4e}, max|diff|={max_abs:.4e} {units}",
        units,
        symmetric=True,
        log10=False,
    )

    save_map(
        out_dir, x, y, sqerr, electrodes,
        f"{name}_squared_error_map.png",
        f"{name}: local squared error\nMSE={mse:.4e} {units}^2, RMSE={rmse:.4e} {units}",
        f"({units})^2",
        symmetric=False,
        log10=True,
    )

    return {
        f"{name}_mse": mse,
        f"{name}_rmse": rmse,
        f"{name}_nrmse": nrmse,
        f"{name}_max_abs_error": max_abs,
    }


def save_gaussian_overlay(out_dir, electrodes, model):
    mu, sigma, features = model.gaussian_params_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot all Gaussian components. For feature_dim=1 this is just N ellipses.
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            cx, cy = mu[i, j]
            sx, sy = sigma[i, j]
            ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, fill=False, linewidth=0.6, alpha=0.35))

    add_electrodes(ax, electrodes)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title("PIG Gaussian centers and 1-sigma ellipses")
    fig.tight_layout()
    fig.savefig(out_dir / "pig_gaussian_overlay.png", dpi=300)
    plt.close(fig)


# -----------------------------
# Run comparison
# -----------------------------
def main():
    run_dir = Path(PIG_RUN_DIR) if PIG_RUN_DIR is not None else newest_pig_run()
    out_dir = Path(OUT_DIR) if OUT_DIR is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using PIG run: {run_dir}")
    model, config, electrodes, reference_scale = load_pig_model(run_dir)

    rf_mesh_path = DEVSIM_DIR / RF_MESH_FILE
    print(f"Reading DEVSIM RF mesh: {rf_mesh_path}")
    mesh = read_devsim_mesh(rf_mesh_path, BLOCK_INDEX)

    xy_all = np.asarray(mesh.points)[:, :2].astype(np.float32)
    valid = ~inside_any_electrode_np(xy_all, electrodes)

    dev_phi_all, dev_psi_all = devsim_fields(mesh)
    valid &= np.isfinite(dev_phi_all) & np.isfinite(dev_psi_all)

    xy = xy_all[valid]
    x = xy[:, 0]
    y = xy[:, 1]
    dev_phi = dev_phi_all[valid]
    dev_psi = dev_psi_all[valid]

    print(f"Evaluating PIG on {len(xy):,} DEVSIM mesh points")
    print(f"PIG reference scale from training config: {reference_scale:g} V")
    print(f"Extra PIG output scale: {PIG_OUTPUT_SCALE:g}")
    print(f"Effective PIG potential scale: {PIG_OUTPUT_SCALE * reference_scale:g} V per model output unit")
    pig_phi, pig_psi = pig_fields_at_points(model, xy, reference_scale, PIG_OUTPUT_SCALE, BATCH_SIZE)

    metrics = {}
    metrics.update(compare_one_quantity(out_dir, x, y, dev_phi, pig_phi, electrodes, "rf_potential", "V"))
    metrics.update(compare_one_quantity(out_dir, x, y, dev_psi, pig_psi, electrodes, "rf_pseudopotential", "eV"))

    save_gaussian_overlay(out_dir, electrodes, model)

    np.savez(
        out_dir / "pig_devsim_comparison_arrays.npz",
        x=x,
        y=y,
        pig_output_scale=PIG_OUTPUT_SCALE,
        pig_reference_scale_from_config=reference_scale,
        pig_effective_scale=PIG_OUTPUT_SCALE * reference_scale,
        devsim_phi_rf_V=dev_phi,
        pig_phi_rf_V=pig_phi,
        phi_diff_pig_minus_devsim_V=pig_phi - dev_phi,
        devsim_psi_rf_eV=dev_psi,
        pig_psi_rf_eV=pig_psi,
        psi_diff_pig_minus_devsim_eV=pig_psi - dev_psi,
    )

    metrics["pig_output_scale"] = float(PIG_OUTPUT_SCALE)
    metrics["pig_reference_scale_from_config"] = float(reference_scale)
    metrics["pig_effective_scale"] = float(PIG_OUTPUT_SCALE * reference_scale)

    with open(out_dir / "pig_devsim_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")
    print(f"\nSaved plots and arrays to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
