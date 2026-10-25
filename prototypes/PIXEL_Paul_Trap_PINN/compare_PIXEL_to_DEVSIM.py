import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.tri as mtri
import pyvista as pv

# -----------------------------
# User settings
# -----------------------------
# Put this script in the directory containing paul_trap_basis_RF.dat, or set DEVSIM_DIR.
DEVSIM_DIR = Path("/Users/dalejulson/Desktop/DEVSIM/SPINN/DEVSIM/Circular_trap/Neumann_BC/")
RF_MESH_FILE = "paul_trap_basis_RF.dat"

# Set this to a specific PIXEL run directory, or leave as None to use the newest
# runs/paul_trap_pinn_* directory containing model_state_dict.pt and config.json.
PIXEL_RUN_DIR = "runs/paul_trap_pinn_20260707_110950/"

# Output folder for comparison plots.
OUT_DIR = PIXEL_RUN_DIR

# PyVista multiblock index used by your existing visualization script.
BLOCK_INDEX = 4

# Inference batch size. Lower this if you run out of memory.
BATCH_SIZE = 20000

# Extra scale applied to the PIXEL potential after loading the model.
# Example: PIXEL trained with 1 V electrodes but DEVSIM uses 300 V electrodes -> set 300.0.
# The RF potential is multiplied by this factor, and the pseudopotential changes by this factor squared.
PIXEL_OUTPUT_SCALE = 300.0

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

        self.x0 = xmin - self.dx
        self.y0 = ymin - self.dy

        max_x0 = self.x0 + (self.M - 1) * self.offx
        max_y0 = self.y0 + (self.M - 1) * self.offy
        min_x0 = self.x0 + min(0.0, (self.M - 1) * self.offx)
        min_y0 = self.y0 + min(0.0, (self.M - 1) * self.offy)

        self.Nx = int(np.ceil((xmax - min_x0) / self.dx)) + 3
        self.Ny = int(np.ceil((ymax - min_y0) / self.dy)) + 3
        while max_x0 + (self.Nx - 1) * self.dx < xmax + self.dx:
            self.Nx += 1
        while max_y0 + (self.Ny - 1) * self.dy < ymax + self.dy:
            self.Ny += 1

        g_idx = torch.arange(self.M, dtype=torch.float32)
        self.register_buffer("gx0", torch.tensor(self.x0, dtype=torch.float32) + self.offx * g_idx)
        self.register_buffer("gy0", torch.tensor(self.y0, dtype=torch.float32) + self.offy * g_idx)

        C = torch.empty(self.M, self.c, self.Ny, self.Nx)
        nn.init.normal_(C, std=float(pixel_cfg["init_std"]))
        self.C = nn.Parameter(C)

        width = int(model_cfg["width"])
        depth = int(model_cfg["depth"])
        layers = [nn.Linear(self.c, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.mlp = nn.Sequential(*layers)

    def _weights(self, frac):
        if self.kernel == "cosine":
            return 0.5 * (1.0 - torch.cos(np.pi * frac))
        return frac

    def interpolate(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        feat = xy.new_zeros(xy.shape[0], self.c)

        for g in range(self.M):
            fx = (x - self.gx0[g]) / self.dx
            fy = (y - self.gy0[g]) / self.dy

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
            Cg = self.C[g].reshape(self.c, -1)

            def node(jj, ii):
                idx = jj * self.Nx + ii
                return Cg.index_select(1, idx).t()

            c00 = node(j0, i0)
            c01 = node(j0, i0 + 1)
            c10 = node(j0 + 1, i0)
            c11 = node(j0 + 1, i0 + 1)

            feat = feat + wy0 * wx0 * c00 + wy0 * wx1 * c01 + wy1 * wx0 * c10 + wy1 * wx1 * c11

        return feat

    def forward(self, xy):
        return self.mlp(self.interpolate(xy))


class HardElectrodeBCPIXEL(nn.Module):
    def __init__(self, core, electrodes, rf_voltages_n, hard_cfg):
        super().__init__()
        self.core = core
        self.electrode_names = list(electrodes.keys())
        self.electrode_params = [(float(electrodes[k][0]), float(electrodes[k][1]), float(electrodes[k][2])) for k in self.electrode_names]
        self.rf_voltages_n = [float(rf_voltages_n[k]) for k in self.electrode_names]
        self.envelope_width = float(hard_cfg["envelope_width"])
        self.distance_eps = float(hard_cfg["distance_eps"])

    def signed_distances(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        ds = []
        for cx, cy, r in self.electrode_params:
            ds.append(torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-24) - r)
        return torch.cat(ds, dim=1)

    def hard_bc_extension(self, xy):
        d = self.signed_distances(xy)
        voltages = torch.tensor(self.rf_voltages_n, dtype=xy.dtype, device=xy.device).view(1, -1)
        d2 = d ** 2
        products = []
        for i in range(d2.shape[1]):
            others = [j for j in range(d2.shape[1]) if j != i]
            products.append(torch.prod(d2[:, others], dim=1, keepdim=True))
        P = torch.cat(products, dim=1)
        lam = P / (torch.sum(P, dim=1, keepdim=True) + self.distance_eps)
        return torch.sum(lam * voltages, dim=1, keepdim=True)

    def hard_bc_envelope(self, xy):
        d = self.signed_distances(xy)
        return torch.prod(torch.tanh(d / self.envelope_width), dim=1, keepdim=True)

    def forward(self, xy):
        return self.hard_bc_extension(xy) + self.hard_bc_envelope(xy) * self.core(xy)


def electrodes_from_config(config):
    g = config["geometry"]
    return {
        "North": (0.0, float(g["N_distance"]), float(g["radius"])),
        "South": (0.0, -float(g["S_distance"]), float(g["radius"])),
        "West": (-float(g["W_distance"]), 0.0, float(g["radius"])),
        "East": (float(g["E_distance"]), 0.0, float(g["radius"])),
    }


def vref_from_config(config):
    bc = config["boundary_conditions"]
    vals = [abs(float(v)) for v in bc["rf_voltages"].values()]
    world = bc.get("world", {})
    if world.get("type", "").lower() == "dirichlet":
        vals.append(abs(float(world.get("value", 0.0))))
    v_ref = config["training"].get("potential_scale", "auto")
    return max(vals + [1.0]) if v_ref == "auto" else float(v_ref)


def load_pixel_model(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    g = config["geometry"]
    xmin, xmax = -float(g["world_size_x"]) / 2, float(g["world_size_x"]) / 2
    ymin, ymax = -float(g["world_size_y"]) / 2, float(g["world_size_y"]) / 2
    domain = (xmin, xmax, ymin, ymax)

    electrodes = electrodes_from_config(config)
    v_ref = vref_from_config(config)
    rf_voltages = config["boundary_conditions"]["rf_voltages"]
    rf_voltages_n = {k: float(v) / v_ref for k, v in rf_voltages.items()}

    core = PIXEL(domain, config["pixel"], config["model"])
    if config.get("hard_electrode_bc", {}).get("enabled", False):
        model = HardElectrodeBCPIXEL(core, electrodes, rf_voltages_n, config["hard_electrode_bc"])
    else:
        model = core

    state = torch.load(run_dir / "model_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, config, electrodes, v_ref


# -----------------------------
# DEVSIM loading and field calculations
# -----------------------------
def newest_pixel_run():
    candidates = sorted(Path("runs").glob("paul_trap_pinn_*"))
    candidates = [p for p in candidates if (p / "model_state_dict.pt").exists() and (p / "config.json").exists()]
    if not candidates:
        raise FileNotFoundError("No runs/paul_trap_pinn_* directory with model_state_dict.pt and config.json was found.")
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
    raise KeyError(f"Could not find '{name}' in mesh point_data or cell_data. Available point_data: {list(mesh.point_data.keys())}")


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

    grad2_vcm2 = gx ** 2 + gy ** 2
    grad2_vm2 = grad2_vcm2 / (geom_unit_to_m ** 2)

    M = ion_mass_amu * amu
    Omega = 2 * np.pi * rf_frequency_hz
    scale = e_charge ** 2 / (4 * M * Omega ** 2)
    psi_eV = scale * grad2_vm2 / e_charge
    return phi, psi_eV


def pixel_fields_at_points(model, xy, v_ref, output_scale=1.0, batch_size=BATCH_SIZE):
    phi_all = []
    gx_all = []
    gy_all = []

    for i in range(0, len(xy), batch_size):
        xb = torch.tensor(xy[i:i + batch_size], dtype=torch.float32, requires_grad=True)
        u = model(xb)
        phi = output_scale * v_ref * u
        grad = torch.autograd.grad(phi, xb, grad_outputs=torch.ones_like(phi), create_graph=False)[0]
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


def save_map(x, y, values, electrodes, filename, title, cbar_label, symmetric=False, log10=False):
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
    fig.savefig(OUT_DIR / filename, dpi=300)
    plt.close(fig)


def save_side_by_side(x, y, devsim, pixel, electrodes, filename, title, cbar_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    tri = masked_triangulation(x, y, electrodes)
    vmin = np.nanmin([np.nanmin(devsim), np.nanmin(pixel)])
    vmax = np.nanmax([np.nanmax(devsim), np.nanmax(pixel)])

    for ax, vals, label in zip(axes, [devsim, pixel], ["DEVSIM", "PIXEL"]):
        tpc = ax.tricontourf(tri, vals, levels=N_LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax)
        add_electrodes(ax, electrodes)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_title(label)

    fig.colorbar(tpc, ax=axes.ravel().tolist(), label=cbar_label)
    fig.suptitle(title)
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_one_quantity(x, y, devsim, pixel, electrodes, name, units):
    diff, sqerr, mse, rmse, nrmse, max_abs = field_metrics(pixel, devsim)

    save_side_by_side(
        x, y, devsim, pixel, electrodes,
        f"{name}_devsim_vs_pixel.png",
        f"{name}: DEVSIM vs PIXEL",
        units,
    )

    save_map(
        x, y, diff, electrodes,
        f"{name}_difference_pixel_minus_devsim.png",
        f"{name}: PIXEL - DEVSIM\nRMSE={rmse:.4e} {units}, NRMSE={nrmse:.4e}, max|diff|={max_abs:.4e} {units}",
        units,
        symmetric=True,
        log10=False,
    )

    save_map(
        x, y, sqerr, electrodes,
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


# -----------------------------
# Run comparison
# -----------------------------
def main():
    global OUT_DIR
    run_dir = Path(PIXEL_RUN_DIR) if PIXEL_RUN_DIR is not None else newest_pixel_run()
    OUT_DIR = Path(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using PIXEL run: {run_dir}")
    model, config, electrodes, v_ref = load_pixel_model(run_dir)

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

    print(f"Evaluating PIXEL on {len(xy):,} DEVSIM mesh points")
    print(f"PIXEL v_ref from training config: {v_ref:g} V")
    print(f"Extra PIXEL output scale: {PIXEL_OUTPUT_SCALE:g}")
    print(f"Effective PIXEL potential scale: {PIXEL_OUTPUT_SCALE * v_ref:g} V per normalized model unit")
    pix_phi, pix_psi = pixel_fields_at_points(model, xy, v_ref, PIXEL_OUTPUT_SCALE, BATCH_SIZE)

    metrics = {}
    metrics.update(compare_one_quantity(x, y, dev_phi, pix_phi, electrodes, "rf_potential", "V"))
    metrics.update(compare_one_quantity(x, y, dev_psi, pix_psi, electrodes, "rf_pseudopotential", "eV"))

    np.savez(
        OUT_DIR / "pixel_devsim_comparison_arrays.npz",
        x=x,
        y=y,
        pixel_output_scale=PIXEL_OUTPUT_SCALE,
        pixel_v_ref_from_config=v_ref,
        pixel_effective_scale=PIXEL_OUTPUT_SCALE * v_ref,
        devsim_phi_rf_V=dev_phi,
        pixel_phi_rf_V=pix_phi,
        phi_diff_pixel_minus_devsim_V=pix_phi - dev_phi,
        devsim_psi_rf_eV=dev_psi,
        pixel_psi_rf_eV=pix_psi,
        psi_diff_pixel_minus_devsim_eV=pix_psi - dev_psi,
    )

    metrics["pixel_output_scale"] = float(PIXEL_OUTPUT_SCALE)
    metrics["pixel_v_ref_from_config"] = float(v_ref)
    metrics["pixel_effective_scale"] = float(PIXEL_OUTPUT_SCALE * v_ref)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")
    print(f"\nSaved plots and arrays to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
