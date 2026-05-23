"""Visualize the PINN-predicted RF field and pseudopotential.

Produces the same set of PNGs as DEVSIM/3D_Geometries/Classical_3D_Paul_trap/
visualize_results.py:

  Psi_3D_isosurfaces.png
  Psi_slice_z0.png
  Psi_slice_y0.png
  Psi_xsection_x.png
  Psi_xsection_z.png

Sources:
  --source analytic            (pure ideal-Paul lift, no NN, no params required)
  --source pinn --params FILE  (trained PINN; uses JAX autograd for the gradient)

Usage:
    python -m PINN.Classical_3D_Paul.visualize_rf --source analytic
    python -m PINN.Classical_3D_Paul.visualize_rf --source pinn --params pinn_rf_analytic_params.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from . import boundaries, geometry as g

# Optional JAX import (only needed for --source pinn)
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


# ----------------------------
# Physical constants (match DEVSIM/3D_Geometries/Classical_3D_Paul_trap/visualize_results.py)
# ----------------------------
e_charge = 1.602e-19
amu = 1.66054e-27
M_Ca = 40 * amu
M = M_Ca
f_rf = 10.2e6
Omega = 2 * np.pi * f_rf
scale = (e_charge ** 2) / (4 * M * Omega ** 2)   # e^2 / (4 M Omega^2)  [J / (V/m)^2]


def analytic_field(coords):
    """Evaluate the ideal-Paul lift phi(x) = V_RF * (1 - eta(x)) / 2.

    Returns: phi [V], grad_phi [V/cm] of shape (N, 3).
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r2 = x ** 2 + y ** 2
    eta = z ** 2 / g.z0 ** 2 - r2 / g.r0 ** 2
    phi = g.V_RF * (1.0 - eta) / 2.0
    # grad phi = -V_RF/2 * grad eta = V_RF * (x/r0^2, y/r0^2, -z/z0^2)
    grad = np.empty_like(coords)
    grad[:, 0] = g.V_RF * x / g.r0 ** 2
    grad[:, 1] = g.V_RF * y / g.r0 ** 2
    grad[:, 2] = -g.V_RF * z / g.z0 ** 2
    return phi, grad


def pinn_field(coords, params_path):
    """Evaluate a trained PINN at the given coords. Returns phi, grad_phi."""
    if not _HAS_JAX:
        raise SystemExit("JAX not installed; required for --source pinn")
    from .model import (SIREN, HardBC, AnalyticHyperbolicBC, AnalyticHyperbolicBC_Faded,
                        IsotopicHyperbolicBC, IsotopicTrainableShape, IsotopicSeededHoles,
                        TanhMLP, VanillaSoftBC)

    with open(params_path, "rb") as f:
        bundle = pickle.load(f)
    cfg = bundle["config"]
    params = jax.tree_util.tree_map(jnp.asarray, bundle["params"])

    model_type = cfg.get("model_type", "")
    seeded_model = None   # set for the seeded-holes model -> enables discovered-shape mask

    # The trainable-shape variant is a single Flax module (not SIREN + bc).
    if model_type == "isotopic_trainable_shape":
        model = IsotopicTrainableShape(
            r0_init=cfg["r0_init"], z0_init=cfg["z0_init"],
            delta_scale=cfg["delta_scale"],
            siren_hidden=tuple(cfg["hidden"]),
            w0=cfg["w0"], world_half=cfg["world_half"],
        )
        def phi_phys(p, x):
            return cfg["V_RF"] * model.apply(p, x)
    elif model_type == "vanilla_tanh_soft":
        mlp = TanhMLP(hidden=tuple(cfg["hidden"]))
        bc = VanillaSoftBC()
        def phi_phys(p, x):
            return cfg["V_RF"] * bc.phi(mlp.apply, p, x, lambda y: y / cfg["world_half"])
    elif model_type == "isotopic_seeded":
        model = IsotopicSeededHoles(
            ring_R=cfg["ring_R"], ring_a=cfg["ring_a"],
            cap_d=cfg["cap_d"], cap_a=cfg["cap_a"],
            delta_scale=cfg["delta_scale"],
            siren_hidden=tuple(cfg["hidden"]),
            w0=cfg["w0"], world_half=cfg["world_half"],
        )
        def phi_phys(p, x):
            return cfg["V_RF"] * model.apply(p, x)
        seeded_model = model
    else:
        output_dim = cfg.get("output_dim", 1)
        siren = SIREN(hidden=tuple(cfg["hidden"]), w0=cfg["w0"], w0_first=cfg["w0"], output_dim=output_dim)
        if model_type == "analytic_hyperbolic":
            hard_bc = AnalyticHyperbolicBC(r0=cfg["r0"], z0=cfg["z0"])
        elif model_type == "analytic_hyperbolic_faded":
            hard_bc = AnalyticHyperbolicBC_Faded(r0=cfg["r0"], z0=cfg["z0"], world_half=cfg["world_half"])
        elif model_type == "isotopic_hyperbolic":
            hard_bc = IsotopicHyperbolicBC(r0=cfg["r0"], z0=cfg["z0"], delta_scale=cfg["delta_scale"])
        else:
            pA = jnp.asarray(bundle["pA"])
            pB = jnp.asarray(bundle["pB"])
            hard_bc = HardBC(pA=pA, pB=pB)

        def _normalizer(x):
            return x / cfg["world_half"]

        def phi_normalized(p, x):
            return hard_bc.phi(siren.apply, p, x, _normalizer)

        def phi_phys(p, x):
            return cfg["V_RF"] * phi_normalized(p, x)

    grad_phi_phys = jax.grad(phi_phys, argnums=1)
    phi_batch = jax.jit(jax.vmap(phi_phys, in_axes=(None, 0)))
    grad_batch = jax.jit(jax.vmap(grad_phi_phys, in_axes=(None, 0)))

    chunk = 4096
    phi = np.empty(len(coords), dtype=np.float64)
    grad = np.empty_like(coords)
    for i in range(0, len(coords), chunk):
        xs = jnp.asarray(coords[i:i + chunk])
        phi[i:i + chunk] = np.asarray(phi_batch(params, xs))
        grad[i:i + chunk] = np.asarray(grad_batch(params, xs))

    # For the seeded model, derive the DISCOVERED-shape air mask from the model's
    # own (deformed) electrode signed distance, rather than the real Paul outline.
    air_mask = None
    if seeded_model is not None:
        sdf_batch = jax.jit(jax.vmap(lambda p, x: seeded_model.apply(p, x, return_aux=True)[1],
                                     in_axes=(None, 0)))
        sdf = np.empty(len(coords), dtype=np.float64)
        for i in range(0, len(coords), chunk):
            sdf[i:i + chunk] = np.asarray(sdf_batch(params, jnp.asarray(coords[i:i + chunk])))
        air_mask = sdf > 0.0     # True = air (outside all discovered electrodes)
    return phi, grad, air_mask


def make_grid(spacing=0.004):
    """Build a regular ImageData grid covering the world."""
    H = g.world_half
    n = int(round(2 * H / spacing)) + 1
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(spacing, spacing, spacing),
        origin=(-H, -H, -H),
    )
    coords = np.asarray(grid.points, dtype=np.float64)
    return grid, coords, n


def mask_electrode_interior(coords):
    """Return a boolean mask: True if point is in the air region."""
    return boundaries.is_in_air(coords.astype(np.float32), eps=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["analytic", "pinn"], default="analytic")
    parser.add_argument("--params", type=str, default="pinn_rf_analytic_params.pkl",
                        help="PINN params pickle (when --source pinn)")
    parser.add_argument("--spacing", type=float, default=0.004,
                        help="grid spacing [cm]")
    parser.add_argument("--out_prefix", type=str, default="Psi_PINN")
    args = parser.parse_args()

    # ----------------------------
    # Build regular grid and evaluate phi, grad_phi
    # ----------------------------
    print(f"Building regular grid with spacing {args.spacing} cm...")
    grid, coords, n = make_grid(args.spacing)
    print(f"  Grid: {n} x {n} x {n} = {n**3} points")

    print(f"Evaluating {args.source} field...")
    air_mask_model = None
    if args.source == "analytic":
        phi_rf, grad_phi = analytic_field(coords)
    else:
        phi_rf, grad_phi, air_mask_model = pinn_field(coords, (Path(args.params) if Path(args.params).is_absolute() or Path(args.params).exists() else (Path(__file__).parent / 'models' / args.params if (Path(__file__).parent / 'models' / args.params).exists() else Path(__file__).parent / args.params)))

    # |grad phi|^2 in (V/cm)^2, convert to (V/m)^2 by * 1e4.
    grad2_vm2 = 1.0e4 * (grad_phi ** 2).sum(axis=1)
    psi_rf_eV = scale * grad2_vm2 / e_charge

    # DC contribution: ground both endcaps, so phi_dc = 0 across the whole
    # trap. The total trap potential is just the pseudopotential.
    psi_total_eV = psi_rf_eV

    # Mask:
    #  (a) electrode interiors (no physical field there)
    #  (b) for analytic source: points with |eta| > 1 are outside the natural
    #      trap region; the lift gives non-physical extrapolation there. We
    #      mask them so the visualization focuses on the saddle well.
    print("Masking electrode interiors and out-of-trap region...")
    # Seeded model -> mask by its DISCOVERED (deformed) electrodes; otherwise use
    # the real Paul electrode outline.
    valid_air = air_mask_model if air_mask_model is not None else mask_electrode_interior(coords)

    if args.source == "analytic":
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        eta_arr = z ** 2 / g.z0 ** 2 - (x ** 2 + y ** 2) / g.r0 ** 2
        in_trap = np.abs(eta_arr) <= 1.0
    else:
        in_trap = np.ones(len(coords), dtype=bool)

    valid = valid_air & in_trap
    phi_rf_masked = np.where(valid, phi_rf, 0.0)
    psi_total_masked = np.where(valid, psi_total_eV, 0.0)

    grid["phi_rf"] = phi_rf_masked
    grid["psi_total_eV"] = psi_total_masked

    # ----------------------------
    # Color-cap from the trap region (interior to the ring)
    # ----------------------------
    rho = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)
    trap_mask = (rho < g.r0) & valid & (psi_total_masked > 0)
    if trap_mask.any():
        cap = float(np.percentile(psi_total_masked[trap_mask], 99))
    else:
        cap = float(np.percentile(psi_total_masked[psi_total_masked > 0], 95))
    print(f"  Color cap (trap-zone p99): {cap:.3e} eV")

    # Clip the grid display to a region around the trap. The full world cube
    # is 0.6 cm wide but the trap occupies only ~+/- 0.2 cm; clipping makes the
    # slices and isosurfaces directly comparable to the DEVSIM PNGs.
    clip_half = 0.20
    grid_clip = grid.clip_box(bounds=(-clip_half, clip_half,
                                      -clip_half, clip_half,
                                      -clip_half, clip_half),
                              invert=False)

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pv.OFF_SCREEN = True

    # ----------------------------
    # 3D isosurfaces
    # ----------------------------
    p = pv.Plotter(off_screen=True, window_size=(1024, 768))
    isos = grid_clip.contour(np.linspace(0.05 * cap, 0.95 * cap, 10), scalars="psi_total_eV")
    p.add_mesh(grid_clip, scalars="psi_total_eV", cmap="jet", opacity=0.2, clim=(0, cap))
    p.add_mesh(isos, cmap="jet", opacity=0.7, clim=(0, cap))
    p.add_axes()
    p.screenshot(str(out_dir / f"{args.out_prefix}_3D_isosurfaces.png"))
    p.close()

    # ----------------------------
    # z=0 saddle slice
    # ----------------------------
    slice_z0 = grid_clip.slice(normal="z", origin=(0, 0, 0))
    p = pv.Plotter(off_screen=True, window_size=(1024, 768))
    p.add_mesh(slice_z0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
    p.add_axes()
    p.view_xy()
    p.screenshot(str(out_dir / f"{args.out_prefix}_slice_z0.png"))
    p.close()

    # ----------------------------
    # y=0 meridional slice
    # ----------------------------
    slice_y0 = grid_clip.slice(normal="y", origin=(0, 0, 0))
    p = pv.Plotter(off_screen=True, window_size=(1024, 768))
    p.add_mesh(slice_y0, scalars="psi_total_eV", cmap="jet", clim=(0, cap))
    p.add_axes()
    p.view_xz()
    p.screenshot(str(out_dir / f"{args.out_prefix}_slice_y0.png"))
    p.close()

    # ----------------------------
    # 1D radial cut at z=0 (along x)
    # ----------------------------
    def line_plot(p0, p1, title, filename, resolution=400):
        sample = grid_clip.sample_over_line(p0, p1, resolution=resolution)
        d = sample["Distance"]
        psi = sample["psi_total_eV"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(d, psi)
        ax.set_xlabel("Distance along line [cm]")
        ax.set_ylabel("psi_total [eV]")
        ax.set_title(title)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)

    line_plot((-0.08, 0, 0), (0.08, 0, 0),
              f"psi_total along x (radial)  [{args.source}]",
              f"{args.out_prefix}_xsection_x.png")
    line_plot((0, 0, -0.08), (0, 0, 0.08),
              f"psi_total along z (axial)  [{args.source}]",
              f"{args.out_prefix}_xsection_z.png")

    print(f"\nSaved PNGs with prefix '{args.out_prefix}_*' in {out_dir}")


if __name__ == "__main__":
    main()
