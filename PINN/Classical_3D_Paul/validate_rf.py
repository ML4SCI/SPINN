"""Validate the trained RF PINN against the DEVSIM ground-truth Tecplot output.

Loads `DEVSIM/3D_Geometries/Classical_3D_Paul_trap/paul_trap_basis_RF.dat`,
picks the air block, evaluates the PINN at the same node coordinates, and
reports per-point relative error + saves a side-by-side slice comparison PNG.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from . import geometry as g
from .model import SIREN, HardBC, AnalyticHyperbolicBC, AnalyticHyperbolicBC_Faded, IsotopicHyperbolicBC, IsotopicTrainableShape, IsotopicSeededHoles, TanhMLP, VanillaSoftBC


DEVSIM_RF_PATH = Path(__file__).parents[2] / "DEVSIM" / "3D_Geometries" / "Classical_3D_Paul_trap" / "paul_trap_basis_RF.dat"


def _pick_air_block(ds):
    """Return the largest unstructured block (the air region)."""
    best, best_n = None, -1
    for i in range(ds.n_blocks):
        b = ds[i]
        if b is not None and b.n_points > best_n:
            best, best_n = b, b.n_points
    return best


def _normalizer(x):
    return x / g.world_half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="pinn_rf_params.pkl")
    parser.add_argument("--n_eval", type=int, default=20000,
                        help="number of nodes to evaluate against (random subsample for speed)")
    parser.add_argument("--out_png", type=str, default="pinn_vs_devsim.png")
    parser.add_argument("--max_radius", type=float, default=None,
                        help="if set, only validate at DEVSIM nodes with |x|^2+|y|^2+|z|^2 <= R^2 (cm). "
                             "Useful for comparing PINNs solved on infinite vacuum (no world BC) "
                             "against finite-domain DEVSIM in the trap region where world effects "
                             "are negligible.")
    parser.add_argument("--radius_sweep", action="store_true",
                        help="sweep through several radii and print error stats at each")
    parser.add_argument("--devsim_path", type=str, default=None,
                        help="override the DEVSIM RF Tecplot path (e.g., to point at a "
                             "perturbed-geometry simulation)")
    args = parser.parse_args()

    # ----------------------------
    # Load DEVSIM reference
    # ----------------------------
    devsim_path = Path(args.devsim_path) if args.devsim_path else DEVSIM_RF_PATH
    if not devsim_path.exists():
        raise SystemExit(f"DEVSIM file not found: {devsim_path}\n"
                         "Run the DEVSIM RF script first.")
    print(f"Loading DEVSIM reference: {devsim_path}")
    ds = pv.get_reader(str(devsim_path)).read()
    air = _pick_air_block(ds).cell_data_to_point_data()
    coords = np.asarray(air.points, dtype=np.float32)
    phi_devsim = np.asarray(air["Potential"], dtype=np.float32)
    print(f"  DEVSIM air block: {len(coords)} nodes, Potential range "
          f"[{phi_devsim.min():.2f}, {phi_devsim.max():.2f}] V")

    # Optional radius filter (for option-3 infinite-vacuum models).
    if args.max_radius is not None:
        r2 = (coords ** 2).sum(axis=1)
        keep = r2 <= args.max_radius ** 2
        coords = coords[keep]
        phi_devsim = phi_devsim[keep]
        print(f"  After radius<={args.max_radius} filter: {len(coords)} nodes")

    # Subsample for speed (PINN is O(M_collocation * M_boundary) per query).
    rng = np.random.default_rng(0)
    if len(coords) > args.n_eval:
        idx = rng.choice(len(coords), args.n_eval, replace=False)
        coords_eval = coords[idx]
        phi_devsim_eval = phi_devsim[idx]
    else:
        coords_eval = coords
        phi_devsim_eval = phi_devsim

    # ----------------------------
    # Load trained PINN
    # ----------------------------
    # Accept either a bare filename (look in models/) or an explicit path
    p_arg = Path(args.params)
    if p_arg.is_absolute() or p_arg.exists():
        params_path = p_arg
    else:
        params_path = Path(__file__).parent / "models" / args.params
        if not params_path.exists():
            params_path = Path(__file__).parent / args.params      # legacy
    print(f"Loading PINN params: {params_path}")
    with open(params_path, "rb") as f:
        bundle = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, bundle["params"])
    cfg = bundle["config"]

    model_type = cfg.get("model_type", "")

    # Trainable-shape variant: the lift r0/z0 are in the params dict; model is a
    # single Flax module that takes a 3-vector and returns normalized phi.
    if model_type == "isotopic_trainable_shape":
        model = IsotopicTrainableShape(
            r0_init=cfg["r0_init"], z0_init=cfg["z0_init"],
            delta_scale=cfg["delta_scale"],
            siren_hidden=tuple(cfg["hidden"]),
            w0=cfg["w0"], world_half=cfg["world_half"],
        )
        def phi_normalized(p, x):
            return model.apply(p, x)
        phi_batch = jax.jit(jax.vmap(lambda p, x: cfg["V_RF"] * model.apply(p, x), in_axes=(None, 0)))
    elif model_type == "isotopic_seeded":
        model = IsotopicSeededHoles(
            ring_R=cfg["ring_R"], ring_a=cfg["ring_a"],
            cap_d=cfg["cap_d"], cap_a=cfg["cap_a"],
            delta_scale=cfg["delta_scale"],
            siren_hidden=tuple(cfg["hidden"]),
            w0=cfg["w0"], world_half=cfg["world_half"],
        )
        def phi_normalized(p, x):
            return model.apply(p, x)
        phi_batch = jax.jit(jax.vmap(lambda p, x: cfg["V_RF"] * model.apply(p, x), in_axes=(None, 0)))
    elif model_type == "vanilla_tanh_soft":
        mlp = TanhMLP(hidden=tuple(cfg["hidden"]))
        bc = VanillaSoftBC()
        def phi_normalized(p, x):
            return bc.phi(mlp.apply, p, x, _normalizer)
        phi_batch = jax.jit(jax.vmap(lambda p, x: cfg["V_RF"] * phi_normalized(p, x), in_axes=(None, 0)))
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

        def phi_normalized(p, x):
            return hard_bc.phi(siren.apply, p, x, _normalizer)
        # phi_batch already returns V (multiplied by V_RF inside the lambda above)
        phi_batch = jax.jit(jax.vmap(lambda p, x: cfg["V_RF"] * phi_normalized(p, x), in_axes=(None, 0)))

    # ----------------------------
    # Evaluate PINN at DEVSIM nodes (in chunks to keep memory sane).
    # phi_batch is set above to return values in V (already multiplied by V_RF).
    # ----------------------------
    print(f"Evaluating PINN at {len(coords_eval)} nodes...")
    chunk = 1024
    phi_pinn = np.empty(len(coords_eval), dtype=np.float32)
    for i in range(0, len(coords_eval), chunk):
        xs = jnp.asarray(coords_eval[i:i + chunk])
        phi_pinn[i:i + chunk] = np.asarray(phi_batch(params, xs))

    # ----------------------------
    # Error metrics
    # ----------------------------
    err = phi_pinn - phi_devsim_eval
    abs_err = np.abs(err)
    # Relative error normalized to V_RF (avoids divide-by-zero at the saddle).
    rel_err_vrf = abs_err / cfg["V_RF"]
    print(
        "\nError vs DEVSIM:\n"
        f"  mean |err| = {abs_err.mean():.4f} V   ({100 * rel_err_vrf.mean():.3f} % of V_RF)\n"
        f"  p50  |err| = {np.median(abs_err):.4f} V\n"
        f"  p95  |err| = {np.percentile(abs_err, 95):.4f} V\n"
        f"  max  |err| = {abs_err.max():.4f} V"
    )

    # ----------------------------
    # Side-by-side scatter slice at z=0 (project nodes with |z| < tol)
    # ----------------------------
    tol = 0.01
    mask = np.abs(coords_eval[:, 2]) < tol
    xs_s, ys_s = coords_eval[mask, 0], coords_eval[mask, 1]
    phi_d_s = phi_devsim_eval[mask]
    phi_p_s = phi_pinn[mask]
    vmin = min(phi_d_s.min(), phi_p_s.min())
    vmax = max(phi_d_s.max(), phi_p_s.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    s0 = axes[0].scatter(xs_s, ys_s, c=phi_d_s, cmap="jet", s=2, vmin=vmin, vmax=vmax)
    axes[0].set_aspect("equal"); axes[0].set_title("DEVSIM (V)")
    plt.colorbar(s0, ax=axes[0])

    s1 = axes[1].scatter(xs_s, ys_s, c=phi_p_s, cmap="jet", s=2, vmin=vmin, vmax=vmax)
    axes[1].set_aspect("equal"); axes[1].set_title("PINN (V)")
    plt.colorbar(s1, ax=axes[1])

    err_s = phi_p_s - phi_d_s
    a = max(abs(err_s.min()), abs(err_s.max()))
    s2 = axes[2].scatter(xs_s, ys_s, c=err_s, cmap="bwr", s=2, vmin=-a, vmax=a)
    axes[2].set_aspect("equal"); axes[2].set_title("PINN − DEVSIM (V)")
    plt.colorbar(s2, ax=axes[2])

    fig.suptitle(f"RF Potential at z ≈ 0  ({mask.sum()} nodes)")
    fig.tight_layout()
    out_path = Path(__file__).parent / args.out_png
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved comparison plot: {out_path}")


if __name__ == "__main__":
    main()
