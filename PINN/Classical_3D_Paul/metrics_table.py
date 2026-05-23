"""Build a metrics table across all trained PINN models.

Walks PINN/Classical_3D_Paul/models/, runs trap_quality.evaluate on each,
extracts/derives the effective lift shape (r0, z0), and writes:

  - metrics.json: machine-readable list of per-model metrics
  - metrics.md:   pretty-printed markdown table for humans

Metrics per model:
  - architecture (Shepard / analytic / isotopic / isotopic_trainable)
  - r0, z0 (config if explicit; else derived from level-surface intercepts)
  - Paul ratio = r0^2 / z0^2  (ideal Paul = 2.0)
  - isotropy = lambda_min / lambda_max of Hessian at the locus
  - secular frequencies (MHz) — three values, ascending
  - trap depth at r=0.05 cm  (max and mean psi on the bounding sphere)
  - locus position (cm) — distance from origin

Usage:
    python -m PINN.Classical_3D_Paul.metrics_table
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry as g
from .trap_quality import _build_phi, evaluate


def _level_surface_intercepts(phi_fn, V_RF):
    """Find the z-axis intercept of phi=V_RF (gives effective z0_endcap) and
    the radial intercept at z=0 of phi=0 (gives effective r0_ring).

    Wait — the analytic lift has phi(0,0,z0) = (1 - 1)/2 * V_RF = 0 (endcap)
    and phi(r0,0,0) = (1 - (-1))/2 * V_RF = V_RF (ring). So:
      - phi = V_RF at (r0, 0, 0)  ->  ring inner radius at the equator
      - phi = 0    at (0, 0, z0)  ->  endcap inner intercept on the z-axis

    Use bisection along the respective axes.
    """
    # phi = 0 along +z axis  ->  effective z0
    z_grid = np.linspace(0.02, 0.30, 4000)
    pts_z = np.stack([np.zeros_like(z_grid), np.zeros_like(z_grid), z_grid], axis=-1)
    phi_b = jax.jit(jax.vmap(phi_fn))
    phi_z = np.asarray(phi_b(jnp.asarray(pts_z, dtype=jnp.float32)))
    idx = np.where(np.sign(phi_z[:-1]) != np.sign(phi_z[1:]))[0]
    if len(idx):
        j = idx[0]
        z0_eff = float(z_grid[j] - phi_z[j] * (z_grid[j+1] - z_grid[j]) / (phi_z[j+1] - phi_z[j]))
    else:
        z0_eff = float("nan")

    # phi = V_RF along +x axis  ->  effective r0
    r_grid = np.linspace(0.02, 0.30, 4000)
    pts_r = np.stack([r_grid, np.zeros_like(r_grid), np.zeros_like(r_grid)], axis=-1)
    phi_r = np.asarray(phi_b(jnp.asarray(pts_r, dtype=jnp.float32)))
    f = phi_r - V_RF
    idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if len(idx):
        j = idx[0]
        r0_eff = float(r_grid[j] - f[j] * (r_grid[j+1] - r_grid[j]) / (f[j+1] - f[j]))
    else:
        r0_eff = float("nan")

    return r0_eff, z0_eff


def _architecture(cfg):
    mt = cfg.get("model_type", "")
    if mt == "":
        return "Shepard hard-BC (regular PINN)"
    if mt == "analytic_hyperbolic":
        return "Analytic lift + scalar NN correction"
    if mt == "analytic_hyperbolic_faded":
        return "Analytic lift × world fade + soft electrode BC"
    if mt == "isotopic_hyperbolic":
        return "Isotopic (analytic lift + bounded diffeomorphism)"
    if mt == "isotopic_trainable_shape":
        return "Isotopic + trainable r0, z0"
    if mt == "isotopic_seeded":
        return "Seeded-holes discovery (deformed torus + blobs)"
    if mt == "vanilla_tanh_soft":
        return "Vanilla (tanh + soft BC)"
    return mt


def _extract_r0_z0(cfg, phi_fn, V_RF):
    """Pull r0, z0 from config if explicitly present; otherwise derive from
    the trained model's level-surface intercepts."""
    if cfg.get("model_type") == "isotopic_trainable_shape":
        return float(cfg["r0_final"]), float(cfg["z0_final"]), "trained (Flax param)"
    if "r0" in cfg and "z0" in cfg:
        return float(cfg["r0"]), float(cfg["z0"]), "fixed (config)"
    r0_eff, z0_eff = _level_surface_intercepts(phi_fn, V_RF)
    return r0_eff, z0_eff, "derived (level-surface intercept)"


def main():
    models_dir = Path(__file__).parent / "models"
    pkls = sorted(models_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"No .pkl files in {models_dir}")
    print(f"Found {len(pkls)} models in {models_dir}\n")

    rows = []
    for p in pkls:
        print(f"--- {p.name} ---")
        try:
            phi_fn, cfg = _build_phi(p)
        except SystemExit as e:
            print(f"  skipped: {e}")
            continue
        try:
            metrics = evaluate(phi_fn, V_RF=cfg["V_RF"], verbose=False, trap_radius=0.05)
        except Exception as e:
            print(f"  trap_quality failed: {e}")
            continue
        r0, z0, r0z0_source = _extract_r0_z0(cfg, phi_fn, cfg["V_RF"])
        paul_ratio = r0 ** 2 / z0 ** 2 if (np.isfinite(r0) and np.isfinite(z0) and z0 > 0) else float("nan")

        row = {
            "model": p.name,
            "architecture": _architecture(cfg),
            "r0_cm": r0,
            "z0_cm": z0,
            "r0z0_source": r0z0_source,
            "paul_ratio_r0sq_over_z0sq": paul_ratio,
            "isotropy_min_over_max_eig": float(metrics["isotropy"]),
            "secular_freqs_MHz": [float(f) for f in metrics["secular_freqs_MHz"]],
            "trap_depth_max_eV_at_r0.05cm": float(metrics["trap_depth_eV_at_radius"][1]),
            "trap_depth_mean_eV_at_r0.05cm": float(metrics["trap_depth_mean_at_radius"][1]),
            "locus_cm": [float(x) for x in metrics["x_min_cm"]],
            "locus_offset_from_origin_um": float(1e4 * np.linalg.norm(metrics["x_min_cm"])),
            "stable": bool(metrics["stable"]),
        }
        rows.append(row)
        print(f"  arch: {row['architecture']}")
        print(f"  r0={r0:.5f}, z0={z0:.5f}  ({r0z0_source}), Paul ratio={paul_ratio:.3f}")
        print(f"  isotropy={row['isotropy_min_over_max_eig']:.3f}, "
              f"depth={row['trap_depth_max_eV_at_r0.05cm']:.2f} eV, "
              f"freqs={[f'{f:.2f}' for f in row['secular_freqs_MHz']]} MHz, "
              f"locus offset={row['locus_offset_from_origin_um']:.1f} um")
        print()

    out_json = Path(__file__).parent / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out_json}")

    # Markdown table
    out_md = Path(__file__).parent / "metrics.md"
    with open(out_md, "w") as f:
        f.write("# Trap-quality metrics across PINN models\n\n")
        f.write(f"V_RF = 300 V, Ca-40 ion, f_RF = 10.2 MHz, trap depth measured at r = 0.05 cm.\n\n")
        f.write("| Model | Architecture | r0 (cm) | z0 (cm) | Paul ratio | Isotropy | Secular freqs (MHz) | Depth (eV) | Locus offset (µm) | Stable |\n")
        f.write("|---|---|---:|---:|---:|---:|---|---:|---:|:---:|\n")
        for r in rows:
            freqs = ", ".join(f"{x:.2f}" for x in r["secular_freqs_MHz"])
            f.write(f"| `{r['model']}` "
                    f"| {r['architecture']} "
                    f"| {r['r0_cm']:.4f} "
                    f"| {r['z0_cm']:.4f} "
                    f"| {r['paul_ratio_r0sq_over_z0sq']:.3f} "
                    f"| {r['isotropy_min_over_max_eig']:.3f} "
                    f"| {freqs} "
                    f"| {r['trap_depth_max_eV_at_r0.05cm']:.2f} "
                    f"| {r['locus_offset_from_origin_um']:.1f} "
                    f"| {'✓' if r['stable'] else '✗'} |\n")
        f.write("\n*Paul ratio = r0²/z0² — ideal Paul trap is 2.0.*\n")
        f.write("*Isotropy = min(Hessian eigenvalue) / max(Hessian eigenvalue). Ideal Paul = 0.25 (axial freq = √2 × radial).*\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
