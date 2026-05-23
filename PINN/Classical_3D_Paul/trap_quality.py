"""Trap-quality diagnostics for a PINN-derived RF field.

Given a callable phi(x) -> scalar (the RF potential predicted by the PINN),
compute the pseudopotential

    psi(x) = (e^2 / 4 M Omega^2) * |grad phi(x)|^2

and report:

  - Locus: position of the trap minimum (gradient descent on psi)
  - Stability: are all Hessian eigenvalues at the minimum positive?
  - Isotropy: ratio of smallest to largest Hessian eigenvalue
  - Secular frequencies: omega_i = sqrt(eigval_i * 2 / M_ion)  (small-oscillation)
  - Trap depth: maximum psi within a chosen radial cap minus psi(x_min)
  - Sensitivity: how much x_min moves and how much depth changes when
    voltage scaling or lift shape (r0, z0) is perturbed by 1%

Usage as a script:
    python -m PINN.Classical_3D_Paul.trap_quality --params <file>.pkl

Usage as a library:
    from PINN.Classical_3D_Paul.trap_quality import evaluate
    metrics = evaluate(phi_fn, V_RF=300.0)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry as g


# Physical constants (match DEVSIM/visualize)
E_CHARGE = 1.602e-19
AMU = 1.66054e-27
M_CA = 40 * AMU
F_RF = 10.2e6
OMEGA_RF = 2 * np.pi * F_RF
SCALE = (E_CHARGE ** 2) / (4 * M_CA * OMEGA_RF ** 2)   # J / (V/m)^2


def make_psi(phi_fn):
    """Return psi(x) [eV] from phi(x) [V]. Includes the (V/cm)^2 -> (V/m)^2 factor."""
    grad_phi = jax.grad(phi_fn)
    def psi(x):
        g_phi = grad_phi(x)
        grad2_vm2 = 1.0e4 * jnp.dot(g_phi, g_phi)        # V/cm -> V/m squared
        return SCALE * grad2_vm2 / E_CHARGE              # J -> eV
    return psi


def find_locus(psi_fn, x_init=None, n_iter=500, lr=1e-4):
    """Gradient descent on psi to find the trap minimum.

    Returns (x_min, psi_min) as numpy arrays.
    """
    if x_init is None:
        x_init = np.array([0.0, 0.0, 0.0])
    grad_psi = jax.grad(psi_fn)
    x = jnp.array(x_init, dtype=jnp.float32)
    for _ in range(n_iter):
        g_ = grad_psi(x)
        x = x - lr * g_
    return np.asarray(x), float(psi_fn(x))


def hessian_at(psi_fn, x):
    """Hessian of psi at x (3x3 numpy array, in eV/cm^2)."""
    return np.asarray(jax.hessian(psi_fn)(jnp.asarray(x, dtype=jnp.float32)))


def secular_frequencies(eigvals_eV_per_cm2):
    """Convert Hessian eigenvalues (eV/cm^2 of psi at the minimum) to secular
    frequencies omega_i [rad/s] for a calcium-40 ion.

    omega^2 = (1/M) * d^2 U / dx^2,  U = e * psi  (potential energy in J)
    psi in [eV] * E_CHARGE -> J, position in [cm] -> m so divide by 1e-4.
    """
    second_deriv_J_per_m2 = np.asarray(eigvals_eV_per_cm2) * E_CHARGE * 1.0e4
    omegas2 = second_deriv_J_per_m2 / M_CA
    omegas = np.sqrt(np.clip(omegas2, 0.0, None))
    return omegas


def trap_depth(psi_fn, x_min, max_radius=0.05, n_samples=2000, seed=0):
    """Maximum psi on a sphere of radius `max_radius` around x_min minus psi(x_min).

    Approximates the depth of the trap well in eV.
    """
    rng = np.random.default_rng(seed)
    # Random directions on the unit sphere
    u = rng.standard_normal((n_samples, 3))
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    pts = x_min + max_radius * u
    psi_batch = jax.vmap(psi_fn)
    psi_vals = np.asarray(psi_batch(jnp.asarray(pts, dtype=jnp.float32)))
    psi_min = float(psi_fn(jnp.asarray(x_min, dtype=jnp.float32)))
    return float(psi_vals.max() - psi_min), float(psi_vals.mean() - psi_min)


def evaluate(phi_fn, V_RF, verbose=True, trap_radius=0.05):
    """Compute the full set of trap-quality metrics for a given phi_fn.

    phi_fn(x) should take a 3-vector and return phi in VOLTS.
    """
    psi_fn = make_psi(phi_fn)

    x_min, psi_min = find_locus(psi_fn)
    H = hessian_at(psi_fn, x_min)
    eigvals, eigvecs = np.linalg.eigh(H)        # ascending order
    stable = bool(np.all(eigvals > 0))
    isotropy = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else 0.0
    omegas = secular_frequencies(eigvals)
    freqs_MHz = omegas / (2 * np.pi) / 1e6
    depth_max, depth_mean = trap_depth(psi_fn, x_min, max_radius=trap_radius)

    metrics = {
        "x_min_cm": x_min,
        "psi_min_eV": psi_min,
        "hessian_eigvals_eV_per_cm2": eigvals,
        "stable": stable,
        "isotropy": isotropy,
        "secular_freqs_Hz": omegas / (2 * np.pi),
        "secular_freqs_MHz": freqs_MHz,
        "trap_depth_eV_at_radius": (trap_radius, depth_max),
        "trap_depth_mean_at_radius": (trap_radius, depth_mean),
    }

    if verbose:
        print("Trap-quality metrics")
        print(f"  Locus position (cm):   x = {x_min[0]:+.5f}, y = {x_min[1]:+.5f}, z = {x_min[2]:+.5f}")
        print(f"  psi at locus:          {psi_min:.4e} eV")
        print(f"  Stable (Hessian PSD):  {stable}")
        print(f"  Hessian eigenvalues:   {eigvals}")
        print(f"  Isotropy (eig_min/eig_max):  {isotropy:.4f}")
        print(f"  Secular freqs (MHz):   {freqs_MHz}")
        print(f"  Trap depth (max psi at r={trap_radius} cm vs min): {depth_max:.4f} eV")
        print(f"  Trap depth (mean psi at r={trap_radius} cm vs min): {depth_mean:.4f} eV")
    return metrics


def voltage_sensitivity(phi_fn, V_RF, delta_pct=0.01, trap_radius=0.05):
    """How much does the trap minimum move and depth change under a 1% V_RF perturbation?

    For pure RF the potential scales linearly in V_RF, so psi ~ V_RF^2 and
    x_min is V_RF-invariant. So this should give ~zero shift in x_min and
    depth scaled by (1 + delta_pct)^2 - 1 ~ 2*delta_pct. Useful as a
    self-consistency check.
    """
    base = evaluate(phi_fn, V_RF, verbose=False, trap_radius=trap_radius)
    phi_pert = lambda x: (1.0 + delta_pct) * phi_fn(x)
    pert = evaluate(phi_pert, V_RF, verbose=False, trap_radius=trap_radius)
    return {
        "x_min_shift_cm": np.linalg.norm(pert["x_min_cm"] - base["x_min_cm"]),
        "depth_relative_change": (pert["trap_depth_eV_at_radius"][1] / base["trap_depth_eV_at_radius"][1]
                                  if base["trap_depth_eV_at_radius"][1] > 0 else 0),
        "expected_depth_ratio": (1.0 + delta_pct) ** 2,
    }


def _build_phi(params_path):
    """Construct a phi callable from a trained-PINN params file."""
    from .model import (SIREN, HardBC, AnalyticHyperbolicBC,
                        AnalyticHyperbolicBC_Faded, IsotopicHyperbolicBC,
                        IsotopicTrainableShape, IsotopicSeededHoles,
                        TanhMLP, VanillaSoftBC)

    # Accept bare filename (look in models/) or explicit path
    p = Path(str(params_path))
    if not p.is_absolute() and not p.exists():
        candidate = Path(__file__).parent / "models" / p
        if candidate.exists():
            p = candidate
        else:
            legacy = Path(__file__).parent / p
            if legacy.exists():
                p = legacy
    with open(p, "rb") as f:
        bundle = pickle.load(f)
    cfg = bundle["config"]
    params = jax.tree_util.tree_map(jnp.asarray, bundle["params"])
    model_type = cfg.get("model_type", "")

    if model_type == "isotopic_trainable_shape":
        # The trainable-shape model is a single Flax module: just apply it.
        model = IsotopicTrainableShape(
            r0_init=cfg["r0_init"],
            z0_init=cfg["z0_init"],
            delta_scale=cfg["delta_scale"],
            siren_hidden=tuple(cfg["hidden"]),
            w0=cfg["w0"],
            world_half=cfg["world_half"],
        )
        def phi(x):
            return cfg["V_RF"] * model.apply(params, x)
        return phi, cfg

    if model_type == "isotopic_seeded":
        model = IsotopicSeededHoles(
            ring_R=cfg["ring_R"], ring_a=cfg["ring_a"],
            cap_d=cfg["cap_d"], cap_a=cfg["cap_a"],
            delta_scale=cfg["delta_scale"],
            siren_hidden=tuple(cfg["hidden"]),
            w0=cfg["w0"],
            world_half=cfg["world_half"],
        )
        def phi(x):
            return cfg["V_RF"] * model.apply(params, x)
        return phi, cfg

    if model_type == "vanilla_tanh_soft":
        mlp = TanhMLP(hidden=tuple(cfg["hidden"]))
        bc = VanillaSoftBC()
        def phi(x):
            return cfg["V_RF"] * bc.phi(mlp.apply, params, x, lambda y: y / cfg["world_half"])
        return phi, cfg

    output_dim = cfg.get("output_dim", 1)
    siren = SIREN(hidden=tuple(cfg["hidden"]), w0=cfg["w0"], w0_first=cfg["w0"], output_dim=output_dim)
    if model_type == "analytic_hyperbolic":
        bc = AnalyticHyperbolicBC(r0=cfg["r0"], z0=cfg["z0"])
    elif model_type == "analytic_hyperbolic_faded":
        bc = AnalyticHyperbolicBC_Faded(r0=cfg["r0"], z0=cfg["z0"], world_half=cfg["world_half"])
    elif model_type == "isotopic_hyperbolic":
        bc = IsotopicHyperbolicBC(r0=cfg["r0"], z0=cfg["z0"], delta_scale=cfg["delta_scale"])
    elif model_type == "":
        # Shepard hard-BC (regular PINN). Requires pA, pB boundary clouds.
        pA = jnp.asarray(bundle["pA"])
        pB = jnp.asarray(bundle["pB"])
        bc = HardBC(pA=pA, pB=pB)
    else:
        raise SystemExit(f"unsupported model_type {model_type}")

    def phi(x):
        return cfg["V_RF"] * bc.phi(siren.apply, params, x, lambda y: y / cfg["world_half"])
    return phi, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, required=True)
    parser.add_argument("--trap_radius", type=float, default=0.05,
                        help="radius (cm) over which to measure trap depth")
    args = parser.parse_args()

    phi, cfg = _build_phi(args.params)
    print(f"Model: {cfg.get('model_type', '<unknown>')}")
    if "r0_final" in cfg:
        print(f"  Trained r0 = {cfg['r0_final']:.5f}, z0 = {cfg['z0_final']:.5f}")
    elif "z0" in cfg:
        print(f"  Fixed lift z0 = {cfg['z0']:.5f}")
    print()
    evaluate(phi, V_RF=cfg["V_RF"], verbose=True, trap_radius=args.trap_radius)

    print("\nVoltage-sensitivity check (1% scaling of phi):")
    sens = voltage_sensitivity(phi, V_RF=cfg["V_RF"], delta_pct=0.01,
                                trap_radius=args.trap_radius)
    print(f"  locus shift:                {sens['x_min_shift_cm']*1e4:.3f} um  (should be ~0 for pure RF)")
    print(f"  depth ratio (pert / base):  {sens['depth_relative_change']:.4f}  "
          f"(expected {sens['expected_depth_ratio']:.4f})")


if __name__ == "__main__":
    main()
