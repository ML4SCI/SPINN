"""Train the seeded-holes isotopic PINN — shape discovery with NO hyperbolic prior.

The electrodes are simple analytic "holes" (a torus for the ring, two axial blobs
for the endcaps) seeded at the electrode locations with reflection symmetry across
the x, y, z planes. A topology-preserving, symmetry-preserving diffeomorphism
x -> x + delta(x) deforms those seeds into discovered electrode shapes. The seeds
are FIXED; only the deformation (and the bulk Laplace correction) are learned.

Boundary conditions:
  - ring / endcap surfaces: HARD (built into the model via the Shepard blend)
  - world box: SOFT (phi -> 0), enforced as a loss penalty here

Objective: pure Laplace PDE residual + soft world BC. The structural prior is
topology + symmetry only, so this is a fair, shape-free counterpart to the
vanilla soft-BC baseline.

Usage:
    python -m PINN.Classical_3D_Paul.train_rf_seeded --steps 20000
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import boundaries, geometry as g
from .model import IsotopicSeededHoles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="PDE collocation points per step. Larger batch averages out the "
                             "hard-draw spikes that otherwise make the loss bounce ~100x.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000,
                        help="linear LR warmup steps (avoids early high-LR transients).")
    parser.add_argument("--resample_every", type=int, default=500)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128, 128])
    parser.add_argument("--w0", type=float, default=15.0)
    parser.add_argument("--delta_scale", type=float, default=0.02,
                        help="max displacement magnitude [cm] of the deformation field. Keep "
                             "small vs the seed feature sizes / inter-hole gaps so the warp "
                             "stays a topology-preserving diffeomorphism.")
    parser.add_argument("--seed_buffer", type=float, default=0.008,
                        help="exclude PDE collocation points within this distance [cm] of the "
                             "seeded electrode surfaces. Only needs to cover the ~1e-4 softabs "
                             "kink at s=0 plus the ~0.007 deformation reach; kept small so the "
                             "trap center (~0.031 cm from the endcaps) stays PDE-constrained. "
                             "Larger values eat the trap center and corrupt its metrics.")
    parser.add_argument("--bc_weight", type=float, default=100.0)
    parser.add_argument("--center_radius", type=float, default=0.03,
                        help="radius [cm] of the dense center-anchor ball around the seed center "
                             "(origin), where harmonicity is enforced with extra weight.")
    parser.add_argument("--center_weight", type=float, default=10.0,
                        help="weight on the center-anchor PDE residual. Pins a clean harmonic RF "
                             "null (hence a central, stable trap locus) at the center of the seeded "
                             "holes -- improves harmonicity where it matters and stabilizes training.")
    parser.add_argument("--center_points", type=int, default=512)
    parser.add_argument("--axis_buffer", type=float, default=0.0,
                        help="exclude PDE collocation points with sqrt(x^2+y^2) < this [cm]. "
                             "With the smooth polynomial torus distance the on-axis cone is gone, "
                             "so this defaults to 0 (PDE enforced through the trap center). Kept as "
                             "a knob only for diagnostics.")
    parser.add_argument("--ring_R", type=float, default=g.r0)
    parser.add_argument("--ring_a", type=float, default=0.03)
    parser.add_argument("--cap_d", type=float, default=g.z0)
    parser.add_argument("--cap_a", type=float, default=0.04)
    parser.add_argument("--out", type=str, default="pinn_rf_seeded_params.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule", choices=["warmcos", "cosine", "constant"], default="warmcos")
    args = parser.parse_args()

    print("Seeded holes (FIXED, deform-only):")
    print(f"  ring torus:  R = {args.ring_R:.4f}, tube a = {args.ring_a:.4f}  (phi = 1)")
    print(f"  endcap blobs: |z| = {args.cap_d:.4f}, radius a = {args.cap_a:.4f}  (phi = 0)")
    print(f"  delta_scale = {args.delta_scale:.4f} cm   symmetry = x,y,z reflections")

    # ----------------------------
    # Model
    # ----------------------------
    model = IsotopicSeededHoles(
        ring_R=args.ring_R, ring_a=args.ring_a,
        cap_d=args.cap_d, cap_a=args.cap_a,
        delta_scale=args.delta_scale,
        siren_hidden=tuple(args.hidden),
        w0=args.w0,
        world_half=g.world_half,
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.zeros(3))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  SIREN params: {n_params}  (output_dim=4: 3 displacement + 1 bulk)")

    def phi_single(p, x):
        return model.apply(p, x)
    phi_batch = jax.vmap(phi_single, in_axes=(None, 0))

    def laplacian_single(p, x):
        return jnp.trace(jax.hessian(phi_single, argnums=1)(p, x))
    laplacian_batch = jax.vmap(laplacian_single, in_axes=(None, 0))

    def loss_fn(params, xs, world_pts, center_pts):
        lap = laplacian_batch(params, xs)
        loss_pde = jnp.mean(lap ** 2)
        # Center anchor: enforce harmonicity densely in a small ball at the seed
        # center, with extra weight, so the trap locus is a clean harmonic saddle.
        lap_c = laplacian_batch(params, center_pts)
        loss_center = jnp.mean(lap_c ** 2)
        loss_bc = jnp.mean(phi_batch(params, world_pts) ** 2)   # world -> 0 (soft)
        total = loss_pde + args.center_weight * loss_center + args.bc_weight * loss_bc
        return total, (loss_pde, loss_center, loss_bc)

    if args.schedule == "warmcos":
        warmup = min(args.warmup, max(1, args.steps // 2))
        sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr,
            warmup_steps=warmup, decay_steps=args.steps,
            end_value=args.lr * 0.01)
    elif args.schedule == "cosine":
        sched = optax.cosine_decay_schedule(args.lr, decay_steps=args.steps)
    else:
        sched = optax.constant_schedule(args.lr)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = optim.init(params)

    @jax.jit
    def train_step(params, opt_state, xs, world_pts, center_pts):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, xs, world_pts, center_pts)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    # ----------------------------
    # Training
    # ----------------------------
    rng = np.random.default_rng(args.seed)
    world_np = boundaries.sample_world_surface(n_per_face=1500)

    def sub(arr, n, rng):
        idx = rng.choice(len(arr), min(n, len(arr)), replace=False)
        return jnp.asarray(arr[idx])

    def _seed_dists(pts):
        """Undeformed signed distances to the seeded ring torus + endcap blobs."""
        rho = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        s_ring = np.sqrt((rho - args.ring_R) ** 2 + pts[:, 2] ** 2) - args.ring_a
        s_cap_p = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + (pts[:, 2] - args.cap_d) ** 2) - args.cap_a
        s_cap_m = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + (pts[:, 2] + args.cap_d) ** 2) - args.cap_a
        return rho, s_ring, np.minimum(np.abs(s_cap_p), np.abs(s_cap_m))

    def sample_air(n, rng):
        """Air collocation points, excluding a z-axis tube and thin shells around
        the seeded electrode surfaces (where the distance field has kinks)."""
        out = np.empty((0, 3), dtype=np.float32)
        while len(out) < n:
            pts = boundaries.sample_air_points(3 * n, rng, boundary_buffer=0.01)
            rho, s_ring, d_cap = _seed_dists(pts)
            keep = (rho >= args.axis_buffer) & \
                   (np.abs(s_ring) >= args.seed_buffer) & \
                   (d_cap >= args.seed_buffer)
            out = np.concatenate([out, pts[keep]], axis=0)
        return jnp.asarray(out[:n])

    def sample_center(n, rng):
        """Dense cloud in a ball of radius center_radius around the seed center
        (origin), excluding the electrode-surface shells. The harmonicity anchor."""
        out = np.empty((0, 3), dtype=np.float32)
        while len(out) < n:
            p = rng.uniform(-args.center_radius, args.center_radius, (4 * n, 3)).astype(np.float32)
            r = np.linalg.norm(p, axis=1)
            _, s_ring, d_cap = _seed_dists(p)
            keep = (r <= args.center_radius) & (np.abs(s_ring) >= args.seed_buffer) & (d_cap >= args.seed_buffer)
            out = np.concatenate([out, p[keep]], axis=0)
        return jnp.asarray(out[:n])

    xs = sample_air(args.batch_size, rng)
    world_sub = sub(world_np, 512, rng)
    center_pts = sample_center(args.center_points, np.random.default_rng(args.seed + 777))

    # Fixed, held-out validation set (never resampled) -> a clean convergence
    # signal not contaminated by per-step resample variance. Includes a held-out
    # center cloud so best-val checkpointing tracks CENTER harmonicity too.
    val_rng = np.random.default_rng(args.seed + 12345)
    xs_val = jnp.concatenate([sample_air(1536, val_rng),
                              sample_center(512, np.random.default_rng(args.seed + 99))], axis=0)

    @jax.jit
    def val_pde_fn(params):
        return jnp.mean(laplacian_batch(params, xs_val) ** 2)

    _, (lp0, lc0, lb0) = loss_fn(params, xs, world_sub, center_pts)
    print(f"\nInitial losses: PDE = {float(lp0):.4e},  center = {float(lc0):.4e},  BC = {float(lb0):.4e}")

    t0 = time.time()
    print(f"\nTraining {args.steps} steps...")
    log = []
    val_log = []
    best_val = float("inf")
    best_params = jax.tree_util.tree_map(lambda a: np.asarray(a), params)
    for step in range(args.steps):
        if step > 0 and step % args.resample_every == 0:
            xs = sample_air(args.batch_size, rng)
            world_sub = sub(world_np, 512, rng)
        params, opt_state, loss, (loss_pde, loss_center, loss_bc) = train_step(
            params, opt_state, xs, world_sub, center_pts)
        log.append((float(loss), float(loss_pde), float(loss_center), float(loss_bc)))
        if step % 200 == 0 or step == args.steps - 1:
            valp = float(val_pde_fn(params))
            val_log.append((step, valp))
            if valp < best_val:
                best_val = valp
                best_params = jax.tree_util.tree_map(lambda a: np.asarray(a), params)
            print(f"  step {step:6d}  loss={float(loss):.3e}  "
                  f"(PDE {float(loss_pde):.2e},  center {float(loss_center):.2e},  BC {float(loss_bc):.2e})  "
                  f"val_PDE={valp:.2e}  [{time.time() - t0:.1f}s]")

    print(f"\nTotal training time: {time.time() - t0:.1f}s")
    print(f"Best val PDE seen: {best_val:.4e}.  Saving FINAL params (warmup->cosine settles the "
          f"end state; best-val-by-PDE can reward trivial flat/trapless fields, so we don't select on it).")

    out_path = Path(__file__).parent / "models" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "params": jax.tree_util.tree_map(lambda a: np.asarray(a), params),
            "config": {
                "hidden": list(args.hidden),
                "w0": args.w0,
                "V_RF": g.V_RF,
                "world_half": g.world_half,
                "ring_R": args.ring_R,
                "ring_a": args.ring_a,
                "cap_d": args.cap_d,
                "cap_a": args.cap_a,
                "delta_scale": args.delta_scale,
                "model_type": "isotopic_seeded",
                "best_val_pde": best_val,
            },
            "log": log,
            "val_log": val_log,
        }, f)
    print(f"Saved trained PINN to {out_path}")


if __name__ == "__main__":
    main()
