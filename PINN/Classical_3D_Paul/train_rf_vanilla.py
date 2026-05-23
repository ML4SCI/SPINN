"""Train the 'vanilla' PINN baseline (tanh MLP + soft BCs, no analytic lift).

JAX/Flax port of PINN/Vanilla_Paul_Trap_PINN/Paul_Trap_PINN.py, adapted to the
3D Classical Paul geometry so it can be compared directly against the SIREN /
analytic-lift / isotopic variants and validated against the same DEVSIM
ground truth.

Faithful to the vanilla's architectural choices:
  - plain tanh MLP (no lift, no hard-BC factoring)
  - soft BCs on EVERY surface (ring -> 1, endcaps -> 0, world -> 0), bc_weight
  - PDE residual loss = mean (Laplacian phi)^2

Differences made to give it a fair shot (a 'strong' baseline):
  - normalized scale (ring = 1, ground = 0; multiply by V_RF afterward) — keeps
    the loss O(1) instead of O(V_RF^2)
  - gradient clipping (matches the vanilla)

Usage:
    python -m PINN.Classical_3D_Paul.train_rf_vanilla --steps 3000
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
from .model import TanhMLP, VanillaSoftBC


def _normalizer(x):
    return x / g.world_half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="PDE collocation points per step (matched to the seeded model).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--resample_every", type=int, default=500)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128, 128])
    parser.add_argument("--bc_weight", type=float, default=100.0)
    parser.add_argument("--z0_truth_scale", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="pinn_rf_vanilla_params.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule", choices=["warmcos", "cosine", "plateau", "constant"], default="warmcos")
    args = parser.parse_args()

    # ----------------------------
    # Soft-BC point clouds: FULL ring + FULL endcap + world surfaces.
    # The vanilla PINN enforces all boundaries softly, so we need the complete
    # electrode surfaces (inner hyperboloid + truncation), unlike the analytic
    # variants which hard-enforce the inner hyperboloid via the lift.
    # ----------------------------
    z0_truth = g.z0 * args.z0_truth_scale
    print(f"Geometry: z0_truth = {z0_truth:.5f} (scale {args.z0_truth_scale})")

    # boundaries.sample_ring_surface / sample_endcap_surface use the module-level
    # ideal z0; for z0_truth_scale != 1 we'd need to re-sample. For now this
    # script targets the ideal geometry (z0_truth_scale = 1.0).
    if args.z0_truth_scale != 1.0:
        raise SystemExit("train_rf_vanilla currently supports z0_truth_scale=1.0 only "
                         "(boundaries sampler uses the ideal z0).")

    print("Sampling full electrode + world surfaces (all soft BC)...")
    ring_np = boundaries.sample_ring_surface(n_per_face=2000)               # V = 1
    endcap_np = np.concatenate([
        boundaries.sample_endcap_surface(+1, n_per_face=2000),
        boundaries.sample_endcap_surface(-1, n_per_face=2000),
    ], axis=0)                                                              # V = 0
    world_np = boundaries.sample_world_surface(n_per_face=2000)             # V = 0
    print(f"  ring (V=1):   {ring_np.shape[0]}")
    print(f"  endcaps (V=0): {endcap_np.shape[0]}")
    print(f"  world (V=0):  {world_np.shape[0]}")

    # ----------------------------
    # Model
    # ----------------------------
    mlp = TanhMLP(hidden=tuple(args.hidden))
    bc = VanillaSoftBC()
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = mlp.init(init_key, jnp.zeros((1, 3)))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  tanh MLP params: {n_params}")

    def phi_single(p, x):
        return bc.phi(mlp.apply, p, x, _normalizer)
    phi_batch = jax.vmap(phi_single, in_axes=(None, 0))

    def laplacian_single(p, x):
        return jnp.trace(jax.hessian(phi_single, argnums=1)(p, x))
    laplacian_batch = jax.vmap(laplacian_single, in_axes=(None, 0))

    def loss_fn(params, xs, ring_pts, endcap_pts, world_pts):
        lap = laplacian_batch(params, xs)
        loss_pde = jnp.mean(lap ** 2)
        loss_bc = (jnp.mean((phi_batch(params, ring_pts) - 1.0) ** 2)
                   + jnp.mean(phi_batch(params, endcap_pts) ** 2)
                   + jnp.mean(phi_batch(params, world_pts) ** 2))
        return loss_pde + args.bc_weight * loss_bc, (loss_pde, loss_bc)

    if args.schedule == "warmcos":
        warmup = min(args.warmup, max(1, args.steps // 2))
        sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr,
            warmup_steps=warmup, decay_steps=args.steps, end_value=args.lr * 0.01)
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    elif args.schedule == "cosine":
        sched = optax.cosine_decay_schedule(args.lr, decay_steps=args.steps)
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    elif args.schedule == "plateau":
        # Mirror the vanilla's ReduceLROnPlateau via optax.contrib
        optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(args.lr),
            optax.contrib.reduce_on_plateau(factor=0.5, patience=300, min_scale=1e-3),
        )
    else:
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = optim.init(params)

    use_plateau = (args.schedule == "plateau")

    @jax.jit
    def train_step(params, opt_state, xs, ring_pts, endcap_pts, world_pts):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, xs, ring_pts, endcap_pts, world_pts)
        if use_plateau:
            updates, opt_state = optim.update(grads, opt_state, params,
                                              value=loss)
        else:
            updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    # ----------------------------
    # Training
    # ----------------------------
    rng = np.random.default_rng(args.seed)

    def sub(arr, n, rng):
        idx = rng.choice(len(arr), min(n, len(arr)), replace=False)
        return jnp.asarray(arr[idx])

    xs = jnp.asarray(boundaries.sample_air_points(args.batch_size, rng))
    ring_sub = sub(ring_np, 512, rng)
    endcap_sub = sub(endcap_np, 512, rng)
    world_sub = sub(world_np, 512, rng)

    # Held-out validation set (fixed) -> clean PDE-residual convergence signal,
    # not contaminated by per-step resample variance. Mirrors train_rf_seeded.
    xs_val = jnp.asarray(boundaries.sample_air_points(2048, np.random.default_rng(args.seed + 12345)))

    @jax.jit
    def val_pde_fn(params):
        return jnp.mean(laplacian_batch(params, xs_val) ** 2)

    t0 = time.time()
    print(f"\nTraining {args.steps} steps...")
    log = []
    val_log = []
    best_val = float("inf")
    best_params = jax.tree_util.tree_map(lambda a: np.asarray(a), params)
    for step in range(args.steps):
        if step > 0 and step % args.resample_every == 0:
            xs = jnp.asarray(boundaries.sample_air_points(args.batch_size, rng))
            ring_sub = sub(ring_np, 512, rng)
            endcap_sub = sub(endcap_np, 512, rng)
            world_sub = sub(world_np, 512, rng)
        params, opt_state, loss, (loss_pde, loss_bc) = train_step(
            params, opt_state, xs, ring_sub, endcap_sub, world_sub)
        log.append((float(loss), float(loss_pde), float(loss_bc)))
        if step % 200 == 0 or step == args.steps - 1:
            valp = float(val_pde_fn(params))
            val_log.append((step, valp))
            if valp < best_val:
                best_val = valp
                best_params = jax.tree_util.tree_map(lambda a: np.asarray(a), params)
            print(f"  step {step:6d}  loss={float(loss):.3e}  "
                  f"(PDE {float(loss_pde):.2e},  BC {float(loss_bc):.2e})  "
                  f"val_PDE={valp:.2e}  [{time.time() - t0:.1f}s]")

    print(f"\nTotal training time: {time.time() - t0:.1f}s")
    print(f"Best val PDE seen: {best_val:.4e}.  Saving FINAL params (best-val-by-PDE can reward "
          f"trivial flat/trapless fields, so we don't select on it).")

    out_path = Path(__file__).parent / "models" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "params": jax.tree_util.tree_map(lambda a: np.asarray(a), params),
            "config": {
                "hidden": list(args.hidden),
                "V_RF": g.V_RF,
                "world_half": g.world_half,
                "model_type": "vanilla_tanh_soft",
                "best_val_pde": best_val,
            },
            "log": log,
            "val_log": val_log,
        }, f)
    print(f"Saved trained PINN to {out_path}")


if __name__ == "__main__":
    main()
