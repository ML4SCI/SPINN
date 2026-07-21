#!/usr/bin/env python3
"""Step-count sweep: is the wrong-sign gradient an under-training artifact?

For the existing coordinate-projection PINN this trains at several step counts
and records the shape gradient ``dJ/dd``, so we can see whether more training
flips its sign to agree with the DEVSIM reference (negative).

For the instructor coordinate map there is no ``dJ/dd`` to flip; the sweep
instead records identity fidelity and Jacobian conditioning versus step count,
showing whether the map keeps improving with training.

Results are written to ``results/step_sweep.json`` and a convergence figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPINN_ROOT = Path(__file__).resolve().parents[3]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

from benchmark.methods.pinn import CoordinateProjectionPINNMethod
from benchmark.task import DesignTask

from .diagnostics import map_diagnostics
from .model import (
    CoordinateProjectionNet,
    TrainingConfig,
    TrapGeometry,
    train_identity_map,
)

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
REFERENCE_GRADIENT = -1.44


def sweep_existing_pinn(
    step_counts: list[int],
    seeds: list[int],
    cache_root: Path,
    task: DesignTask,
) -> list[dict]:
    records = []
    for steps in step_counts:
        for seed in seeds:
            method = CoordinateProjectionPINNMethod(
                cache_dir=cache_root / f"pinn_steps{steps}_seed{seed}",
                training_steps=steps,
                seed=seed,
            )
            result = method.evaluate(task)
            gradient = result.gradient
            records.append({
                "method": "existing_coordinate_projection_pinn",
                "steps": steps,
                "seed": seed,
                "gradient": gradient,
                "objective": result.objective,
                "sign_matches_devsim": (gradient < 0) if gradient is not None else None,
                "train_seconds": result.setup_seconds,
            })
            print(
                f"[sweep] PINN steps={steps} seed={seed}: "
                f"dJ/dd={gradient:+.3f} "
                f"(sign {'OK' if gradient is not None and gradient < 0 else 'wrong'})",
                flush=True,
            )
    return records


def sweep_instructor_map(
    step_counts: list[int],
    seeds: list[int],
    geometry: TrapGeometry,
) -> list[dict]:
    records = []
    for steps in step_counts:
        for seed in seeds:
            model = CoordinateProjectionNet(width=50, depth=6, zero_last_layer=False)
            config = TrainingConfig(steps=steps)
            training = train_identity_map(model, geometry, config, seed=seed)
            diagnostics = map_diagnostics(model, geometry, seed=seed)
            records.append({
                "method": "instructor_coordinate_projection",
                "steps": steps,
                "seed": seed,
                "train_seconds": training.train_seconds,
                **diagnostics,
            })
            print(
                f"[sweep] instructor steps={steps} seed={seed}: "
                f"rms={diagnostics['rms_identity_error']:.3e} "
                f"detJ in [{diagnostics['det_jacobian_min']:.3f}, "
                f"{diagnostics['det_jacobian_max']:.3f}]",
                flush=True,
            )
    return records


def plot_sweep(records: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pinn = [r for r in records if r["method"] == "existing_coordinate_projection_pinn"]
    instructor = [r for r in records if r["method"] == "instructor_coordinate_projection"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    if pinn:
        steps = [r["steps"] for r in pinn]
        grads = [r["gradient"] for r in pinn]
        axes[0].axhline(0.0, color="black", linewidth=1.2)
        axes[0].axhline(REFERENCE_GRADIENT, color="#1b7837", linestyle="--",
                        label=f"DEVSIM reference = {REFERENCE_GRADIENT}")
        axes[0].scatter(steps, grads, color="#c0392b", zorder=3)
        for r in pinn:
            axes[0].annotate(f"{r['gradient']:+.1f}", (r["steps"], r["gradient"]),
                             fontsize=8, ha="left", va="bottom")
        axes[0].set_xlabel("training steps")
        axes[0].set_ylabel(r"shape gradient  $dJ/dd$")
        axes[0].set_title("Existing PINN: does the sign flip with training?")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.25)

    if instructor:
        steps = [r["steps"] for r in instructor]
        rms = [r["rms_identity_error"] for r in instructor]
        axes[1].semilogy(steps, rms, marker="o", color="#2c7fb8")
        axes[1].set_xlabel("training steps")
        axes[1].set_ylabel("RMS identity error")
        axes[1].set_title("Instructor map: identity fidelity vs. training")
        axes[1].grid(alpha=0.25)

    fig.suptitle("Step-count sweep: under-training check", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[6000, 20000, 50000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--skip-instructor", action="store_true")
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    task = DesignTask()
    geometry = TrapGeometry()

    records: list[dict] = []
    if not args.skip_pinn:
        records += sweep_existing_pinn(
            args.steps, args.seeds, args.results_dir / "pinn_cache", task
        )
    if not args.skip_instructor:
        records += sweep_instructor_map(args.steps, args.seeds, geometry)

    (args.results_dir / "step_sweep.json").write_text(json.dumps(records, indent=2))
    plot_sweep(records, args.results_dir / "step_sweep.png")
    print(f"[sweep] wrote {args.results_dir / 'step_sweep.json'}")
    print(f"[sweep] wrote {args.results_dir / 'step_sweep.png'}")


if __name__ == "__main__":
    main()
