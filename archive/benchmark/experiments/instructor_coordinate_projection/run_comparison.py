#!/usr/bin/env python3
"""Experiment 3: diagnose the instructor coordinate map against the trap problem.

This trains the instructor coordinate map and reports its identity fidelity and
Jacobian conditioning, presented next to the DEVSIM reference shape gradient and
the existing PINN's shape gradients (read from the main benchmark scorecard for
context). The instructor map does not emit ``dJ/dd``, so the comparison is about
whether the map is clean enough to ever support a correct shape gradient, not a
direct gradient-sign contest.

Outputs ``results/instructor_comparison.json`` and a diagnostics figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPINN_ROOT = Path(__file__).resolve().parents[3]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

from benchmark.methods.instructor_coordinate_projection import (
    InstructorCoordinateProjectionMethod,
)
from benchmark.task import DesignTask

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
MAIN_SCORECARD = SPINN_ROOT / "benchmark" / "results" / "scorecard.json"


def existing_gradient_context() -> dict:
    """Read DEVSIM and existing-PINN gradients from the main scorecard, if present."""

    if not MAIN_SCORECARD.exists():
        return {"available": False}
    data = json.loads(MAIN_SCORECARD.read_text())
    devsim = next((r for r in data["results"] if r["name"] == "devsim_fem"), None)
    pinns = [r for r in data["results"] if "pinn" in r["name"] and "instructor" not in r["name"]]
    return {
        "available": True,
        "devsim_gradient": devsim["gradient"] if devsim else None,
        "existing_pinn_gradients": {r["name"]: r["gradient"] for r in pinns},
    }


def run(steps: int, seeds: list[int]) -> dict:
    task = DesignTask()
    runs = []
    for seed in seeds:
        method = InstructorCoordinateProjectionMethod(training_steps=steps, seed=seed)
        result = method.evaluate(task)
        runs.append({"seed": seed, "train_seconds": result.setup_seconds, **result.extra})
        diag = result.extra
        print(
            f"[exp3] instructor seed={seed}: rms={diag['rms_identity_error']:.3e}, "
            f"detJ in [{diag['det_jacobian_min']:.3f}, {diag['det_jacobian_max']:.3f}], "
            f"cond_max={diag['condition_number_max']:.3f}",
            flush=True,
        )
    return {
        "task": {"distance": task.distance},
        "instructor_training_steps": steps,
        "instructor_runs": runs,
        "gradient_context": existing_gradient_context(),
    }


def plot_comparison(report: dict, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = report["instructor_runs"]
    seeds = [str(r["seed"]) for r in runs]
    rms = [r["rms_identity_error"] for r in runs]
    det_min = [r["det_jacobian_min"] for r in runs]
    det_max = [r["det_jacobian_max"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    axes[0].bar(seeds, rms, color="#2c7fb8", edgecolor="black", linewidth=0.6)
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("RMS identity error")
    axes[0].set_title("Instructor map: identity fidelity")
    axes[0].grid(alpha=0.25, axis="y")

    x = range(len(seeds))
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle=":")
    axes[1].fill_between(x, det_min, det_max, alpha=0.3, color="#1b7837")
    axes[1].plot(x, det_min, marker="o", color="#1b7837", label="min det J")
    axes[1].plot(x, det_max, marker="s", color="#7fbf7b", label="max det J")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(seeds)
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel(r"$\det J$")
    axes[1].set_title("Instructor map: Jacobian determinant (want > 0, near 1)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle("Instructor coordinate map diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = run(args.steps, args.seeds)
    (args.results_dir / "instructor_comparison.json").write_text(json.dumps(report, indent=2))
    plot_comparison(report, args.results_dir / "instructor_comparison.png")
    print(f"[exp3] wrote {args.results_dir / 'instructor_comparison.json'}")
    print(f"[exp3] wrote {args.results_dir / 'instructor_comparison.png'}")


if __name__ == "__main__":
    main()
