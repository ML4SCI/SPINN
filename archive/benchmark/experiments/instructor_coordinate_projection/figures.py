"""Notebook-facing figure builders for the instructor coordinate-projection work.

Each function loads a results JSON and returns one matplotlib Figure, so a
notebook cell renders a plot by calling a single function. The data layout
matches what the ``run_*`` drivers write.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

RESULTS = Path(__file__).resolve().parent / "results"
MAIN_SCORECARD = Path(__file__).resolve().parents[2] / "results" / "scorecard.json"
DEVSIM_REFERENCE_GRADIENT = -1.44


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}; run the corresponding driver first")
    return json.loads(path.read_text())


def plot_pinn_vs_fem(scorecard_path: Path = MAIN_SCORECARD) -> Figure:
    """Shape gradient (sign) and training cost for PINN vs DEVSIM FEM."""

    data = _load(scorecard_path)
    results = [r for r in data["results"] if r.get("available")]
    labels = ["DEVSIM (FEM)" if r["name"] == "devsim_fem" else r["name"].replace(
        "coordinate_projection_pinn_seed", "PINN s") for r in results]
    ref = next((r for r in results if r["name"] == "devsim_fem"), None)
    ref_rel = (ref["gradient"] / ref["objective"]) if ref and ref["objective"] else None

    def rel(r):
        return r["gradient"] / r["objective"] if r.get("gradient") is not None and r.get("objective") else None

    sensitivities = [rel(r) for r in results]
    costs = [r["setup_seconds"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colors = ["#1b7837" if (s is not None and ref_rel is not None and (s >= 0) == (ref_rel >= 0))
              else "#c0392b" for s in sensitivities]
    axes[0].axhline(0.0, color="black", linewidth=1.4)
    axes[0].bar(labels, [s if s is not None else 0 for s in sensitivities], color=colors,
                edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel(r"shape sensitivity $d(\log J)/dd$")
    axes[0].set_title("Shape gradient: red bars point the WRONG way")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(labels, costs, color="#2c7fb8", edgecolor="black", linewidth=0.6)
    axes[1].set_ylabel("training / setup cost [s]")
    axes[1].set_title("Cost: PINN training vs. FEM solve")
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle("Paul-trap design: PINN vs. finite-element solver", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_instructor_map_diagnostics(path: Path = RESULTS / "instructor_comparison.json") -> Figure:
    """Identity fidelity and Jacobian determinant of the instructor map per seed."""

    report = _load(path)
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
    return fig


def plot_step_sweep(path: Path = RESULTS / "step_sweep.json") -> Figure:
    """Does the PINN gradient sign flip with more training; instructor fidelity vs steps."""

    records = _load(path)
    pinn = [r for r in records if r["method"] == "existing_coordinate_projection_pinn"]
    instructor = [r for r in records if r["method"] == "instructor_coordinate_projection"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    if pinn:
        steps = [r["steps"] for r in pinn]
        grads = [r["gradient"] for r in pinn]
        axes[0].axhline(0.0, color="black", linewidth=1.2)
        axes[0].axhline(DEVSIM_REFERENCE_GRADIENT, color="#1b7837", linestyle="--",
                        label=f"DEVSIM reference = {DEVSIM_REFERENCE_GRADIENT}")
        colors = ["#1b7837" if g < 0 else "#c0392b" for g in grads]
        axes[0].scatter(steps, grads, color=colors, zorder=3, s=60)
        for r in pinn:
            axes[0].annotate(f"{r['gradient']:+.1f}", (r["steps"], r["gradient"]),
                             fontsize=9, ha="left", va="bottom")
        axes[0].set_xlabel("training steps")
        axes[0].set_ylabel(r"shape gradient $dJ/dd$")
        axes[0].set_title("Existing PINN: sign flips negative with more training")
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
    return fig
