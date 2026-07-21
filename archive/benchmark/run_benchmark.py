#!/usr/bin/env python3
"""Run the PINN-vs-classical Paul-trap design benchmark and report results.

Usage (results first; sensible defaults)::

    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --training-steps 4000 --plot

Outputs land in ``benchmark/results/``:

* ``scorecard.json``   -- machine-readable full result,
* ``scorecard.md``     -- the comparison table you actually read,
* ``gradient_comparison.png`` (with ``--plot``).

The DEVSIM finite-element method is the reference. Each other method's shape
gradient is scored against it. Because the PINN works in a normalized potential
while DEVSIM works in volts, the headline accuracy metric is the *relative*
(logarithmic) sensitivity d(log J)/d(distance), which is scale-invariant and so
compares the two fairly; raw gradients are also reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPINN_ROOT = Path(__file__).resolve().parents[1]
if str(SPINN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINN_ROOT))

import json

from benchmark.methods import (
    CoordinateProjectionPINNMethod,
    DevsimFiniteDifferenceMethod,
    ShapeMethod,
)
from benchmark.task import DesignTask, GradientComparison, MethodResult, Scorecard

DEFAULT_RESULTS = SPINN_ROOT / "benchmark" / "results"


def relative_sensitivity(result: MethodResult) -> float | None:
    """Scale-invariant gradient d(log J)/d(distance) = (dJ/dd) / J."""

    if result.gradient is None or result.objective in (None, 0.0):
        return None
    return result.gradient / result.objective


def score_against_reference(
    results: list[MethodResult],
    reference: MethodResult,
) -> None:
    """Attach a GradientComparison to every non-reference method in place."""

    reference_rel = relative_sensitivity(reference)
    for result in results:
        if result.name == reference.name or not result.available:
            continue
        method_rel = relative_sensitivity(result)
        if method_rel is None or reference_rel is None:
            continue
        result.gradient_vs_reference = GradientComparison(
            method_gradient=method_rel,
            reference_gradient=reference_rel,
        )


def build_methods(args: argparse.Namespace) -> list[ShapeMethod]:
    methods: list[ShapeMethod] = [
        DevsimFiniteDifferenceMethod(
            cache_dir=args.results_dir / "devsim_cache",
            world_mesh_size=args.world_mesh_size,
            electrode_mesh_size=args.electrode_mesh_size,
        ),
    ]
    for seed in args.pinn_seeds:
        method = CoordinateProjectionPINNMethod(
            cache_dir=args.results_dir / f"pinn_cache_seed{seed}",
            training_steps=args.training_steps,
            seed=seed,
        )
        # Disambiguate multiple PINN rows in the scorecard.
        method.name = (
            "coordinate_projection_pinn"
            if len(args.pinn_seeds) == 1
            else f"coordinate_projection_pinn_seed{seed}"
        )
        methods.append(method)
    return methods


def run(args: argparse.Namespace) -> Scorecard:
    task = DesignTask(
        distance=args.distance,
        gradient_epsilon=args.gradient_epsilon,
        world_size_x=2.5,
        world_size_y=2.0,
    )
    methods = build_methods(args)

    results: list[MethodResult] = []
    reference: MethodResult | None = None
    for method in methods:
        print(f"[benchmark] running {method.name} ...", flush=True)
        result = method.evaluate(task)
        results.append(result)
        if method.is_reference and result.available:
            reference = result
        status = "ok" if result.available else f"unavailable ({result.notes})"
        print(f"[benchmark]   {method.name}: {status}", flush=True)

    if reference is not None:
        score_against_reference(results, reference)

    return Scorecard(
        task=task,
        reference_method=reference.name if reference else "none",
        results=results,
    )


# -- reporting -------------------------------------------------------------


def format_markdown(scorecard: Scorecard) -> str:
    lines = [
        "# Paul-Trap Design Benchmark: PINN vs. Classical FEM",
        "",
        f"**Task:** maximize quadrupole efficiency at electrode distance "
        f"`{scorecard.task.distance}` cm (interval "
        f"`[{scorecard.task.distance_min}, {scorecard.task.distance_max}]`).",
        f"**Reference (ground truth):** `{scorecard.reference_method}`.",
        "",
        "## Accuracy and cost",
        "",
        "| Method | Available | Objective J | d(log J)/dd | Grad rel. err vs ref | Sign agrees | Training s | Eval s | Deterministic |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in scorecard.results:
        rel = relative_sensitivity(result)
        cmp = result.gradient_vs_reference
        is_ref = result.name == scorecard.reference_method
        cached = result.available and not result.extra.get("freshly_trained", True)
        lines.append(
            "| {name} | {avail} | {obj} | {rel} | {err} | {sign} | {setup} | {ev:.3f} | {det} |".format(
                name=f"`{result.name}`" + (" (ref)" if is_ref else ""),
                avail="yes" if result.available else "no",
                obj=_fmt(result.objective),
                rel=_fmt(rel),
                err="reference" if is_ref else (_pct(cmp.relative_error) if cmp else "n/a"),
                sign="reference" if is_ref else (("yes" if cmp.sign_agrees else "NO") if cmp else "n/a"),
                setup=f"{result.setup_seconds:.1f}" + (" (cached)" if cached else ""),
                ev=result.eval_seconds,
                det="yes" if result.deterministic else "no",
            )
        )

    lines += ["", "## Verdict", "", *_verdict_lines(scorecard)]
    lines += ["", "## Notes per method", ""]
    for result in scorecard.results:
        lines.append(f"- `{result.name}`: {result.notes or '—'}")
    return "\n".join(lines) + "\n"


def _verdict_lines(scorecard: Scorecard) -> list[str]:
    ref = next((r for r in scorecard.results if r.name == scorecard.reference_method), None)
    pinns = [r for r in scorecard.results if r.gradient_vs_reference is not None]
    if ref is None or not ref.available:
        return ["- DEVSIM reference unavailable; no accuracy verdict can be drawn."]
    if not pinns:
        return ["- No method could be scored against the reference."]

    lines = []
    for pinn in pinns:
        cmp = pinn.gradient_vs_reference
        accurate = cmp.relative_error < 0.05 and cmp.sign_agrees
        cheaper = pinn.setup_seconds + pinn.eval_seconds < ref.eval_seconds
        lines.append(
            f"- `{pinn.name}`: shape-gradient relative error "
            f"{_pct(cmp.relative_error)} vs FEM; sign "
            f"{'agrees' if cmp.sign_agrees else 'DISAGREES'}; "
            f"end-to-end cost {pinn.setup_seconds + pinn.eval_seconds:.1f}s "
            f"vs FEM {ref.eval_seconds:.1f}s. "
            f"=> gradient {'usable' if accurate else 'NOT trustworthy'}, "
            f"and {'cheaper' if cheaper else 'not cheaper'} end-to-end."
        )
    return lines


def _fmt(value: float | None) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else "—"


def _pct(value: float | None) -> str:
    return f"{100.0 * value:.2f}%" if isinstance(value, (int, float)) else "—"


def _short_label(name: str, reference_method: str) -> str:
    """Compact, readable x-axis labels."""

    if name == reference_method:
        return "DEVSIM\n(FEM)"
    if "seed" in name:
        return "PINN\n" + name.split("seed")[-1].join(("seed ", ""))
    return name.replace("coordinate_projection_pinn", "PINN")


def plot_comparison(scorecard: Scorecard, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [r for r in scorecard.results if r.available and relative_sensitivity(r) is not None]
    if not scored:
        print("[benchmark] nothing to plot (no available methods).")
        return

    labels = [_short_label(r.name, scorecard.reference_method) for r in scored]
    sensitivities = [relative_sensitivity(r) for r in scored]
    costs = [r.setup_seconds + r.eval_seconds for r in scored]
    is_ref = [r.name == scorecard.reference_method for r in scored]
    ref_rel = next(
        (relative_sensitivity(r) for r in scored if r.name == scorecard.reference_method),
        None,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # -- Panel 1: shape gradient, colored by whether the SIGN matches FEM ----
    # Green = correct direction (agrees with FEM), red = wrong direction.
    def sign_color(value: float, ref: bool) -> str:
        if ref:
            return "#1b7837"  # reference itself
        if ref_rel is None:
            return "#2c7fb8"
        return "#1b7837" if (value >= 0) == (ref_rel >= 0) else "#c0392b"

    grad_colors = [sign_color(v, r) for v, r in zip(sensitivities, is_ref)]
    bars = axes[0].bar(labels, sensitivities, color=grad_colors, edgecolor="black", linewidth=0.6)
    axes[0].axhline(0.0, color="black", linewidth=1.4)  # the decisive line
    if ref_rel is not None:
        axes[0].axhline(ref_rel, color="#1b7837", linestyle="--", linewidth=1.2,
                        label=f"FEM reference = {ref_rel:.2f}")
        axes[0].legend(frameon=False, loc="upper left")

    # Annotate every bar with its value so the sign is unambiguous.
    for bar, value in zip(bars, sensitivities):
        offset = 0.6 if value >= 0 else -0.6
        axes[0].annotate(
            f"{value:+.2f}",
            (bar.get_x() + bar.get_width() / 2, value + offset),
            ha="center", va="bottom" if value >= 0 else "top", fontsize=9,
        )
    axes[0].set_ylabel(r"shape sensitivity  $d(\log J)/dd$")
    axes[0].set_title("Shape gradient: red bars point the WRONG way")
    axes[0].grid(alpha=0.25, axis="y")

    # Zoomed inset proving the FEM bar is genuinely below zero.
    if ref_rel is not None and ref_rel < 0:
        inset = axes[0].inset_axes([0.04, 0.30, 0.34, 0.32])
        inset.bar([0], [ref_rel], color="#1b7837", edgecolor="black", linewidth=0.6)
        inset.axhline(0.0, color="black", linewidth=1.2)
        inset.set_xticks([0])
        inset.set_xticklabels(["DEVSIM"], fontsize=8)
        inset.set_ylim(min(ref_rel * 1.6, -0.2), 0.3)
        inset.tick_params(labelsize=7)
        inset.set_title("zoom: FEM < 0", fontsize=8)
        axes[0].annotate("FEM is\nnegative", (0, ref_rel), (0.05, 6),
                         textcoords="axes fraction", fontsize=8, color="#1b7837")

    # -- Panel 2: cost --------------------------------------------------------
    # The bar is the one-time training/setup cost; evaluation is milliseconds
    # and reported separately. A bar loaded from cache this run is hatched and
    # labelled so its wall clock is not mistaken for cost paid this run.
    training_costs = [r.setup_seconds for r in scored]
    eval_costs = [r.eval_seconds for r in scored]
    fresh = [bool(r.extra.get("freshly_trained", True)) for r in scored]
    cost_colors = ["#1b7837" if r else "#2c7fb8" for r in is_ref]
    cost_bars = axes[1].bar(
        labels, training_costs, color=cost_colors, edgecolor="black", linewidth=0.6,
        hatch=["" if f else "//" for f in fresh],
    )
    for bar, train_s, eval_s, was_fresh in zip(cost_bars, training_costs, eval_costs, fresh):
        tag = "" if was_fresh else " (cached)"
        axes[1].annotate(
            f"{train_s:.0f}s{tag}\n+{eval_s * 1e3:.0f}ms eval",
            (bar.get_x() + bar.get_width() / 2, train_s),
            ha="center", va="bottom", fontsize=8,
        )
    axes[1].set_ylabel("one-time training / setup cost [s]")
    axes[1].set_title("Cost: PINN training vs. FEM solve (eval time labelled separately)")
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Paul-trap shape design: classical FEM beats the PINN on direction AND speed",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=float, default=0.125)
    parser.add_argument("--gradient-epsilon", type=float, default=0.0025)
    parser.add_argument("--training-steps", type=int, default=4000)
    parser.add_argument("--pinn-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--world-mesh-size", type=float, default=0.04)
    parser.add_argument("--electrode-mesh-size", type=float, default=0.004)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    scorecard = run(args)

    json_path = args.results_dir / "scorecard.json"
    md_path = args.results_dir / "scorecard.md"
    json_path.write_text(json.dumps(scorecard.to_dict(), indent=2))
    markdown = format_markdown(scorecard)
    md_path.write_text(markdown)

    if args.plot:
        plot_comparison(scorecard, args.results_dir / "gradient_comparison.png")

    print("\n" + markdown)
    print(f"[benchmark] wrote {json_path}")
    print(f"[benchmark] wrote {md_path}")


if __name__ == "__main__":
    main()
