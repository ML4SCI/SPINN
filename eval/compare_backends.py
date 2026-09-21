"""Aggregate joint-run metrics into the backend comparison and hypothesis table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_ORDER = ["mlp", "pixel", "pig"]
VALID_DETJ_EPS = 0.2
REQUIRED_PLOTS = [
    "fixed_geometry_potential_fields.png",
    "fixed_geometry_error_heatmaps.png",
    "fixed_geometry_pde_residual_maps.png",
    "fixed_geometry_eta_error_bar.png",
    "eta_pinn_vs_epoch.png",
    "eta_fem_vs_checkpoint.png",
    "pinn_eta_vs_fem_eta_scatter.png",
    "initial_vs_optimized_electrodes.png",
    "deformation_field.png",
    "jacobian_determinant_heatmap.png",
    "loss_terms_vs_epoch.png",
    "constraints_vs_epoch.png",
    "pixel_feature_activity.png",
    "pig_gaussian_centers.png",
    "pig_gaussian_covariances.png",
]


def summarize(metrics_csv: Path) -> dict:
    df = pd.read_csv(metrics_csv)
    df = df[df["eta_fem"].notna()] if "eta_fem" in df else df
    summary: dict[str, dict] = {}
    for backend, group in df.groupby("backend"):
        finals = group.sort_values("step").groupby("seed").tail(1)
        if "min_detJ" in group:
            valid = group[group["min_detJ"].astype(float) > VALID_DETJ_EPS]
        else:
            valid = group
        best = valid.sort_values("eta_fem").groupby("seed").tail(1) if not valid.empty else valid
        final_eta_fem = finals["eta_fem"].astype(float)
        best_eta_fem = best["eta_fem"].astype(float) if not best.empty else pd.Series(dtype=float)
        mismatch_source = best if not best.empty else finals
        mismatch = (
            mismatch_source["eta_pinn"].astype(float) - mismatch_source["eta_fem"].astype(float)
        ).abs()
        min_detj = group["min_detJ"].astype(float).min() if "min_detJ" in group else float("nan")
        summary[backend] = {
            "final_eta_fem_mean": float(final_eta_fem.mean()),
            "final_eta_fem_std": float(final_eta_fem.std(ddof=0)) if len(final_eta_fem) > 1 else 0.0,
            "best_valid_eta_fem_mean": float(best_eta_fem.mean()) if not best_eta_fem.empty else float("nan"),
            "best_valid_eta_fem_std": float(best_eta_fem.std(ddof=0)) if len(best_eta_fem) > 1 else 0.0,
            "best_valid_step_mean": float(best["step"].astype(float).mean()) if not best.empty else float("nan"),
            "eta_mismatch_mean": float(mismatch.mean()),
            "min_detJ_over_run": min_detj,
            "n_seeds": int(finals["seed"].nunique()),
            "n_valid_fem_checkpoints": int(len(valid)),
            "valid_geometry": bool(min_detj > VALID_DETJ_EPS) if np.isfinite(min_detj) else None,
        }
    return summary


def _final_rows(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    return df.sort_values("step").groupby(["backend", "seed"], as_index=False).tail(1)


def hypothesis_verdict(summary: dict) -> dict:
    """Apply plan sec. 7 thresholds: PIG vs best of {MLP, PIXEL}."""
    if "pig" not in summary:
        return {"verdict": "incomplete", "reason": "no PIG run found"}
    pig = summary["pig"].get("best_valid_eta_fem_mean", float("nan"))
    others = [
        summary[b].get("best_valid_eta_fem_mean", float("nan"))
        for b in summary
        if b != "pig"
    ]
    others = [v for v in others if np.isfinite(v)]
    if not np.isfinite(pig):
        return {"verdict": "incomplete", "reason": "no valid PIG FEM checkpoint found"}
    if not others:
        return {"verdict": "incomplete", "reason": "no valid baseline FEM checkpoint found"}
    best_other = max(others)
    abs_gain = pig - best_other
    rel_gain = abs_gain / abs(best_other) if best_other else float("nan")
    h1 = abs_gain >= 0.03 or rel_gain >= 0.05
    return {
        "pig_eta_fem": pig,
        "best_baseline_eta_fem": best_other,
        "absolute_gain": abs_gain,
        "relative_gain": rel_gain,
        "H1_supported": bool(h1),
        "verdict": "H1 supported (PIG best)" if h1 else "H0 not rejected",
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "missing"
    try:
        if not np.isfinite(float(value)):
            return "missing"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _plot_status(results_root: Path) -> dict[str, str]:
    figures = results_root / "figures"
    status = {}
    for name in REQUIRED_PLOTS:
        direct = figures / name
        nested = list(figures.glob(f"*/{name}"))
        status[name] = "generated" if direct.exists() or nested else "missing"
    return status


def _conclusion(summary: dict, verdict: dict, finals: pd.DataFrame) -> str:
    if not summary:
        return "Incomplete: no joint backend comparison rows were found."
    ranked = sorted(
        summary.items(),
        key=lambda item: item[1].get("best_valid_eta_fem_mean", -np.inf),
        reverse=True,
    )
    best = ranked[0][0]
    pixel = finals[finals["backend"] == "pixel"]
    pixel_mismatch = None
    if not pixel.empty and {"eta_pinn", "eta_fem"}.issubset(pixel.columns):
        row = pixel.sort_values("step").tail(1).iloc[0]
        pixel_mismatch = abs(float(row["eta_pinn"]) - float(row["eta_fem"]))

    if best == "pig" and verdict.get("H1_supported"):
        return "PIG wins: adaptive locality is supported by the current FEM-validated eta comparison."
    if not np.isfinite(ranked[0][1].get("best_valid_eta_fem_mean", float("nan"))):
        return "Incomplete: no backend has a valid non-folded FEM checkpoint above the detJ safety threshold."
    if pixel_mismatch is not None and pixel_mismatch > 10.0:
        return (
            "PINN eta is not validated by FEM for PIXEL: the internal eta rises far above the "
            "FEM/reference eta, so the optimizer is exploiting physics-backend error in this short run."
        )
    if len(ranked) > 1 and abs(
        ranked[0][1]["best_valid_eta_fem_mean"] - ranked[1][1]["best_valid_eta_fem_mean"]
    ) < 0.02:
        return "All similar: the current FEM eta values are too close to identify the backend as the bottleneck."
    return f"{best.upper()} wins by best valid FEM checkpoint: backend complexity is not yet improving validated eta."


def write_results_summary(metrics_csv: Path, summary: dict, verdict: dict, out: Path) -> None:
    results_root = metrics_csv.parents[1]
    finals = _final_rows(metrics_csv)
    fixed_metrics = results_root / "fixed" / "metrics.csv"
    first_result = next((results_root / "joint").glob("*_seed*/result.json"), None)
    setup = {}
    if first_result is not None:
        with first_result.open() as handle:
            setup = json.load(handle).get("config", {})

    lines: list[str] = []
    lines.append("# SPINN Experiment Results Summary")
    lines.append("")
    lines.append("## Experiment setup")
    exp = setup.get("experiment", {})
    sampling = setup.get("sampling", {})
    optimizer = setup.get("optimizer", {})
    validation = setup.get("validation", {})
    lines.append(f"- Geometry used: {exp.get('geometry', 'missing')}")
    lines.append("- Backends compared: coordinate projection + MLP, coordinate projection + PIXEL, coordinate projection + PIG")
    lines.append(f"- Number of collocation points: interior={sampling.get('interior_points', 'missing')}, electrode boundary per electrode={sampling.get('electrode_boundary_points_per_electrode', 'missing')}, outer boundary={sampling.get('outer_boundary_points', 'missing')}")
    lines.append("- Boundary conditions: +V0/2 on x-axis RF electrodes, -V0/2 on y-axis RF electrodes, 0 on the outer boundary")
    loss_weights = setup.get("loss_weights", {})
    lines.append(f"- Optimizer settings: {optimizer.get('type', 'missing')}, lr_theta={optimizer.get('lr_theta', 'missing')}, lr_phi={optimizer.get('lr_phi', 'missing')}")
    lines.append(f"- Geometry/objective safeguards: jacobian_weight={loss_weights.get('jacobian', 'missing')}, displacement_weight={loss_weights.get('displacement', 'missing')}, quadrupole_weight={loss_weights.get('quadrupole', 'missing')}, valid_checkpoint_min_detJ>{VALID_DETJ_EPS}")
    lines.append(f"- Training steps/epochs represented in saved data: {int(finals['step'].max()) if not finals.empty else 'missing'}")
    lines.append(f"- FEM/reference validation method: finite-difference reference via `fem_eta`, grid={validation.get('fem_grid', 'missing')}")
    lines.append("")
    lines.append("## Main hypothesis")
    lines.append("Better physics representations should give more accurate field derivatives, which should produce better coordinate-projection shape optimization and higher FEM-validated eta.")
    lines.append("")
    lines.append("## Key metrics table")
    lines.append("| Backend | eta_PINN_final | eta_FEM_final | best_valid_eta_FEM | best_valid_step | eta_error | PDE_residual_mean | BC_error_mean | min_detJ | min_gap | max_curvature | runtime_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for backend in BACKEND_ORDER:
        row_df = finals[finals["backend"] == backend]
        backend_summary = summary.get(backend, {})
        if row_df.empty:
            lines.append(f"| {backend.upper()} | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |")
            continue
        row = row_df.sort_values("step").tail(1).iloc[0]
        eta_error = abs(float(row["eta_pinn"]) - float(row["eta_fem"])) if "eta_fem" in row and pd.notna(row["eta_fem"]) else np.nan
        lines.append(
            f"| {backend.upper()} | {_fmt(row.get('eta_pinn'))} | {_fmt(row.get('eta_fem'))} | "
            f"{_fmt(backend_summary.get('best_valid_eta_fem_mean'))} | {_fmt(backend_summary.get('best_valid_step_mean'), 0)} | "
            f"{_fmt(eta_error)} | "
            f"{_fmt(row.get('loss_pde'))} | {_fmt(row.get('loss_bc'))} | {_fmt(row.get('min_detJ'))} | "
            f"{_fmt(row.get('min_gap'))} | {_fmt(row.get('max_curv'))} | {_fmt(row.get('wall_time_sec'), 1)} |"
        )
    lines.append("")
    lines.append("## Plot checklist")
    for name, status in _plot_status(results_root).items():
        lines.append(f"- {name}: {status}")
    lines.append("")
    lines.append("## Warnings and missing data")
    if not fixed_metrics.exists():
        lines.append("- Fixed-geometry benchmark metrics are missing at `spinn/results/fixed/metrics.csv`; fixed-geometry potential/error/residual/eta-error plots were not generated.")
    if finals["seed"].nunique() < 3:
        lines.append(f"- Saved joint comparison has {finals['seed'].nunique()} seed(s), not the planned 3-seed validation.")
    if int(finals["step"].max()) < 1000:
        lines.append("- Saved joint comparison is a short smoke-scale run; do not treat it as the final 20k-60k epoch experiment.")
    lines.append("- PDE_residual_mean and BC_error_mean in the table are logged loss proxies from the final training row, not a full post-hoc statistical residual audit.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(_conclusion(summary, verdict, finals))
    lines.append("")
    lines.append("The conclusion is based on the best valid FEM/reference checkpoint where available, not final PINN-predicted eta.")
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="spinn/results/joint/metrics.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    path = Path(args.results)
    summary = summarize(path)
    verdict = hypothesis_verdict(summary)
    report = {"summary": summary, "hypothesis": verdict}
    text = json.dumps(report, indent=2)
    print(text)
    out = Path(args.out) if args.out else path.parent / "comparison_summary.json"
    out.write_text(text)
    print(f"[compare] wrote {out}")
    summary_path = path.parents[1] / "results_summary.md"
    write_results_summary(path, summary, verdict, summary_path)
    # Keep the older report path in sync for notebooks/slides that already link to it.
    write_results_summary(path, summary, verdict, path.parents[1] / "report.md")
    print(f"[compare] wrote {summary_path}")


if __name__ == "__main__":
    main()
