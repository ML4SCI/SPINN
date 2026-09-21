#!/usr/bin/env python3
"""Render the plan's result tables (markdown) from the logged CSVs.

Produces:
* a fixed-geometry solver-quality table (eta_pinn, eta_fem, |error|),
* a joint final-result table (final eta_fem, eta mismatch, min detJ, validity),
* the hypothesis verdict line (plan sec. 7).

Usage:
    python -m spinn.scripts.make_report_tables --out spinn/results/report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spinn.eval.compare_backends import hypothesis_verdict, summarize

ROOT = Path(__file__).resolve().parents[1] / "results"


def _fixed_table(path: Path) -> str:
    if not path.exists():
        return "_No fixed-geometry results found._\n"
    df = pd.read_csv(path)
    finals = df[df["eta_fem"].notna()].sort_values("step").groupby(["backend", "seed"]).tail(1)
    agg = finals.groupby("backend").agg(
        eta_pinn=("eta_pinn", "mean"),
        eta_fem=("eta_fem", "mean"),
        eta_abs_error=("eta_abs_error", "mean"),
    )
    lines = ["| backend | eta_pinn | eta_fem | \\|error\\| |", "|---|---|---|---|"]
    for backend, row in agg.iterrows():
        lines.append(f"| {backend} | {row.eta_pinn:.4f} | {row.eta_fem:.4f} | {row.eta_abs_error:.4f} |")
    return "\n".join(lines) + "\n"


def _joint_table(summary: dict) -> str:
    lines = [
        "| backend | best valid eta_fem (mean) | final eta_fem (mean) | eta mismatch | min detJ | valid |",
        "|---|---|---|---|---|---|",
    ]
    for backend, s in summary.items():
        lines.append(
            f"| {backend} | {s['best_valid_eta_fem_mean']:.4f} +/- {s['best_valid_eta_fem_std']:.4f} "
            f"| {s['final_eta_fem_mean']:.4f} +/- {s['final_eta_fem_std']:.4f} "
            f"| {s['eta_mismatch_mean']:.4f} | {s['min_detJ_over_run']:.4f} | {s['valid_geometry']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "report.md"))
    args = parser.parse_args(argv)

    joint_csv = ROOT / "joint" / "metrics.csv"
    summary = summarize(joint_csv) if joint_csv.exists() else {}
    verdict = hypothesis_verdict(summary) if summary else {"verdict": "no joint runs"}

    parts = ["# SPINN Paul-trap shape-optimization report\n"]
    parts.append("## Fixed-geometry solver quality\n")
    parts.append(_fixed_table(ROOT / "fixed" / "metrics.csv"))
    parts.append("\n## Joint shape-optimization result\n")
    parts.append(_joint_table(summary) if summary else "_No joint results found._\n")
    parts.append("\n## Hypothesis verdict (plan sec. 7)\n")
    parts.append("```\n" + "\n".join(f"{k}: {v}" for k, v in verdict.items()) + "\n```\n")

    out = Path(args.out)
    out.write_text("\n".join(parts))
    print(f"[report] wrote {out}")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
