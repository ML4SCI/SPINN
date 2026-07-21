#!/usr/bin/env python3
"""Build a minimalist LaTeX/Beamer results deck for the SPINN sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY = PROJECT_ROOT / "SPINN" / "Hyperparameter" / "results" / "summary"


def read_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.exists():
        return []
    with open(summary_csv, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def as_float(row: dict[str, str], key: str, default=float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def number(value, digits=3) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return ""


def tex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def tex_path(path: Path) -> str:
    return path.as_posix().replace("\\", "/")


def best_row(rows: list[dict[str, str]], key: str) -> dict[str, str] | None:
    valid = [r for r in rows if r.get(key) not in {"", "nan", "NaN", None}]
    return min(valid, key=lambda r: as_float(r, key)) if valid else None


def worst_row(rows: list[dict[str, str]], key: str) -> dict[str, str] | None:
    valid = [r for r in rows if r.get(key) not in {"", "nan", "NaN", None}]
    return max(valid, key=lambda r: as_float(r, key)) if valid else None


def row_by_variant(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r.get("variant_name", ""): r for r in rows}


def itemize(items: list[str]) -> str:
    body = "\n".join(rf"  \item {tex_escape(item)}" for item in items if item)
    return "\\begin{itemize}\n" + body + "\n\\end{itemize}"


def frame(title: str, body: str) -> str:
    return "\n".join([
        rf"\begin{{frame}}[t]{{{tex_escape(title)}}}",
        body,
        r"\end{frame}",
        "",
    ])


def two_column_frame(title: str, left: str, right: str, left_width="0.58", right_width="0.38") -> str:
    return frame(
        title,
        "\n".join([
            r"\begin{columns}[T,onlytextwidth]",
            rf"\begin{{column}}{{{left_width}\textwidth}}",
            left,
            r"\end{column}",
            rf"\begin{{column}}{{{right_width}\textwidth}}",
            right,
            r"\end{column}",
            r"\end{columns}",
        ]),
    )


def graphic(path: Path, width=r"\linewidth") -> str:
    return rf"\includegraphics[width={width}]{{{tex_path(path)}}}"


def overview_items(rows: list[dict[str, str]]) -> list[str]:
    by_name = row_by_variant(rows)
    best = best_row(rows, "phi_mse")
    baseline = by_name.get("baseline")
    relu = by_name.get("activation_relu")
    rmsprop = by_name.get("optimizer_rmsprop")
    items = []
    if best:
        items.append(
            f"Best DEVSIM agreement came from {best['variant_name']}: "
            f"Phi MSE {number(best['phi_mse'])}, Psi MSE {number(best['psi_mse'])}."
        )
    if best and baseline:
        ratio = as_float(baseline, "phi_mse") / as_float(best, "phi_mse")
        items.append(f"That is about {number(ratio)}x lower RF-potential MSE than the baseline.")
    if relu:
        items.append("ReLU is the cautionary case: tiny training loss but the worst DEVSIM field error.")
    if rmsprop:
        items.append(f"Among optimizer-only variants, RMSprop was best: Phi MSE {number(rmsprop['phi_mse'])}.")
    return items


def graph_analysis(rows: list[dict[str, str]], key: str) -> list[str]:
    by_name = row_by_variant(rows)
    if key == "convergence":
        relu = by_name.get("activation_relu")
        bc1000 = by_name.get("bc_weight_1000")
        return [
            "Normalizing by initial loss makes runs with different starting scales comparable.",
            "Fast normalized convergence does not guarantee physical accuracy.",
            f"ReLU ended at normalized loss {number(relu['final_normalized_loss'])}, but failed the DEVSIM comparison." if relu else "",
            f"BC weight 1000 had both low normalized loss and low field error: Phi MSE {number(bc1000['phi_mse'])}." if bc1000 else "",
        ]
    if key == "phi":
        best = best_row(rows, "phi_mse")
        worst = worst_row(rows, "phi_mse")
        return [
            f"Lowest RF-potential error: {best['variant_name']} at Phi MSE {number(best['phi_mse'])}." if best else "",
            f"Largest RF-potential error: {worst['variant_name']} at Phi MSE {number(worst['phi_mse'])}." if worst else "",
            "Boundary-condition weighting was the strongest lever in this sweep.",
            "Wider networks and sinusoidal activation were strong architecture-side options.",
        ]
    if key == "psi":
        best = best_row(rows, "psi_mse")
        return [
            f"Lowest pseudopotential error: {best['variant_name']} at Psi MSE {number(best['psi_mse'])}." if best else "",
            "The pseudopotential depends on gradients, so it exposes field-shape errors.",
            "The best Phi runs generally also improved Psi, but noisy PDE spikes can still damage gradients.",
        ]
    if key == "loss":
        return [
            "Final training loss alone is not a reliable model-selection metric.",
            "ReLU is artificially favored because its second derivative is zero almost everywhere.",
            "BC weight 1000 is the most credible winner because it combines low loss with low DEVSIM error.",
        ]
    if key == "loss_vs_phi":
        return [
            "Ideal runs appear near the lower-left: low training loss and low DEVSIM error.",
            "ReLU is isolated as a false low-loss solution.",
            "BC weight 1000 is the cleanest point because it pairs low loss with the lowest field error.",
        ]
    return []


def sweep_takeaway(sweep_name: str, rows: list[dict[str, str]]) -> list[str]:
    ranked = sorted(rows, key=lambda r: as_float(r, "phi_mse"))
    best = ranked[0]
    worst = ranked[-1]
    if sweep_name == "baseline":
        return [
            f"Baseline Phi MSE is {number(best['phi_mse'])}; baseline Psi MSE is {number(best['psi_mse'])}.",
            "Several variants beat this reference decisively, so the PINN is hyperparameter-sensitive.",
        ]
    if sweep_name == "width":
        return [
            f"Best width run: {best['variant_name']} with Phi MSE {number(best['phi_mse'])}.",
            f"Worst width run: {worst['variant_name']} with Phi MSE {number(worst['phi_mse'])}.",
            "Width improved physical field accuracy more than final loss alone suggested.",
        ]
    if sweep_name == "depth":
        return [
            f"Best depth run: {best['variant_name']} with Phi MSE {number(best['phi_mse'])}.",
            "Depth variants were unstable; increasing depth did not improve DEVSIM agreement.",
        ]
    if sweep_name == "activation":
        return [
            "Sinusoidal activation was the best activation-only variant.",
            "ReLU produced misleadingly tiny loss because the Laplace residual degenerates under second derivatives.",
        ]
    if sweep_name == "bc_weight":
        return [
            f"Best boundary-weight run: {best['variant_name']} with Phi MSE {number(best['phi_mse'])}.",
            "Too little boundary weight left boundary mismatch dominant; 1000 gave the cleanest DEVSIM match.",
        ]
    if sweep_name == "optimizer":
        return [
            f"Best optimizer-only run: {best['variant_name']} with Phi MSE {number(best['phi_mse'])}.",
            "SGD plateaued with poor boundary accuracy; RMSprop was the strongest optimizer-only result.",
        ]
    if sweep_name == "learning_rate":
        return [
            f"Best learning-rate run: {best['variant_name']} with Phi MSE {number(best['phi_mse'])}.",
            "Lower learning rates were smoother but too slow for the fixed training budget.",
        ]
    return [f"Best run: {best['variant_name']}."]


def table_rows(rows: list[dict[str, str]], limit: int | None = None) -> str:
    ranked = sorted(rows, key=lambda r: as_float(r, "phi_mse"))
    if limit:
        ranked = ranked[:limit]
    lines = [
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & Final norm. loss & AUC norm. loss & Phi MSE & Psi MSE \\",
        r"\midrule",
    ]
    for row in ranked:
        lines.append(
            " & ".join([
                tex_escape(row.get("variant_name", "")),
                number(row.get("final_normalized_loss", "")),
                number(row.get("auc_normalized_loss", "")),
                number(row.get("phi_mse", "")),
                number(row.get("psi_mse", "")),
            ]) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def threshold_step(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if value in {"", "nan", "NaN", None}:
        return r"$>$19.9k"
    return number(value, digits=4)


def final_metrics_table(rows: list[dict[str, str]]) -> str:
    sweep_order = {
        "baseline": 0,
        "width": 1,
        "depth": 2,
        "activation": 3,
        "bc_weight": 4,
        "optimizer": 5,
        "learning_rate": 6,
    }
    ordered = sorted(
        rows,
        key=lambda r: (
            sweep_order.get(r.get("sweep_name", ""), 99),
            r.get("variant_name", ""),
        ),
    )
    midpoint = (len(ordered) + 1) // 2

    def mini_table(table_rows_subset: list[dict[str, str]]) -> str:
        lines = [
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Variant & $t_{.1}$ & $t_{.01}$ & $\widehat{\mathcal{L}}_f$ & $\mathrm{MSE}_{\phi}$ & $\mathrm{MSE}_{\Psi}$ \\",
            r"\midrule",
        ]
        for row in table_rows_subset:
            lines.append(
                " & ".join([
                    tex_escape(row.get("variant_name", "")),
                    threshold_step(row, "step_norm_loss_le_1e-1"),
                    threshold_step(row, "step_norm_loss_le_1e-2"),
                    number(row.get("final_normalized_loss", "")),
                    number(row.get("phi_mse", "")),
                    number(row.get("psi_mse", "")),
                ]) + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
        return "\n".join(lines)

    lines = [
        r"\begingroup",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\renewcommand{\arraystretch}{0.74}",
        r"\begin{columns}[T,onlytextwidth]",
        r"\begin{column}{0.49\textwidth}",
        mini_table(ordered[:midpoint]),
        r"\end{column}",
        r"\begin{column}{0.49\textwidth}",
        mini_table(ordered[midpoint:]),
        r"\end{column}",
        r"\end{columns}",
        r"\vspace{0.015in}",
        r"{\tiny $t_{.1}$ and $t_{.01}$ are the first logged steps where normalized loss crossed 10$^{-1}$ and 10$^{-2}$; $>$19.9k means not reached. Raw PINN--DEVSIM error maps for each row are shown in the preceding appendix slides.}",
        r"\endgroup",
    ]
    return "\n".join(lines)


def final_discussion_summary(rows: list[dict[str, str]]) -> str:
    by_name = row_by_variant(rows)
    relu = by_name.get("activation_relu")
    rmsprop = by_name.get("optimizer_rmsprop")
    bc1000 = by_name.get("bc_weight_1000")
    width256 = by_name.get("width_256")
    sinusoidal = by_name.get("activation_sinusoidal")

    paragraph = (
        "Overall, normalized loss convergence and physical agreement with DEVSIM are related but not interchangeable. "
    )
    if relu:
        paragraph += (
            f"ReLU converged fastest by the normalized-loss metric, reaching 10^-3 by step "
            f"{number(relu.get('step_norm_loss_le_1e-3'), digits=4)} with final normalized loss "
            f"{number(relu.get('final_normalized_loss'))}, but it produced the worst raw field mismatch "
            f"(Phi MSE {number(relu.get('phi_mse'))}, Psi MSE {number(relu.get('psi_mse'))}); "
            "this makes it a negative control rather than a physically credible winner for a second-derivative PINN. "
        )
    if rmsprop and bc1000:
        paragraph += (
            f"Among credible runs, RMSprop reached normalized loss 10^-2 first at step "
            f"{number(rmsprop.get('step_norm_loss_le_1e-2'), digits=4)}, while BC weight 1000 reached 10^-3 by step "
            f"{number(bc1000.get('step_norm_loss_le_1e-3'), digits=4)} and gave the best total PINN--ground-truth MSE "
            f"(Phi MSE {number(bc1000.get('phi_mse'))}, Psi MSE {number(bc1000.get('psi_mse'))}). "
        )
    if width256 and sinusoidal:
        paragraph += (
            f"The next-best architecture-side cases were width 256 "
            f"(Phi MSE {number(width256.get('phi_mse'))}, Psi MSE {number(width256.get('psi_mse'))}) "
            f"and sinusoidal activation (Phi MSE {number(sinusoidal.get('phi_mse'))}, Psi MSE {number(sinusoidal.get('psi_mse'))}), "
            "which also had cleaner raw PINN--DEVSIM residual maps than the baseline. "
        )
    paragraph += (
        "Depth increases, sigmoid, SGD, low boundary-condition weight, and the slowest learning-rate settings either failed to pass useful normalized-loss thresholds within the fixed training budget or retained broad residual structure in the raw error plots."
    )

    return "\n".join([
        r"\scriptsize",
        r"\setlength{\parindent}{0pt}",
        tex_escape(paragraph).replace("10\\textasciicircum{}-3", r"$10^{-3}$").replace("10\\textasciicircum{}-2", r"$10^{-2}$"),
        r"\vspace{0.07in}",
        final_metrics_table(rows),
    ])


def variant_takeaway(row: dict[str, str]) -> str:
    name = row.get("variant_name", "")
    if name == "activation_relu":
        return "Misleading loss case: training loss collapses, but field error is the worst in the sweep."
    if name == "bc_weight_1000":
        return "Best overall DEVSIM match; strong boundary enforcement did not destroy field accuracy."
    if name == "width_256":
        return "Best width result; wider representation improved physical field shape."
    if name == "activation_sinusoidal":
        return "Best activation-only result; smooth derivatives help the PDE residual and gradient-derived Psi."
    if name == "optimizer_sgd":
        return "SGD plateaued; comparison confirms poor field accuracy for this training budget."
    if name == "baseline":
        return "Reference run for all hyperparameter comparisons."
    return (
        f"Final normalized loss {number(row.get('final_normalized_loss'))}; "
        f"Phi MSE {number(row.get('phi_mse'))}; Psi MSE {number(row.get('psi_mse'))}."
    )


def audit_frames(audit: dict, summary_dir: Path) -> list[str]:
    if not audit:
        return []
    fixed = audit["best_fixed_geometry"]
    gate = fixed["single_seed_derivative_gate"]
    gradient_rows = audit.get("gradient_convergence", [])
    validity = audit["coordinate_map_validity"]
    fit_range = 100.0 * audit["fit_radius_efficiency_relative_range"]

    metric_table = "\n".join([
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Quantity & Audited error & Proposed gate \\",
        r"\midrule",
        rf"Potential MAE / 300 V & {gate['potential_mae_percent_of_voltage']:.3f}\% & $<0.5\%$ \\",
        rf"Electric-field relative $L^2$ & {100.0 * fixed['electric_field_V_per_cm']['relative_l2']:.2f}\% & $<5\%$ \\",
        rf"Center Hessian relative Frobenius & {100.0 * fixed['center_hessian_relative_frobenius_error']:.2f}\% & $<5\%$ \\",
        rf"Quadrupole-efficiency relative error & {100.0 * fixed['center_efficiency_relative_error']:.2f}\% & $<5\%$ \\",
        rf"Secular-frequency relative $L^2$ & {100.0 * fixed['secular_frequency_relative_l2']:.2f}\% & $<5\%$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.12in}",
        r"\begin{itemize}",
        rf"\item Single-seed derivative gate: \textbf{{{'PASS' if gate['passed'] else 'FAIL'}}}.",
        r"\item This certifies a fixed-geometry checkpoint, not robustness across seeds or geometries.",
        r"\item The earlier width-64 diagnostic was not the best model and should not represent the final baseline.",
        r"\end{itemize}",
    ])

    formulation = "\n".join([
        r"The mentor-specified shape network follows Equation 13 of Zhang et al.:",
        r"\[",
        r"NN_\phi(\mathbf{z})=\mathbf{z}+\widehat{NN}_\phi(\mathbf{z}),",
        r"\qquad",
        r"\widehat{NN}_\phi(\mathbf{z})\big|_{\mathrm{init}}=\mathbf{0}.",
        r"\]",
        r"The physical-field network is evaluated on the projected coordinates:",
        r"\[",
        r"\mathbf{x}=NN_\phi(\mathbf{z}),",
        r"\qquad",
        r"\phi_\theta(\mathbf{x})=NN_\theta(NN_\phi(\mathbf{z})).",
        r"\]",
        r"\begin{itemize}",
        r"\item Both networks use six hidden layers of width 50 with $\tanh$ in the reference method.",
        r"\item The residual output layer is zero-initialized, making the complete shape map exactly identity initially.",
        r"\item Positive-Jacobian and fixed-region penalties preserve topology and anchor non-design regions.",
        r"\item Shape and field parameters are ultimately trained jointly against physics, constraints, and the trap objective.",
        r"\end{itemize}",
    ])

    transformed_physics = "\n".join([
        r"For the constrained validation map $\mathbf{x}=F_d(\mathbf{z})$, the equivalent pulled-back PDE is",
        r"\[",
        r"\Delta_{\mathbf{x}}\phi =",
        r"\frac{1}{\det J}\nabla_{\mathbf{z}}\!\cdot",
        r"\left(\det J\,J^{-1}J^{-T}\nabla_{\mathbf{z}}u_\theta\right)=0.",
        r"\]",
        r"\begin{itemize}",
        r"\item Stage 1 uses an analytic electrode-translation map to validate geometry derivatives against DEVSIM.",
        r"\item Stage 2 replaces that analytic map with the trainable residual network from Equation 13.",
        r"\item The analytic map is a controlled benchmark, not the intended final shape architecture.",
        r"\item Ordinary $\Delta_{\mathbf{z}}u_\theta=0$ would still solve the wrong transformed problem.",
        r"\end{itemize}",
    ])

    gradient_lines = [
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"DEVSIM mesh & 3-point derivative & 5-point derivative & stencil spread \\",
        r"\midrule",
    ]
    for row in gradient_rows:
        three = row.get("three_point_derivative_per_cm")
        five = row.get("five_point_derivative_per_cm")
        spread = row.get("stencil_relative_spread")
        gradient_lines.append(
            f"{tex_escape(row['label'])} & "
            f"{three:.4f} & "
            f"{five:.4f}" if five is not None else f"{tex_escape(row['label'])} & {three:.4f} & --"
        )
        gradient_lines[-1] += (
            f" & {100.0 * spread:.2f}\\% \\\\" if spread is not None else r" & -- \\"
        )
    gradient_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.12in}",
        r"\begin{itemize}",
        r"\item Fine-mesh three- and five-point derivatives agree within 0.39\%.",
        r"\item Fine vs. medium discretization changes the derivative by about 6\%; this is the present reference uncertainty.",
        rf"\item Local DEVSIM efficiency changes by only {fit_range:.4f}\% across fit radii 0.015--0.06 cm.",
        r"\item A PINN shape derivative is accepted only after agreement with this independent reference.",
        r"\end{itemize}",
    ])

    residual_plot = summary_dir / "residual_coordinate_projection_diagnostic.png"
    residual_body = "\n".join([
        r"\begin{columns}[T,onlytextwidth]",
        r"\begin{column}{0.62\textwidth}",
        graphic(residual_plot),
        r"\end{column}",
        r"\begin{column}{0.35\textwidth}",
        r"\begin{itemize}",
        r"\item Diagnostic of the Equation-13 residual map $NN_\phi(\mathbf{z})=\mathbf{z}+\widehat{NN}(\mathbf{z})$ on the $[-0.35,0.35]^2$ scan.",
        r"\item Shown at identity initialization: displacement $\approx 0$, $\det J\approx 1$, condition number $\approx 1$.",
        r"\item This is an architecture check for the freeform Stage 2 network, not a trained geometry.",
        r"\item No optimized freeform geometry is claimed; a non-trivial map only appears after joint training (or an explicitly labeled synthetic demo).",
        r"\end{itemize}",
        r"\end{column}",
        r"\end{columns}",
    ])

    min_det = min(row["minimum_determinant"] for row in validity)
    max_condition = max(row["maximum_condition"] for row in validity)
    audit_plot = summary_dir / "coordinate_projection_audit.png"
    audit_body = "\n".join([
        r"\begin{columns}[T,onlytextwidth]",
        r"\begin{column}{0.62\textwidth}",
        graphic(audit_plot),
        r"\end{column}",
        r"\begin{column}{0.35\textwidth}",
        r"\begin{itemize}",
        rf"\item Dense map scan: minimum $\det J={min_det:.3f}$.",
        rf"\item Maximum Jacobian condition number: {max_condition:.2f}.",
        r"\item Four manufactured coordinate-map tests recover harmonic solutions.",
        r"\item The full coordinate-projection model has only been smoke-tested; no optimized geometry is claimed.",
        r"\end{itemize}",
        r"\end{column}",
        r"\end{columns}",
    ])

    status_body = "\n".join([
        r"\begin{columns}[T,onlytextwidth]",
        r"\begin{column}{0.48\textwidth}",
        r"\textbf{Validated now}",
        r"\begin{itemize}",
        r"\item Best fixed-geometry checkpoint passes derivative-aware gates.",
        r"\item Pullback operator passes affine and radial manufactured tests.",
        r"\item Coordinate map is injective over the tested one-parameter interval.",
        r"\item Independent DEVSIM shape-gradient workflow is operational.",
        r"\end{itemize}",
        r"\end{column}",
        r"\begin{column}{0.48\textwidth}",
        r"\textbf{Not yet a scientific result}",
        r"\begin{itemize}",
        r"\item Three-seed confirmation of the best fixed model.",
        r"\item Converged training of $u_\theta(\mathbf{z},d)$.",
        r"\item PINN--DEVSIM shape-gradient agreement after convergence.",
        r"\item A physical 3D escape depth or a freeform optimized electrode.",
        r"\end{itemize}",
        r"\end{column}",
        r"\end{columns}",
    ])

    return [
        frame("Derivative-Aware Audit of the Best Fixed Model", metric_table),
        frame("Mentor-Specified Residual Projection", formulation),
        frame("Residual Projection Architecture Diagnostic", residual_body),
        frame("Constrained Map vs. Freeform Network", transformed_physics),
        frame("Coordinate Map and Independent Solver Audit", audit_body),
        frame("DEVSIM Shape-Gradient Convergence", "\n".join(gradient_lines)),
        frame("What Is a Result Today?", status_body),
    ]


def latex_document(rows: list[dict[str, str]], summary_dir: Path, audit: dict | None = None) -> str:
    by_sweep = OrderedDict()
    for row in rows:
        by_sweep.setdefault(row.get("sweep_name", "unknown"), []).append(row)

    best = best_row(rows, "phi_mse")

    parts = [
        r"\documentclass[aspectratio=169,professionalfonts]{beamer}",
        r"\usepackage{fontspec}",
        r"\usepackage{amsmath}",
        r"\usepackage{unicode-math}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\setmainfont{Times New Roman}",
        r"\setsansfont{Times New Roman}",
        r"\setmonofont{Courier New}",
        r"\setmathfont{STIX Two Math}",
        r"\usetheme{default}",
        r"\usefonttheme{professionalfonts}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamertemplate{footline}[frame number]",
        r"\setbeamertemplate{itemize items}[circle]",
        r"\setbeamercolor{normal text}{fg=black,bg=white}",
        r"\setbeamercolor{structure}{fg=black,bg=white}",
        r"\setbeamercolor{frametitle}{fg=black,bg=white}",
        r"\setbeamercolor{title}{fg=black,bg=white}",
        r"\setbeamercolor{item}{fg=black}",
        r"\setbeamerfont{frametitle}{family=\rmfamily,size=\Large}",
        r"\setbeamerfont{title}{family=\rmfamily,size=\LARGE}",
        r"\setbeamerfont{normal text}{family=\rmfamily,size=\small}",
        r"\setbeamerfont{itemize/enumerate body}{size=\small}",
        r"\setbeamerfont{itemize/enumerate subbody}{size=\footnotesize}",
        r"\hypersetup{colorlinks=false}",
        r"\title{SPINN Paul-Trap PINNs}",
        r"\subtitle{Hyperparameters, derivative validation, and coordinate projection}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        "",
        r"\begin{frame}[plain]",
        r"\titlepage",
        r"\vspace{-0.35in}",
        itemize([
            f"Runs completed: {len(rows)}.",
            f"Best DEVSIM match: {best['variant_name']} with Phi MSE {number(best['phi_mse'])}." if best else "",
            "Best fixed model passes the single-seed derivative-aware validation gate." if audit else "",
            "Coordinate projection is implemented and operator-tested; converged geometry optimization is still pending." if audit else "",
        ]),
        r"\end{frame}",
        "",
    ]

    parts.append(frame("Study Setup", itemize([
        "Ground truth: Circular Trap DEVSIM RF solve with East/West electrodes at 300 V.",
        "PINN target: solve the RF potential on matching geometry.",
        "Comparison: evaluate the trained PINN at DEVSIM mesh nodes and compute common-point error metrics.",
        "Sweeps: width, depth, activation, boundary-condition weight, optimizer, and learning rate.",
    ])))

    parts.append(frame("Governing Equation and PINN Objective", "\n".join([
        r"The RF potential is modeled as a Laplace problem:",
        r"\[",
        r"\nabla^2 \phi_{\mathrm{RF}}(x,y) = 0,",
        r"\]",
        r"with electrode boundary conditions enforced through a weighted boundary loss:",
        r"\[",
        r"\mathcal{L} = \mathcal{L}_{\mathrm{PDE}} + \lambda_{\mathrm{BC}}\mathcal{L}_{\mathrm{BC}},",
        r"\]",
        r"\[",
        r"\mathcal{L}_{\mathrm{PDE}} = \frac{1}{N_c}\sum_i \left|\nabla^2 \phi_\theta(x_i,y_i)\right|^2,",
        r"\qquad",
        r"\mathcal{L}_{\mathrm{BC}} = \frac{1}{N_b}\sum_j \left|\phi_\theta(x_j,y_j)-\phi_j\right|^2.",
        r"\]",
    ])))

    parts.append(frame("Evaluation Metrics", "\n".join([
        r"Training curves were compared using normalized loss:",
        r"\[",
        r"\widehat{\mathcal{L}}(t) = \frac{\mathcal{L}(t)}{\mathcal{L}(0)}.",
        r"\]",
        r"Ground-truth agreement was evaluated with DEVSIM common-node mean-square error:",
        r"\[",
        r"\mathrm{MSE}_{\phi} = \frac{1}{N}\sum_i \left(\phi_{\theta,i}-\phi_{\mathrm{DEVSIM},i}\right)^2.",
        r"\]",
        r"The RF pseudopotential was computed from the RF-field gradient:",
        r"\[",
        r"\Psi_{\mathrm{RF}} = \frac{q^2}{4m\Omega^2}\left|\nabla \phi_{\mathrm{RF}}\right|^2,",
        r"\qquad",
        r"\mathrm{MSE}_{\Psi} = \frac{1}{N}\sum_i \left(\Psi_{\theta,i}-\Psi_{\mathrm{DEVSIM},i}\right)^2.",
        r"\]",
    ])))

    parts.append(two_column_frame(
        "Main Results",
        table_rows(rows, limit=8),
        itemize(overview_items(rows)),
        left_width="0.62",
        right_width="0.35",
    ))

    parts.extend(audit_frames(audit or {}, summary_dir))

    graph_slides = [
        ("Normalized Convergence", summary_dir / "normalized_loss_curves.png", "convergence"),
        ("RF Potential Error", summary_dir / "phi_mse_by_variant.png", "phi"),
        ("RF Pseudopotential Error", summary_dir / "psi_mse_by_variant.png", "psi"),
        ("Final Normalized Loss", summary_dir / "final_normalized_loss_by_variant.png", "loss"),
        ("Training Loss vs. Field Error", summary_dir / "loss_vs_phi_mse.png", "loss_vs_phi"),
    ]
    for title, path, key in graph_slides:
        parts.append(two_column_frame(
            title,
            graphic(path, width=r"\linewidth"),
            itemize(graph_analysis(rows, key)),
            left_width="0.64",
            right_width="0.33",
        ))

    for sweep_name, sweep_rows in by_sweep.items():
        display_name = "BC Weight" if sweep_name == "bc_weight" else sweep_name.replace("_", " ").title()
        parts.append(two_column_frame(
            f"{display_name} Sweep",
            table_rows(sweep_rows),
            itemize(sweep_takeaway(sweep_name, sweep_rows)),
            left_width="0.62",
            right_width="0.35",
        ))

    parts.append(frame("Discussion Points", itemize([
        "Repeat BC weight 1000, width 256, and sinusoidal activation across at least three seeds.",
        "Should ReLU be excluded for second-derivative PINNs, or kept only as a negative control?",
        "Train the parametric coordinate-projection model until its DEVSIM shape derivative passes the reference gate.",
        "After the one-parameter proof, compare constrained freeform maps with PIXEL, PIG, and harmonic-by-construction backends.",
    ])))

    parts.append(frame("Error Map Appendix", itemize([
        "Each following slide shows raw PINN minus DEVSIM error maps.",
        "Left: RF potential error.",
        "Right: RF pseudopotential error.",
        "Text remains black on a transparent/plain slide background; plots are included as generated image data.",
    ])))

    ordered = sorted(rows, key=lambda r: (r.get("sweep_name", ""), r.get("variant_name", "")))
    for row in ordered:
        run_dir = Path(row.get("run_dir", ""))
        phi_plot = Path(os.path.relpath(run_dir / "comparison_phi_rf.png", summary_dir))
        psi_plot = Path(os.path.relpath(run_dir / "comparison_psi_rf_eV.png", summary_dir))
        if not (summary_dir / phi_plot).exists() and not (summary_dir / psi_plot).exists():
            continue
        body = "\n".join([
            r"\begin{columns}[T,onlytextwidth]",
            r"\begin{column}{0.49\textwidth}",
            graphic(phi_plot, width=r"\linewidth") if (summary_dir / phi_plot).exists() else "",
            r"\end{column}",
            r"\begin{column}{0.49\textwidth}",
            graphic(psi_plot, width=r"\linewidth") if (summary_dir / psi_plot).exists() else "",
            r"\end{column}",
            r"\end{columns}",
            r"\vspace{0.08in}",
            rf"\small {tex_escape(variant_takeaway(row))}",
        ])
        parts.append(frame(row.get("variant_name", "variant"), body))

    parts.append(frame("Discussion Summary", final_discussion_summary(rows)))

    parts.append(r"\end{document}")
    return "\n".join(parts)


def compile_pdf(tex_path: Path) -> None:
    xelatex = shutil.which("xelatex")
    if not xelatex:
        print("xelatex not found; wrote .tex only.")
        return
    for _ in range(2):
        subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a LaTeX/Beamer SPINN hyperparameter results deck.")
    parser.add_argument("--summary_dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUMMARY / "spinn_hyperparameter_results_latex.tex")
    parser.add_argument("--compile", action="store_true", help="Compile the deck with xelatex.")
    args = parser.parse_args(argv)

    rows = read_rows(args.summary_dir / "summary.csv")
    audit = read_json(args.summary_dir / "experiment_audit.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex_document(rows, args.summary_dir, audit))
    print(f"Saved LaTeX deck to {args.out}")
    if args.compile:
        compile_pdf(args.out)
        pdf_path = args.out.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"Saved PDF deck to {pdf_path}")


if __name__ == "__main__":
    main()
