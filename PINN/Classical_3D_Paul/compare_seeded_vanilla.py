"""Compare the seeded-holes discovery PINN against the vanilla soft-BC baseline.

Both are trained at matched budget (50k steps, batch 1024, warmup->cosine,
best-val checkpoint). The two solve DIFFERENT boundary-value problems -- the
seeded model discovers its own electrode geometry, while the vanilla fits the
fixed Paul electrodes -- so they are NOT comparable on "accuracy vs DEVSIM".
Instead we compare geometry-agnostic quantities:

  1. Convergence: held-out validation PDE residual vs training step.
  2. Trap quality: pseudopotential isotropy (vs the 0.25 harmonic ceiling) and
     locus centering -- the discriminators that matter. Trap depth / secular
     frequencies are reported as annotations but are geometry-dependent and are
     NOT a goal in themselves (a deep trap you can't move/tune is useless --
     controllability is the real figure of merit, handled separately).

Usage:
    python -m PINN.Classical_3D_Paul.compare_seeded_vanilla
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .trap_quality import _build_phi, evaluate


MODELS_DIR = Path(__file__).parent / "models"
FIGURES_DIR = Path(__file__).parent / "figures"

MODELS = {
    "Seeded discovery":  ("pinn_rf_seeded_params.pkl", "tab:blue"),
    "Vanilla (soft BC)": ("pinn_vanilla_50k.pkl",      "black"),
}


def _val_curve(path):
    with open(path, "rb") as f:
        b = pickle.load(f)
    vl = b.get("val_log")
    if vl:
        arr = np.array(vl, dtype=float)
        return arr[:, 0], arr[:, 1]
    return None, None


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = list(MODELS.keys())
    colors = [MODELS[l][1] for l in labels]

    # ---- Panel 1: convergence (held-out validation PDE residual) ----
    ax = axes[0]
    metrics = {}
    for label, (fname, color) in MODELS.items():
        steps, val = _val_curve(MODELS_DIR / fname)
        if steps is not None:
            ax.semilogy(steps, np.maximum(val, 1e-12), color=color, label=label, lw=1.8)
        phi, cfg = _build_phi(MODELS_DIR / fname)
        metrics[label] = evaluate(phi, V_RF=cfg["V_RF"], verbose=False, trap_radius=0.05)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Held-out PDE residual  mean$(\nabla^2\phi)^2$")
    ax.set_title("Convergence (held-out validation)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # ---- Panel 2: isotropy vs the 0.25 harmonic ceiling ----
    ax = axes[1]
    iso = [metrics[l]["isotropy"] for l in labels]
    ax.bar(labels, iso, color=colors, alpha=0.8)
    ax.axhline(0.25, color="red", ls="--", label="harmonic optimum (0.25)")
    ax.set_ylabel(r"Isotropy  $\lambda_{min}/\lambda_{max}$")
    ax.set_title("Pseudopotential isotropy")
    for i, v in enumerate(iso):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.legend()

    # ---- Panel 3: locus centering ----
    ax = axes[2]
    off = [1e4 * np.linalg.norm(metrics[l]["x_min_cm"]) for l in labels]
    ax.bar(labels, off, color=colors, alpha=0.8)
    ax.set_ylabel("Locus offset from center [um]")
    ax.set_title("Trap-center localization")
    for i, v in enumerate(off):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")

    # depth + frequencies as a caption (reported, not a goal)
    txt = []
    for l in labels:
        m = metrics[l]
        fr = ", ".join(f"{x:.2f}" for x in m["secular_freqs_MHz"])
        txt.append(f"{l}: depth {m['trap_depth_eV_at_radius'][1]:.1f} eV, freqs [{fr}] MHz, "
                   f"stable={m['stable']}")
    fig.suptitle("Seeded discovery vs vanilla baseline (50k steps, batch 1024)\n"
                 + "    |    ".join(txt), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGURES_DIR / "compare_seeded_vanilla.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
