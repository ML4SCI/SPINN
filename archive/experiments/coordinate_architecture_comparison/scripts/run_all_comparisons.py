"""Train all three architectures and write the cross-architecture comparison.

Each model is trained through the shared pipeline, then the convergence plot,
MSE bar chart, and metrics CSV are written to ``results/comparison``.

    python -m experiments.coordinate_architecture_comparison.scripts.run_all_comparisons
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from experiments.coordinate_architecture_comparison.plotting.plot_comparison import (
    plot_convergence_comparison,
    plot_mse_comparison,
    write_metrics_summary,
)
from experiments.coordinate_architecture_comparison.training.train_one_model import (
    RESULTS_DIR,
    run_model,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def main() -> None:
    with open(CONFIG_DIR / "comparison.yaml") as handle:
        kinds = (yaml.safe_load(handle) or {}).get("models", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [run_model(kind, device=device) for kind in kinds]

    out_dir = RESULTS_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence_comparison(out_dir / "convergence_comparison.png", results)
    plot_mse_comparison(out_dir / "mse_comparison.png", results)
    write_metrics_summary(out_dir / "metrics_summary.csv", results)
    print(f"saved comparison to {out_dir}")


if __name__ == "__main__":
    main()
