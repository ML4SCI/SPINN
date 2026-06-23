"""CLI for the potential PINN architecture comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import make_config
from .models import MODEL_KINDS
from .plotting import comparison_plots
from .train import train_one


def run_kind(kind: str, args) -> None:
    overrides = list(args.override or [])
    if args.steps is not None:
        overrides.append(f"training.steps={args.steps}")
    if args.n_interior is not None:
        overrides.append(f"training.n_interior={args.n_interior}")
    if args.n_world is not None:
        overrides.append(f"training.n_world={args.n_world}")
    if args.n_electrode is not None:
        overrides.append(f"training.n_electrode={args.n_electrode}")
    config = make_config(kind, overrides)
    output_dir = Path(config.output_root) / kind
    train_one(
        config,
        device=args.device,
        output_dir=output_dir,
        run_devsim_compare=not args.skip_devsim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_KINDS, "all"), default="all")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-interior", type=int, default=None)
    parser.add_argument("--n-world", type=int, default=None)
    parser.add_argument("--n-electrode", type=int, default=None)
    parser.add_argument("--skip-devsim", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    kinds = list(MODEL_KINDS) if args.model == "all" else [args.model]
    for kind in kinds:
        run_kind(kind, args)
    if len(kinds) > 1:
        root = Path(make_config(kinds[0], args.override).output_root)
        comparison_plots(root, kinds)


if __name__ == "__main__":
    main()

