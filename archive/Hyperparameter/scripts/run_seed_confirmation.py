#!/usr/bin/env python3
"""Run three-seed confirmation for the strongest existing 2D configurations."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SPINN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SPINN_ROOT.parent
TRAIN_SCRIPT = SPINN_ROOT / "PINN" / "Vanilla_Paul_Trap_PINN" / "Paul_Trap_PINN.py"
BENCHMARK_SCRIPT = Path(__file__).with_name("fixed_geometry_benchmark.py")
BASE_CONFIG = SPINN_ROOT / "Hyperparameter" / "configs" / "baseline.json"
VARIANTS = {
    "bc_weight_1000": {"training": {"bc_weight": 1000.0}},
    "width_256": {"model": {"width": 256}},
    "activation_sinusoidal": {"model": {"activation": "sinusoidal"}},
}


def deep_update(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def parse_run_dir(output: str) -> Path:
    marker = "Saved run to:"
    for line in reversed(output.splitlines()):
        if marker in line:
            return Path(line.split(marker, 1)[1].strip())
    raise RuntimeError("training completed without reporting its run directory")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args(argv)

    base = json.loads(BASE_CONFIG.read_text())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = SPINN_ROOT / "Hyperparameter" / "results" / f"seed_confirmation_{stamp}"
    config_dir = root / "configs"
    log_dir = root / "logs"
    config_dir.mkdir(parents=True)
    log_dir.mkdir()
    manifest = []

    for variant in args.variants:
        for seed in args.seeds:
            config = deep_update(base, VARIANTS[variant])
            config["seed"] = seed
            config["run_name"] = f"confirmation_{variant}"
            config["variant_name"] = f"{variant}_seed_{seed}"
            config["sweep_name"] = "seed_confirmation"
            if args.output_root is not None:
                config["output_root"] = str(args.output_root.resolve())
            if args.steps is not None:
                config["training"]["steps"] = args.steps
            if args.smoke:
                config["training"].update({
                    "steps": 3,
                    "n_int": 64,
                    "n_world": 64,
                    "n_electrode": 32,
                    "print_every": 1,
                    "use_lr_scheduler": False,
                })
                config["evaluation"] = {"nx": 40, "ny": 32}
            config_path = config_dir / f"{variant}_seed_{seed}.json"
            config_path.write_text(json.dumps(config, indent=2))

            process = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            (log_dir / f"{variant}_seed_{seed}_train.log").write_text(process.stdout)
            record = {
                "variant": variant,
                "seed": seed,
                "train_returncode": process.returncode,
                "config": str(config_path),
            }
            if process.returncode == 0:
                run_dir = parse_run_dir(process.stdout)
                record["run_dir"] = str(run_dir)
                if not args.skip_benchmark:
                    benchmark = subprocess.run(
                        [
                            sys.executable,
                            str(BENCHMARK_SCRIPT),
                            "--run-dir",
                            str(run_dir),
                        ],
                        cwd=SPINN_ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    (log_dir / f"{variant}_seed_{seed}_benchmark.log").write_text(
                        benchmark.stdout
                    )
                    record["benchmark_returncode"] = benchmark.returncode
            manifest.append(record)
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(
                f"{variant} seed={seed}: train={record['train_returncode']} "
                f"benchmark={record.get('benchmark_returncode', 'skipped')}"
            )
    print(f"Saved seed-confirmation manifest to {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
