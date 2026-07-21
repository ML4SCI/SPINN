#!/usr/bin/env python3
"""Run a one-factor-at-a-time hyperparameter sweep for the Circular Trap PINN."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PINN_SCRIPT = PROJECT_ROOT / "SPINN" / "PINN" / "Vanilla_Paul_Trap_PINN" / "Paul_Trap_PINN.py"
COMPARE_SCRIPT = PROJECT_ROOT / "SPINN" / "Hyperparameter" / "scripts" / "compare_to_devsim.py"


def deep_update(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def read_json(path):
    with open(path) as f:
        return json.load(f)


def parse_run_dir(stdout):
    for line in reversed(stdout.splitlines()):
        marker = "Saved run to:"
        if marker in line:
            return Path(line.split(marker, 1)[1].strip())
    return None


def run_command(cmd, cwd):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout)
    return proc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run configured PINN hyperparameter experiments.")
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "SPINN" / "Hyperparameter" / "configs" / "sweep_plan.json")
    parser.add_argument("--only", nargs="*", default=None, help="Optional variant_name filter.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected experiments.")
    parser.add_argument("--no_compare", action="store_true", help="Skip DEVSIM comparison after training.")
    parser.add_argument("--max_compare_points", type=int, default=None)
    args = parser.parse_args(argv)

    plan = read_json(args.plan)
    base_config_path = (PROJECT_ROOT / plan["base_config"]).resolve()
    base_config = read_json(base_config_path)
    compare_after_each = bool(plan.get("compare_after_each", True)) and not args.no_compare
    max_compare_points = args.max_compare_points or int(plan.get("max_compare_points", 20000))

    experiments = plan["experiments"]
    if args.only:
        wanted = set(args.only)
        experiments = [exp for exp in experiments if exp["variant_name"] in wanted]
    if args.limit is not None:
        experiments = experiments[:args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = PROJECT_ROOT / "SPINN" / "Hyperparameter" / "results" / f"sweep_{stamp}"
    config_root = sweep_root / "generated_configs"
    log_root = sweep_root / "logs"
    config_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "plan": str(args.plan.resolve()),
        "base_config": str(base_config_path),
        "sweep_root": str(sweep_root),
        "experiments": [],
    }

    for idx, exp in enumerate(experiments, start=1):
        variant_name = exp["variant_name"]
        sweep_name = exp.get("sweep_name", "sweep")
        cfg = deep_update(base_config, exp.get("overrides", {}))
        cfg["sweep_name"] = sweep_name
        cfg["variant_name"] = variant_name
        cfg["run_name"] = f"{sweep_name}_pinn"
        cfg["output_root"] = "SPINN/Hyperparameter/results/runs"

        cfg_path = config_root / f"{idx:03d}_{variant_name}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"\n[{idx}/{len(experiments)}] Training {variant_name}")
        train_cmd = [sys.executable, str(PINN_SCRIPT), "--config", str(cfg_path)]
        train_proc = run_command(train_cmd, PROJECT_ROOT)
        (log_root / f"{idx:03d}_{variant_name}_train.log").write_text(train_proc.stdout)
        run_dir = parse_run_dir(train_proc.stdout)

        record = {
            "sweep_name": sweep_name,
            "variant_name": variant_name,
            "config_path": str(cfg_path),
            "train_returncode": train_proc.returncode,
            "run_dir": str(run_dir) if run_dir else None,
        }

        if train_proc.returncode == 0 and run_dir is not None and compare_after_each:
            print(f"[{idx}/{len(experiments)}] Comparing {variant_name} to DEVSIM")
            compare_cmd = [
                sys.executable,
                str(COMPARE_SCRIPT),
                "--run_dir",
                str(run_dir),
                "--max_points",
                str(max_compare_points),
            ]
            compare_proc = run_command(compare_cmd, PROJECT_ROOT)
            (log_root / f"{idx:03d}_{variant_name}_compare.log").write_text(compare_proc.stdout)
            record["compare_returncode"] = compare_proc.returncode
        elif train_proc.returncode != 0:
            print(f"Training failed for {variant_name}; continuing to next experiment.")

        manifest["experiments"].append(record)
        with open(sweep_root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nSweep manifest: {sweep_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
