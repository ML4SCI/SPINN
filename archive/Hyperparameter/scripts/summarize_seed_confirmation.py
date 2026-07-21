#!/usr/bin/env python3
"""Aggregate a seed-confirmation manifest into mean/std benchmark tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "potential_relative_l2": ("potential_V", "relative_l2"),
    "field_relative_l2": ("electric_field_V_per_cm", "relative_l2"),
    "efficiency_relative_error": ("center_efficiency_relative_error",),
    "frequency_relative_l2": ("secular_frequency_relative_l2",),
    "escape_barrier_eV": ("pinn_escape", "barrier_eV"),
}


def nested_value(record: dict, path: tuple[str, ...]) -> float:
    value = record
    for key in path:
        value = value[key]
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    grouped: dict[str, list[dict]] = defaultdict(list)
    per_run = []
    for experiment in manifest:
        run_dir_value = experiment.get("run_dir")
        if not run_dir_value:
            continue
        benchmark_path = Path(run_dir_value) / "fixed_geometry_benchmark.json"
        if not benchmark_path.exists():
            continue
        benchmark = json.loads(benchmark_path.read_text())
        row = {
            "variant": experiment["variant"],
            "seed": experiment["seed"],
            "run_dir": run_dir_value,
        }
        for name, path in METRICS.items():
            row[name] = nested_value(benchmark, path)
        per_run.append(row)
        grouped[experiment["variant"]].append(row)

    summary = []
    for variant, rows in sorted(grouped.items()):
        output = {"variant": variant, "n_seeds": len(rows)}
        for name in METRICS:
            values = np.asarray([row[name] for row in rows])
            output[f"{name}_mean"] = float(np.mean(values))
            output[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(output)

    output_dir = args.manifest.parent
    for filename, rows in [
        ("seed_benchmark_runs.csv", per_run),
        ("seed_benchmark_summary.csv", summary),
    ]:
        if rows:
            with (output_dir / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    (output_dir / "seed_benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
