#!/usr/bin/env bash
# Build comparison summary, hypothesis verdict, and all result plots.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> SPINN/
python -m spinn.eval.compare_backends --results spinn/results/joint/metrics.csv
python -m spinn.eval.make_plots --metrics spinn/results/joint/metrics.csv --out spinn/results/figures

python -m spinn.scripts.make_report_tables --out spinn/results/report.md
