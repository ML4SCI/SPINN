#!/usr/bin/env bash
# Fixed-geometry PDE-solver benchmark for all backends and seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> SPINN/
for seed in 0 1 2; do
  for backend in mlp pixel pig; do
    python -m spinn.train.train_fixed --config "fixed_${backend}.yaml" --seed "$seed"
  done
done
