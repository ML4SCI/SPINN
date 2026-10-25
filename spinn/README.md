# SPINN Paul-Trap Shape Optimization

Implementation of `SPINN_Experiment_Plan.pdf`: a coordinate-projection shape
optimizer for a 2D four-rod linear Paul trap, comparing three physics backends
(MLP / PIXEL / PIG) under one optimizer, with the **geometric efficiency `eta`**
as the headline metric and an **independent finite-difference (FEM) solver** as
the validator.

The main comparison is **CPN + MLP vs CPN + PIXEL vs CPN + PIG**, where the
coordinate-projection network (CPN, `NN_phi`) is the *fixed* geometry
parameterization and the backend (`NN_theta`) is the *varied* physics
representation. Any difference in final `eta` should come from the physics
representation, not from a different shape optimizer.

## Active package layout

```text
spinn/
  config.py                 typed config, validation, overrides, YAML inheritance
  configs/                  shared fixed/joint bases plus backend overrides
  geometry/
    four_rod.py             normalized circular four-rod trap (r0=1, R_rod=1.145, R_out=5)
    collocation.py          reproducible interior and boundary training batches
    masks.py                s_design / anchor masks
    export_fem.py           deformed/reference electrode polygons for the validator
  models/
    layers.py               shared MLP builder, activations, [-R_out, R_out]^2 box normalization
    shape_network.py        NN_phi: x = z + alpha s(z) Delta_phi(z), zero-init, symmetry projection
    composed.py             V_theta(NN_phi(z)) and the identity shape map
    backends/
      mlp.py pixel.py pig.py decoders.py
  physics/
    laplace.py              pullback + cartesian Laplacian, map Jacobian
    eta.py                  eta_autograd, eta_fit, harmonic_purity
    fem_reference.py        finite-difference Laplace solve + FEM-validated eta
  losses/
    pde.py boundary.py geometry_constraints.py objective.py
  train/
    train_fixed.py          fixed-geometry backend benchmark
    train_joint.py          joint field/shape optimization
    runtime.py              CLI, seeding, device/dtype, LR schedules, gradient steps,
                            CSV/JSON/checkpoint artifacts, run manifests
  eval/
    evaluate.py make_plots.py compare_backends.py
  foundations/              backend-independent problem specs and the coordinate-gauge audit
  scripts/                  run_all_*.sh, make_report.sh, ablation grid, hard-BC tuner, diagnostics
  tests/                    identity, laplace autograd, eta, jacobian, BCs, config, train smoke
  results/                  fixed/ joint/ backend/ figures/
```

## Geometry and units

Dimensionless: `r0 = 1`, `V0 = 1`, rod radius `R_rod = 1.145 r0`, rod centers at
`(+/-(r0+R_rod), 0)` and `(0, +/-(r0+R_rod))`, outer disk `R_out = 5 r0`.
Boundary conditions: `+V0/2` on the x-axis rods, `-V0/2` on the y-axis rods, `0` on
the outer circle.

## Installation and commands

From `SPINN/`:

```bash
python3 -m pip install -e '.[dev]'

# 1. Fixed-geometry PDE-solver benchmark
python3 -m spinn.train.train_fixed --config fixed_mlp.yaml   --seed 0
python3 -m spinn.train.train_fixed --config fixed_pixel.yaml --seed 0
python3 -m spinn.train.train_fixed --config fixed_pig.yaml   --seed 0

# 2. Main joint shape-optimization benchmark
python3 -m spinn.train.train_joint --config joint_mlp.yaml   --seed 0
python3 -m spinn.train.train_joint --config joint_pixel.yaml --seed 0
python3 -m spinn.train.train_joint --config joint_pig.yaml   --seed 0

# 3. All seeds
bash spinn/scripts/run_all_fixed.sh
bash spinn/scripts/run_all_joint.sh

# 4. Compare + plot
bash spinn/scripts/make_report.sh
# per-run geometry/backend diagnostics:
python3 -m spinn.eval.evaluate --config joint_pig.yaml \
  --model spinn/results/joint/pig_seed0/model.pt --out spinn/results/figures/pig_seed0
```

First start with a small smoke run with small counts and few steps:

```bash
python3 -m spinn.train.train_joint --config joint_pig.yaml --steps 20 \
  --override sampling.interior_points=500 --override validation.fem_grid=80
```

Tests:

```bash
python3 -m pytest
ruff check spinn
ruff format --check spinn
```

Every training run writes a `manifest.json` beside its model and result. The
manifest records the resolved configuration, Python/package versions, Git
revision and working-tree state, timestamps, device settings, and final metrics.

## Outputs (plan §6, §8)

* `results/fixed/metrics.csv` — `backend, seed, step, loss_pde, loss_bc, eta_pinn, eta_fem, eta_abs_error, wall_time_sec`
* `results/joint/metrics.csv` — per-checkpoint eta_pinn / eta_fem, all loss
  terms, `min_detJ`, wall time
* `results/joint/checkpoints.csv` — checkpoint/geometry/FEM paths + validity
* `results/figures/` — `eta_fem_vs_checkpoint.png` (headline),
  `eta_pinn_vs_fem_scatter.png` (trustworthiness), `eta_pinn_vs_epoch.png`,
  `constraints_vs_epoch.png`, and per-run potential/residual/deformation/detJ/
  contour/backend-diagnostic plots
* `results/joint/comparison_summary.json` — per-backend final FEM eta, mismatch,
  and the H1/H0 verdict (plan §7 thresholds: PIG wins if `>=0.03` absolute or
  `>=5%` relative over the best baseline)
