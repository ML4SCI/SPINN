# SPINN Hyperparameter Study

This folder contains the workflow for testing how the Vanilla PINN performance changes under hyperparameter variation for the 2D Circular Paul Trap.

The immediate study uses:

- DEVSIM Circular Trap as ground truth.
- Vanilla Paul Trap PINN as the model under test.
- RF-only comparison first: `Phi_RF` and derived `psi_RF`.
- One-factor-at-a-time sweeps so results are easy to interpret in group discussion.

## Adjusted Plan

1. Generate or refresh the Circular Trap DEVSIM result.
2. Train the default Vanilla PINN baseline.
3. Sweep one hyperparameter family at a time:
   - width
   - depth
   - activation function
   - boundary-condition loss weight
   - optimizer
   - learning rate
   - optional sampling counts and random seeds
4. For each run, save:
   - `config.json`
   - `training_history.csv`
   - `model_state_dict.pt`
   - `fields.npz`
   - `loss_history.png`
   - `phi_rf.png`
   - `psi_rf_eV.png`
5. Compare each trained model against DEVSIM at DEVSIM mesh nodes.
6. Aggregate normalized loss curves and DEVSIM error metrics.
7. Generate a slide deck for group discussion.

## Why The Plan Was Adjusted

The original plan is correct scientifically, but the code should not be run by manually editing constants. The implemented workflow exposes the key PINN settings through JSON configs, generates per-run artifacts, and compares every run to a common DEVSIM reference.

The comparison is done at DEVSIM mesh nodes to avoid interpolation hiding errors. The primary first-pass metrics are:

- normalized total loss: `loss(t) / loss(0)`
- time to normalized-loss thresholds: `1e-1`, `1e-2`, `1e-3`
- `MSE(Phi_PINN - Phi_DEVSIM)`
- `MSE(psi_PINN - psi_DEVSIM)`
- raw error plots for `Phi_RF` and `psi_RF`

## Files

```text
configs/
  smoke.json          Fast sanity-check config.
  baseline.json       Canonical full baseline.
  sweep_plan.json     First one-factor hyperparameter sweep.

scripts/
  run_sweep.py           Runs variants from a sweep plan.
  compare_to_devsim.py   Compares one run against DEVSIM RF output.
  aggregate_results.py   Builds summary CSV/JSON and plots.
  make_slides.py         Generates a PowerPoint deck.

results/
  runs/       Trained PINN run directories.
  summary/    Aggregated tables, plots, and slides.
```

## Environment

Virtual environments are still fine for this project.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r SPINN/DEVSIM/requirements.txt
pip install -r SPINN/PINN/Vanilla_Paul_Trap_PINN/requirements.txt
```

`pyvista` is preferred for reading DEVSIM Tecplot files. The comparison script also has a plain-text Tecplot fallback for the current 2D output.

## Step 1: Generate Circular Trap DEVSIM Ground Truth

```bash
cd SPINN/DEVSIM/Circular_trap
python gmsh_script.py
python paul_trap_sim_DC.py
python paul_trap_sim_RF.py
python visualize_results.py
cd ../../..
```

Expected RF output:

```text
SPINN/DEVSIM/Circular_trap/paul_trap_basis_RF.dat
```

If that file is not present, `compare_to_devsim.py` will also look for the older root-level `paul_trap_basis_RF.dat`.

## Step 2: Smoke Test The PINN Pipeline

```bash
python SPINN/PINN/Vanilla_Paul_Trap_PINN/Paul_Trap_PINN.py \
  --config SPINN/Hyperparameter/configs/smoke.json
```

Then compare the saved run:

```bash
python SPINN/Hyperparameter/scripts/compare_to_devsim.py \
  --run_dir SPINN/Hyperparameter/results/runs/<RUN_DIRECTORY>
```

## Step 3: Run The Sweep

Run the full configured sweep:

```bash
python SPINN/Hyperparameter/scripts/run_sweep.py \
  --plan SPINN/Hyperparameter/configs/sweep_plan.json
```

Run only selected variants:

```bash
python SPINN/Hyperparameter/scripts/run_sweep.py \
  --only baseline activation_sinusoidal optimizer_adamw
```

Run the first two variants only:

```bash
python SPINN/Hyperparameter/scripts/run_sweep.py --limit 2
```

## Step 4: Aggregate Results

Aggregate all runs:

```bash
python SPINN/Hyperparameter/scripts/aggregate_results.py
```

Aggregate a specific sweep manifest:

```bash
python SPINN/Hyperparameter/scripts/aggregate_results.py \
  --manifest SPINN/Hyperparameter/results/sweep_<TIMESTAMP>/manifest.json
```

Outputs:

```text
SPINN/Hyperparameter/results/summary/summary.csv
SPINN/Hyperparameter/results/summary/summary.json
SPINN/Hyperparameter/results/summary/normalized_loss_curves.png
SPINN/Hyperparameter/results/summary/phi_mse_by_variant.png
SPINN/Hyperparameter/results/summary/psi_mse_by_variant.png
SPINN/Hyperparameter/results/summary/final_normalized_loss_by_variant.png
```

## Step 5: Generate Slides

```bash
python SPINN/Hyperparameter/scripts/make_slides.py
```

Output:

```text
SPINN/Hyperparameter/results/summary/spinn_hyperparameter_results.pptx
```

## First Sweep Matrix

The first sweep plan includes:

| Family | Variants |
|---|---|
| Baseline | `width=128`, `depth=4`, `activation=tanh`, `bc_weight=100`, `optimizer=adam`, `lr=1e-3` |
| Width | `32`, `64`, `256` |
| Depth | `2`, `6`, `8` |
| Activation | `relu`, `sigmoid`, `sinusoidal` |
| BC weight | `1`, `10`, `1000` |
| Optimizer | `adamw`, `rmsprop`, `sgd` |
| Learning rate | `1e-4`, `3e-4`, `3e-3` |

Adam followed by LBFGS is intentionally left as a second-pass optimizer study. LBFGS is usually more meaningful with deterministic or quasi-deterministic collocation batches, while the current PINN resamples points every step.

## Interpretation Notes

- `Phi_RF` error measures voltage-field accuracy.
- `psi_RF` error is more demanding because it depends on gradients of `Phi_RF`.
- ReLU is worth testing, but it is not expected to be ideal for a Laplacian-based PINN because second derivatives are central to the residual.
- Sinusoidal activations may improve derivative quality, but a full SIREN-style initialization should be tested in a later architecture-focused study.
- A run can have good training loss and poor DEVSIM error, so the slide deck should report both.
