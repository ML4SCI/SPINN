# Potential PINN Architecture Comparison

This isolated experiment compares three potential models on the same circular
Paul-trap RF problem:

1. `vanilla`: ordinary smooth MLP, `(x, y) -> phi`.
2. `pixel`: PIXEL-style learnable grid features with cosine interpolation,
   followed by a shallow MLP, `(x, y) -> phi`.
3. `pig`: PIG-style learnable Gaussian feature embedding followed by a shallow
   MLP, `(x, y) -> phi`.

All three solve the same strong-form problem:

```text
L = mean((Delta phi)^2) + lambda_bc * mean((phi_boundary - phi_bc)^2)
```

and are evaluated against the circular-trap DEVSIM RF solution at
`DEVSIM/Circular_trap/paul_trap_basis_RF.dat`. The evaluation is only `phi`
error by design: MSE, RMSE, MAE, percent of 300 V, and an error map.

This is intentionally separate from `coordinate_architecture_comparison`.
PIXEL and PIG are used here in the role described by their papers: potential
solution representations. They are not coordinate projection networks and do
not output deformed coordinates.

## Run

From `SPINN/`:

```bash
python -m experiments.potential_pinn_architectures.run --model all
```

For a quick smoke run:

```bash
python -m experiments.potential_pinn_architectures.run \
  --model all --steps 10 --n-interior 64 --n-world 32 --n-electrode 32 --skip-devsim
```

Run one model:

```bash
python -m experiments.potential_pinn_architectures.run --model pixel --steps 5000
```

Override hyperparameters:

```bash
python -m experiments.potential_pinn_architectures.run \
  --model pig \
  --override model.pig_num_gaussians=3000 \
  --override training.optimizer=adam
```

## Outputs

Each model writes to `experiments/potential_pinn_architectures/results/<model>/`:

- `config.json`
- `training_history.csv`
- `loss_history.png`
- `model_state_dict.pt`
- `comparison_phi_rf.png`
- `comparison_metrics.json`
- `result.json`

Running all models also writes:

- `convergence_comparison.png`
- `phi_mse_comparison.png`

## Architecture assumptions

- PIXEL implements the paper's smooth cosine interpolation rather than PyTorch
  bilinear `grid_sample`, because Laplace requires second derivatives.
- PIXEL multigrid is configurable through `model.pixel_num_grids`. The default
  is modest for local runs; paper-scale values should be swept separately.
- PIG uses diagonal covariance, matching the efficient default discussed in the
  paper.
- PIG coordinates are rescaled to `[0, 1]^2` before Gaussian evaluation, matching
  the paper's 2D Helmholtz setup.
- PIG defaults to fewer than 3000 Gaussians for practical local training. Use
  `--override model.pig_num_gaussians=3000` for the paper-like 2D Helmholtz
  scale.

## Open questions

- Whether the fair comparison should be equal steps, equal wall time, or equal
  parameter count.
- Whether to add an L-BFGS phase, since PIXEL and PIG both report strong
  second-order-optimizer results on static second-order PDEs.
- Whether the next metric should include electric-field and Hessian error, not
  just `phi` MSE.
