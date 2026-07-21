# Coordinate Architecture Comparison

An isolated experiment comparing three feature representations for the
coordinate-projection (shape) network of the Zhang et al. framework:

1. **Vanilla** — residual MLP, mirroring
   `PINN/Coordinate_Projection_Network/CPN.py`.
2. **PIXEL** — learnable feature grid with bilinear interpolation
   (Kang et al., AAAI 2023).
3. **PIG** — learnable Gaussian feature embedding
   (Kang et al., ICLR 2025).

All three share the interface `model(z) -> predicted_coordinates`, where `z`
is a batch of 2D reference coordinates and the output is a batch of 2D
predicted coordinates, and all preserve the residual structure

```
phi(z) = z + phi_hat(z)
```

so only the displacement `phi_hat` is learned. The whole experiment lives
under this folder and can be deleted, merged, or promoted without touching
the rest of the repository.

## What this experiment compares

A fair, shared pipeline (same sampler, seed, optimizer, learning rate, step
count, batch sizes, domain bounds, normalization, and metrics — see
`config.py`) trains each architecture on the same reference geometry. The
default training loss is the **mean Euclidean coordinate distance** (matching
CPN.py); **mean squared coordinate error (MSE)** is tracked alongside it for
the comparison plots.

## How the vanilla model is reused

`CPN.py` runs a full training job at module import time, so it cannot be
imported as a library without side effects. By project preference CPN.py is
left **completely untouched**. Instead:

- `models/vanilla_wrapper.py` re-declares the same small residual-MLP
  architecture as CPN.py's `CoordinateProjectionNet` / `MLP`.
- `training/data.py` re-implements the same geometry and interior /
  world-boundary / electrode sampling.

Both are faithful mirrors of CPN.py. If CPN.py's network or geometry changes,
update these two files to match.

## Files

| Architecture | Model file | Config |
| --- | --- | --- |
| Vanilla | `models/vanilla_wrapper.py` | `configs/vanilla.yaml` |
| PIXEL | `models/pixel_model.py` | `configs/pixel.yaml` |
| PIG | `models/pig_model.py` | `configs/pig.yaml` |

Shared pipeline: `config.py`, `models/base.py`, `models/__init__.py`
(registry), `training/{data,losses,metrics,train_one_model}.py`,
`plotting/{plot_losses,plot_fields,plot_comparison}.py`.

## Running

Run one model (either form works):

```bash
python -m experiments.coordinate_architecture_comparison.scripts.run_vanilla
python -m experiments.coordinate_architecture_comparison.scripts.run_pixel
python -m experiments.coordinate_architecture_comparison.scripts.run_pig

python -m experiments.coordinate_architecture_comparison.training.train_one_model --model vanilla
python -m experiments.coordinate_architecture_comparison.training.train_one_model --model pixel
python -m experiments.coordinate_architecture_comparison.training.train_one_model --model pig
```

Run the full comparison (trains all three, writes comparison artifacts):

```bash
python -m experiments.coordinate_architecture_comparison.scripts.run_all_comparisons
```

Run the smoke tests:

```bash
python -m pytest experiments/coordinate_architecture_comparison/tests -q
```

## Where results are saved

```
results/
  vanilla/  loss_history.png  displacement_field.png  error_heatmap.png  metrics.json
  pixel/    loss_history.png  displacement_field.png  error_heatmap.png  metrics.json
  pig/      loss_history.png  displacement_field.png  error_heatmap.png  metrics.json
  comparison/
    convergence_comparison.png
    mse_comparison.png
    metrics_summary.csv
```

Each `metrics.json` stores the resolved config and final metrics
(`mean_euclidean_error`, `mse`, `rmse_coordinate`, `max_euclidean_error`).

## Architecture assumptions

PIXEL and PIG are **scalar physics-field solvers** in both papers and in the
SPINN project proposal — they replace the MLP *physics* network `NN_theta`,
not the *coordinate* network `NN_phi`. This experiment instead compares them
**as coordinate-projection maps**, so the following adaptations were made.
They preserve each paper's feature representation but change the output role.

**PIXEL** (`pixel_model.py`)

- Faithful: a learnable 2D feature grid queried by bilinear interpolation
  (`F.grid_sample`, `align_corners=True`), followed by a shallow MLP.
- Adapted: the MLP outputs a 2D **displacement** instead of a scalar field,
  and the result is wrapped in `z + phi_hat(z)`.
- Assumption: query coordinates are normalized to `[-1, 1]` using the fixed
  reference-domain bounds; grid resolution and feature dimension are config
  knobs (defaults 32 and 4).

**PIG** (`pig_model.py`)

- Faithful: the feature embedding
  `FE(x) = sum_i f_i * exp(-0.5 (x - mu_i)^T Sigma_i^{-1} (x - mu_i))`
  with learnable means, diagonal covariances (the paper's efficient choice),
  and feature vectors, followed by a single-hidden-layer MLP.
- Adapted: the MLP outputs a 2D **displacement** instead of a scalar field,
  and the result is wrapped in `z + phi_hat(z)`.
- Assumptions: means initialized uniformly over the reference domain;
  covariances initialized isotropic (`init_sigma`, default 0.1) and
  parameterized through `log_sigma` to stay positive. The paper uses tanh
  hidden units and feature dimension ~1–4; defaults here are 16 hidden units
  and feature dimension 4.

All three displacement heads are zero-initialized so every model starts as
the exact identity map, matching the vanilla `zero_last_layer` behavior.

## Open questions

- **Identity target makes the comparison trivial.** The current supervised
  target is the identity map, and all three models start *at* the identity
  (zeroed heads). The residual therefore starts at zero with zero gradient,
  so every architecture reports ~0 loss with nothing to distinguish them.
  This faithfully reproduces the vanilla setup ("the residual should learn
  approximately zero") but is uninformative as an *architecture* comparison.
  To get a meaningful comparison, drive the models away from the trivial
  solution by either (a) disabling zero-initialization so each architecture
  must learn its way back to identity from a random start, or (b) supplying a
  non-trivial deformation target. The hook for this is intentionally left to
  you rather than chosen here — see `training/train_one_model.py` and the
  `zero_last_layer` / model init flags.
- **Loss weighting across point sets.** Interior, world-boundary, and
  electrode points are concatenated into one batch and weighted only by their
  relative counts (as in CPN.py). Whether boundary points should be weighted
  more heavily for the projection task is unresolved.
- **PIG covariance form.** Only diagonal covariances are implemented (the
  paper's efficient default). Full covariances may matter for anisotropic
  deformations and are not implemented.

## References

- P. Zhang et al., *A physics-informed neural network approach for shape
  optimization*, 2024.
- N. Kang et al., *PIXEL: Physics-Informed Cell Representations for Fast and
  Accurate PDE Solvers*, AAAI 2023. <https://arxiv.org/abs/2207.12800>
- N. Kang et al., *PIG: Physics-Informed Gaussians as Adaptive Parametric
  Mesh Representations*, ICLR 2025. <https://arxiv.org/abs/2412.05994>
