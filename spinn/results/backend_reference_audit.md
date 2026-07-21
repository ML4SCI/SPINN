# PIXEL/PIG Reference Audit

This compares the local SPINN physics backends against the cloned reference repositories in `external/`.

## Bottom Line

The local SPINN backends now match the **core official PIXEL and PIG parameterizations** much more closely:

- PIXEL now uses learnable multigrid cell features, official-style cosine interpolation with per-cell offset, tiny cell initialization, summed cell features, and a feature-only decoder.
- PIG now uses the official channel-wise layout: `feature_dim` independent Gaussian channels, each with `K` centers, sigmas, and scalar weights, followed by a decoder MLP.

The remaining differences are intentional SPINN adaptations:

- Coordinates are mapped from Paul-trap physical coordinates `[-R_out, R_out]^2` into the official normalized coordinate range.
- Training is embedded inside the coordinate-projection shape-optimization loop.
- Losses are Paul-trap Laplace, boundary, eta, and geometry-constraint losses rather than the official repos' Helmholtz/convection/etc. example losses.

Safe wording:

> We implemented PIXEL and PIG physics backends in SPINN using the official feature parameterizations, adapted to the Paul-trap coordinate-projection problem.

## Reference Files Checked

- `external/PIXEL/network.py`
- `external/PIXEL/pixel.py`
- `external/PIXEL/utils/grid_sample.py`
- `external/PIXEL/main.py`
- `external/Physics-Informed-Gaussians/Helmholtz2d/network.py`
- `external/Physics-Informed-Gaussians/Helmholtz2d/pig.py`
- `external/Physics-Informed-Gaussians/Helmholtz2d/main.py`

## Local Files Updated

- `spinn/models/backends/pixel.py`
- `spinn/models/backends/pig.py`
- `spinn/models/backends/decoders.py`
- `spinn/config.py`
- `spinn/configs/joint_pixel.yaml`
- `spinn/configs/joint_pig.yaml`

## PIXEL Status

### Now Matching

- Official-style learnable cells:

```text
cells / feature_grids: (n_cells, cell_dim, cell_size, cell_size)
```

- Official-style interpolation:

```text
coordinates in [-1, 1]
scale to cell grid
add per-cell offset
cosine interpolation
sum over n_cells
```

- Official-style decoder input:

```text
decoder input = interpolated cell features only
```

- Official-style initialization:

```text
feature cells initialized near zero, around [-1e-5, 1e-5]
```

### Remaining SPINN Adaptations

- SPINN maps trap coordinates from `[-R_out, R_out]` to `[-1, 1]`.
- SPINN uses a pure PyTorch interpolation path rather than the official CUDA `CosineSampler2d`.
- SPINN exposes the backend through `build_backend(config)` and trains it jointly with the coordinate projection network.

### Current Official-Scale Defaults

```text
pixel_grid_resolution = 16
pixel_feature_dim = 4
pixel_num_grids = 16
pixel_decoder_layers = 2
pixel_decoder_width = 16
```

The official PIXEL command file also uses larger `n_cells` variants such as 64 and 96. Those can be tested by changing `pixel_num_grids`.

## PIG Status

### Now Matching

- Official channel-wise Gaussian layout:

```text
means:   (feature_dim, K, 2)
sigmas:  (feature_dim, K, 2)
weights: (feature_dim, K)
```

- Official diagonal covariance feature equation:

```text
feature_j(x) = sum_k weight[j, k] * exp(-0.5 * sum(((x - mu[j, k]) / sigma[j, k])^2))
```

- Official decoder layout:

```text
decoder input = feature_dim Gaussian-channel features
```

- Official-style initialization:

```text
centers uniform in [-1, 1]
weights normal with std 0.01
sigmas initialized to sigma_init
```

### Remaining SPINN Adaptations

- SPINN maps trap coordinates from `[-R_out, R_out]` to `[-1, 1]`.
- SPINN currently implements the diagonal covariance path, which is the normal efficient path in the reference code.
- Full covariance is not wired into SPINN yet.
- SPINN uses separate learning rates for decoder/weights, centers, sigmas, and the shape network during joint optimization.

### Current Official-Scale Defaults

```text
K = 800
feature_dim = 4
decoder_layers = 2
decoder_width = 16
init_sigma = 0.10
```

These match the scale of the official Helmholtz2d PIG example.

## Remaining Caveats

The code is now much closer to the official implementations, but a result from SPINN is still not a reproduction of the PIXEL/PIG papers because the PDE, geometry, objective, and optimizer coupling are different.

For presentations, distinguish:

- **Implementation match:** the feature representation now follows the official structures.
- **Experiment difference:** SPINN uses those representations inside Paul-trap coordinate-projection shape optimization.

## Suggested Next Checks

1. Rerun the fixed-geometry benchmark for MLP, PIXEL, and PIG.
2. Rerun the 3000-step joint benchmark after clearing or separating the old smoke metrics.
3. Compare PIXEL with `pixel_num_grids = 16`, `64`, and `96`.
4. Compare PIG with diagonal covariance against a future full-covariance variant only after the diagonal version is stable.
5. Regenerate plots and update `plot_navigation_summary.md` after new metrics exist.
