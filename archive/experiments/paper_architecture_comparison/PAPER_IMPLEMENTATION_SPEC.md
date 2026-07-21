# Paper implementation spec

This document records the implementation choices for building PIXEL and PIG as
potential PINN architectures for the Paul-trap RF solve. The target interface is
`(x, y) -> phi`, solving Laplace's equation with boundary conditions and
comparing RF potential error against the existing circular-trap DEVSIM field.

The important correction is that PIXEL and PIG should replace the potential
model `NN_theta`, not the coordinate projection map `NN_phi`. Existing
coordinate-shape experiments are useful prototypes, but they do not answer the
mentor-specified potential-model comparison.

## PIXEL

### What the paper actually proposes

PIXEL, "Physics-Informed Cell Representations for Fast and Accurate PDE
Solvers", proposes a trainable grid representation coupled to a small neural
network. Input coordinates query nearby grid cells through differentiable
interpolation; the resulting feature vector is processed by an MLP to produce
the scalar PDE solution.

The paper's motivation is to reduce spectral bias by assigning local trainable
parameters to local regions of the domain, while retaining PINN-style automatic
differentiation for PDE residuals.

### Equations needed for implementation

- PINN-style objective: Section "Physics-informed neural networks", Eq. (4),
  uses a weighted sum of residual, initial-condition, boundary-condition, and
  data losses:

  ```text
  L = lambda_res L_res + lambda_ic L_ic + lambda_bc L_bc + lambda_data L_data
  ```

- Residual loss: Eq. (5) defines mean squared PDE residual over collocation
  points.

- PIXEL solution approximation: Section "Neural networks and grid
  representations", Eq. (6):

  ```text
  u(x, t) ~= f(phi(x_hat, t_hat, C); theta)
  ```

  where `C` is the learnable grid and `f` is the MLP head.

- Grid interpolation: Eq. (7) defines the feature extractor as a weighted sum
  of neighboring cell features. For our 2D spatial Paul-trap problem this should
  be adapted from `(x, t)` to `(x, y)`.

- Smooth kernel: the text after Eq. (7) specifies the cosine interpolation
  kernel

  ```text
  k(s) = 0.5 * (1 - cos(pi * s))
  ```

  because higher-order PDE derivatives require a multiply differentiable
  interpolation function. This matters for Laplace residuals.

- Multigrid representation: Section "Multigrid representations", Eq. (8),
  sums features from shifted grids:

  ```text
  phi_multi = sum_m phi(shifted_coordinate_m, C_m)
  ```

  Figure 2 illustrates the shifted multigrid construction.

### Architecture

- Input representation: physical coordinates `(x, y)`.
- Coordinate normalization: map the rectangular computational box to the grid
  coordinate range expected by the interpolation routine. The paper describes
  normalized coordinates tightly bounded by the rectangular grid.
- Feature representation: learnable grid tensor `C`.
- Interpolation: differentiable smooth interpolation using the cosine kernel.
  A bilinear interpolation kernel is mentioned as valid for first-order PDEs,
  but the Paul-trap Laplace residual is second order, so the cosine kernel is
  the paper-supported choice.
- MLP head: shallow MLP after feature extraction. Appendix experimental setup
  reports one hidden layer, hidden dimension 16, and `tanh` activation for the
  shallow MLP architecture.
- Output representation: scalar potential `phi(x, y)`.
- Multigrid: include as a configurable option. The paper reports that
  multigrid was critical to performance and overfitting control.

### Losses

For the Paul-trap forward problem:

```text
L_res = mean((Delta phi)^2)
L_bc = mean((phi(boundary) - phi_bc)^2)
L_data = 0
```

There is no initial-condition term for the static RF Laplace problem. Use
Dirichlet boundary conditions from the circular trap convention: RF electrodes
driven and world/grounded electrodes held at the appropriate voltage.

### Training procedure

- The paper uses automatic differentiation to compute PDE derivatives with
  respect to input coordinates and to optimize both grid features `C` and MLP
  parameters `theta`.
- The main experimental setup uses L-BFGS with learning rate 1.0 and strong
  Wolfe line search, with randomly sampled collocation points in each L-BFGS
  iteration.
- For this repository, Adam can be used for compatibility with the existing
  Vanilla PINN comparison, but the spec should record that L-BFGS is the
  paper's default experimental optimizer.

### Assumptions needed for this repository

- Adapt `(x, t)` notation to `(x, y)` because the Paul-trap RF solve is a static
  2D Laplace problem.
- Use the same geometry sampler and boundary voltage convention as the Vanilla
  Paul-trap PINN.
- Compare only RF potential `phi` MSE against
  `DEVSIM/Circular_trap/paul_trap_basis_RF.dat`, per the current experiment
  scope.
- If a native cosine grid sampler is not available, either implement the cosine
  kernel directly in PyTorch or explicitly document any temporary use of
  `grid_sample` as an approximation to the paper.

### Parts that are ambiguous

- The paper's fast implementation relies on a custom CUDA sampler for higher
  derivatives. This repository should start with a clear PyTorch version and
  record any performance gap.
- The exact grid size and multigrid count for the Paul-trap geometry are not
  specified by the paper. Table 3 gives examples such as `(96, 4, 16, 16)` for
  PIXEL experiments, but the Paul-trap case needs a memory/accuracy sweep.
- The paper's boundary-condition treatment is generic PINN MSE. Hard boundary
  enforcement is not part of the base PIXEL method.

## PIG

### What the paper actually proposes

PIG, "Physics-Informed Gaussians as Adaptive Parametric Mesh
Representations", proposes a learnable Gaussian feature embedding followed by a
lightweight MLP. Each Gaussian has trainable position, covariance, and feature
value. During PINN training, the Gaussians can move and change scale, acting as
an adaptive parametric mesh.

The paper explicitly frames PIG as a PDE solution representation, not as a
coordinate deformation network.

### Equations needed for implementation

- PDE and boundary problem: Section 3.1, Eqs. (1)-(2), define the differential
  operator and boundary operator.

- Physics-informed objective: Section 3.1, Eq. (3), minimizes PDE residual and
  boundary losses:

  ```text
  L = mean(|D[u](x) - f(x)|^2) + boundary terms
  ```

- Gaussian feature embedding: Section 3.2.1, Eq. (4):

  ```text
  FE_phi(x) = sum_i f_i G_i(x)
  G_i(x) = exp(-0.5 * (x - mu_i)^T Sigma_i^{-1} (x - mu_i))
  ```

- Feature refinement: Section 3.2.2, Eq. (5):

  ```text
  u_{phi,theta}(x) = NN_theta(FE_phi(x))
  ```

- Neural-network interpretation: Section 3.2.3 and Figure 3 interpret PIG as
  an RBF layer with `N` Gaussian units, a linear feature layer, then a
  single-hidden-layer `tanh` MLP.

- Enhanced feature embedding: Appendix A.1, Eq. (10), allows different Gaussian
  parameters per feature dimension. This is optional; the base method is enough
  for a first Paul-trap comparison.

### Architecture

- Input representation: physical coordinates `(x, y)`.
- Coordinate normalization: rescale inputs to a compact box before Gaussian
  evaluation. Appendix A.2.2 rescales the 2D Helmholtz inputs to `[0, 1]^2`.
- Gaussian parameters:
  - `mu_i`: learnable center in 2D.
  - `Sigma_i`: learnable positive-definite covariance. The main experiments use
    diagonal covariance for efficiency.
  - `f_i`: learnable feature vector.
- Feature embedding dimension: the paper discusses `k=1` for simplicity and
  uses `k=4` in the 2D Helmholtz experiment.
- MLP head: shallow `tanh` MLP with one hidden layer. Section 3.2.3 and Figure
  3 support this structure; Appendix A.2.2 uses 16 hidden nodes for the 2D
  Helmholtz experiment.
- Output representation: scalar potential `phi(x, y)`.

### Losses

For the Paul-trap forward problem:

```text
L_pde = mean((Delta phi)^2)
L_bc = mean((phi(boundary) - phi_bc)^2)
L_total = lambda_pde L_pde + lambda_bc L_bc
```

No observational data term is required for training, but DEVSIM data should be
used for evaluation after training.

### Training procedure

- Section 4.1 uses Adam for most equations and L-BFGS for Helmholtz to compare
  fairly with PIXEL.
- The optimizer updates Gaussian centers, covariances, feature values, and MLP
  parameters jointly.
- The paper emphasizes that Gaussian centers move toward high-residual or
  high-frequency regions through the gradient of the loss with respect to
  `mu_i`.

### Assumptions needed for this repository

- Use the 2D Helmholtz setup as the closest implementation reference because it
  is a static 2D second-order PDE.
- Start with diagonal covariance and parameterize standard deviations through
  `log_sigma` or `softplus` so they remain positive.
- Use `N=3000`, feature dimension `k=4`, initial scale `sigma=0.1`, hidden size
  16, and Glorot-normal MLP initialization as the paper-supported 2D Helmholtz
  starting point. Smaller smoke-test values are acceptable for tests.
- Rescale Paul-trap coordinates from the physical box to `[0, 1]^2` before the
  Gaussian embedding, matching Appendix A.2.2.

### Parts that are ambiguous

- The paper reports multiple scale and weight initialization conventions across
  PDEs. For the Paul-trap Laplace problem, the 2D Helmholtz values are the
  closest match but not a proven optimum.
- The dense covariance variant is discussed but not the default; diagonal
  covariance is the appropriate first implementation.
- The paper sometimes evaluates with L-BFGS for fair comparison to PIXEL, while
  this repository may initially use Adam for comparable training loops.

## Mapping to this repository

### Existing files to reuse

- `PINN/Vanilla_Paul_Trap_PINN/Paul_Trap_PINN.py`: reference Vanilla potential
  model, geometry convention, voltage scaling, and pseudopotential convention.
- `Hyperparameter/scripts/compare_to_devsim.py`: DEVSIM RF reader, error
  statistics, and existing comparison plots.
- `PINN/common/tecplot.py`: reusable Tecplot parsing.
- `PINN/common/metrics.py`: derivative-aware field metrics if the comparison is
  later expanded beyond `phi` MSE.
- `experiments/coordinate_architecture_comparison/models/pixel_model.py`:
  reusable idea for learnable grid features, but its role is currently a
  coordinate displacement model and must not be reused without changing the
  output role to scalar potential.
- `experiments/coordinate_architecture_comparison/models/pig_model.py`:
  reusable Gaussian embedding code, with the same warning that the existing
  experiment predicts coordinate displacements, not potential.

### New files to add

Recommended isolated experiment:

```text
SPINN/experiments/potential_pinn_architectures/
  README.md
  config.py
  geometry.py
  losses.py
  train.py
  devsim_compare.py
  plotting.py
  run.py
  figures.py
  models/
    __init__.py
    base.py
    vanilla.py
    pixel.py
    pig.py
  tests/
    test_smoke.py
  results/
    .gitkeep
```

This should be separate from `coordinate_architecture_comparison`, which answers
a different question.

### Interfaces to preserve

All potential models should expose:

```text
model(points) -> potential
```

where `points` has shape `(N, 2)` and `potential` has shape `(N, 1)`.

The shared training loop should be architecture-agnostic:

```text
loss = lambda_pde * laplace_residual(model, interior_points)
     + lambda_bc * boundary_loss(model, boundary_points, boundary_values)
```

The shared evaluation loop should load DEVSIM RF data and compute:

```text
phi_mse
phi_rmse
phi_mae
phi_p95_abs
phi_mae_percent_of_300V
```

### Experiments to run

1. Smoke test each architecture for a few optimization steps.
2. Train Vanilla, PIXEL, and PIG for 5000 steps with the same sampler,
   collocation counts, boundary counts, seed, optimizer, learning rate, and
   boundary weight.
3. Save convergence curves for all three architectures.
4. Compare RF potential `phi` MSE against
   `DEVSIM/Circular_trap/paul_trap_basis_RF.dat`.
5. Save error maps `phi_model - phi_devsim` for each architecture.

## Open questions

- Should the first fair run use Adam for all three models, or should a second
  L-BFGS run be added because PIXEL and PIG both report strong L-BFGS results
  on second-order PDE benchmarks?
- Should PIXEL use an exact cosine interpolation implementation immediately, or
  is a temporary `grid_sample` baseline acceptable if explicitly labeled as not
  paper-faithful?
- What parameter budget should define fairness: equal wall-clock time, equal
  trainable parameters, or equal number of optimization steps?
- Should later metrics include electric-field and Hessian errors, even though
  the current mentor request is `phi` MSE only?
- Should the potential-architecture comparison be folded into the existing
  hyperparameter slide generator, or reported as a separate experiment first?
