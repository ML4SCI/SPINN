# Coordinate-Projection Paul-Trap Experiment

This experiment learns a parametric RF potential on a fixed reference domain
while a differentiable map moves the four circular electrode centers. It is the
first constrained geometry step in `Resources/Project_Audit_and_Roadmap.md`.

## What is implemented

- Identity-initialized freeform coordinate map.
- Exact compact-support map for symmetric electrode translations.
- Correct strong-form Laplace pullback:

  \[
  \Delta_x\phi =
  \frac{1}{\det J}\nabla_z\cdot
  \left(\det J J^{-1}J^{-T}\nabla_z u\right).
  \]

- Jacobian determinant, distortion, world-anchor, symmetry, gap, and curvature
  utilities.
- Manufactured affine and radial deformation tests.
- Parametric potential network `u(z, d)` over electrode distance `d`.
- Differentiable quadrupole-efficiency optimization.
- Independent parameterized Gmsh/DEVSIM solves and central finite-difference
  comparison with the PINN geometry derivative. The default check uses a fine
  mesh and both three- and five-point centered stencils.

The initial validated distance interval is `0.11-0.145 cm`. It is deliberately
constrained so the compact-support map remains injective and translates each
electrode exactly. Expanding this range requires a more global bijective map,
not merely weaker Jacobian penalties.

## Mentor-Specified Residual Architecture

The intended freeform shape network follows Equation 13 of Zhang, Lin, and
Wang:

\[
NN_\phi(\mathbf z)=\mathbf z+\widehat{NN}_\phi(\mathbf z).
\]

`ResidualCoordinateMap` implements this equation directly. Its default
architecture matches the paper: six hidden layers of width 50 with `tanh`.
The output layer is initialized to zero, so
`\widehat{NN}_\phi(\mathbf z)=0` and the full projection starts as the exact
identity map. It therefore satisfies the same initial functional condition as
the paper's supervised identity pretraining without first approximating a known
identity target.

The two maps have different roles:

- `ElectrodeTranslationMap` is the constrained Stage-1 validation map. It gives
  a known geometry family for checking PDE derivatives against DEVSIM.
- `ResidualCoordinateMap` is the paper-aligned Stage-2 freeform architecture.
  It should be trained jointly with a physical-field network, with Jacobian,
  fixed-region, symmetry, volume/gap, and manufacturing penalties.

`CoordinateProjectionPINN` implements the full composition
`NN_theta(NN_phi(z))`, and `PhysicalPotentialMLP` uses the same default
six-layer, width-50 `tanh` architecture. This paper-aligned composition is now
available for the freeform training stage; it is not substituted silently into
the constrained validation experiment.

The smoke-tested parametric experiment still uses the constrained map. A
converged freeform joint-training result has not yet been claimed.

## Commands

Run tests:

```bash
cd SPINN
python -m pytest
```

Run the inexpensive end-to-end smoke experiment:

```bash
python -m PINN.Coordinate_Projection.train \
  --config PINN/Coordinate_Projection/configs/smoke.json
```

Run the full parametric training and DEVSIM derivative check:

```bash
python -m PINN.Coordinate_Projection.train \
  --config PINN/Coordinate_Projection/configs/full.json
```

Re-run DEVSIM validation for an existing coordinate-projection checkpoint:

```bash
python -m PINN.Coordinate_Projection.validate_gradient \
  --run-dir PINN/Coordinate_Projection/results/<run>
```

Run a standalone DEVSIM geometry sweep:

```bash
python -m PINN.Coordinate_Projection.devsim_reference \
  --distances 0.115 0.125 0.135 \
  --output-dir PINN/Coordinate_Projection/results/devsim_sweep
```

## Acceptance gates

Do not interpret the optimized distance until all of these hold:

- minimum map determinant is positive with margin;
- held-out transformed residual and boundary errors have converged;
- center Hessian obeys the Laplace trace condition;
- PINN efficiency and its derivative agree with mesh-refined DEVSIM;
- the optimum is stable over at least three seeds.

The three-step smoke model only checks execution. Its predicted geometry
gradient is not scientifically meaningful.

The current structured-grid escape-barrier calculation is also diagnostic only:
its value is sensitive to grid resolution and whether the discrete path reaches
an electrode or the outer world first. It must not be reported as physical trap
depth until a mesh-refined saddle/escape calculation is added.
