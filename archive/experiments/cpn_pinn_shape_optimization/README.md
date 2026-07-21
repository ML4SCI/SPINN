# Coupled CPN + Vanilla PINN Shape Optimization

A separate, removable first experiment that couples a Coordinate Projection
Network (CPN) with the Vanilla PINN potential model and optimizes the geometry
under a coupled Laplace + ideal-pseudopotential loss. The aim is to see whether
circular electrodes deform toward a hyperbolic-like shape that produces a more
ideal quadratic pseudopotential, while the field still satisfies Laplace's
equation.

This is a first pass. It is lightly constrained on purpose: pathological maps
(folding, tearing, collapse) are expected and are observed, not prevented.

## What this experiment tests

1. Does the learned geometry move toward a hyperbolic-like shape?
2. Does the pseudopotential approach the ideal quadratic target?
3. Does the PDE residual stay small?
4. Do gradients flow through both the CPN and the PINN?
5. Does the coordinate map stay physically meaningful, or go pathological?

Success is not "the potential looks smooth." It is the five questions above.

## Why couple a CPN with a Vanilla PINN

The composed model is `phi(z) = NN_theta(NN_phi(z))`, where `NN_phi` is the CPN
coordinate map `z -> x` and `NN_theta` is the PINN potential. Optimizing the CPN
deforms the domain; optimizing the PINN solves the field. Coupling them makes
the geometry a differentiable parameter of the physics, which is the point of
coordinate-projection shape optimization.

## Losses

`L_PDE` (Laplace residual, physical coordinates):

    L_PDE = mean( (Delta_x phi)^2 )

computed with `pullback_laplacian` from `PINN/Coordinate_Projection/operators.py`,
so the Laplacian is taken in the deformed physical space `x = NN_phi(z)`.

`L_pseudo` (pseudopotential target):

    L_pseudo = mean( (Psi_model/<Psi_model> - Psi_ideal/<Psi_ideal>)^2 )   over points within fit_radius

`Psi_model` uses the pullback gradient `grad_x phi = J^{-T} grad_z u`:

    Psi_model = (e^2 / (4 M Omega^2)) |grad_x phi|^2,   reported in eV

following the existing convention (Ca-40, 10.2 MHz RF, cm geometry); see
`pseudopotential.py` for the citation.

Both pseudopotentials are normalized by their own near-center mean before
comparison, so the loss measures the *shape* of the well (is it quadratic), not
its absolute depth. This is deliberate. A secular-frequency target in the MHz
range corresponds to a far deeper well in eV than this 300 V, 2.5 cm trap can
produce, so an absolute-eV comparison is dominated by that overall scale
mismatch (target ~10^2 eV vs model ~10^-9 eV) and the loss explodes to
non-finite values rather than improving the geometry. Matching the normalized
shape isolates the question we actually care about: whether the deformed
pseudopotential becomes quadratic. The absolute scale `k` therefore only sets the
target shape, not a depth the model is forced to reach.

`Psi_ideal` (ideal quadratic target):

    Psi_ideal(x, y) = k (x^2 + y^2),   k = M omega_sec^2 / (2 e)   [eV/m^2]

`k` is derived from a target secular frequency so the curvature is physically
meaningful; the derivation closes the loop with `quadrupole_metrics`, which
recovers the same frequency from the Hessian of this target.

`L_BC` (boundary term, added to the user's two-term loss): electrode and world
voltage mismatch (East/West driven, North/South and world grounded), normalized
by the RF voltage squared so it is dimensionless and comparable to the
shape-normalized pseudopotential term. Without it, `phi = const` satisfies
Laplace exactly and the pseudopotential collapses, so the boundary term is
required for a non-trivial solution.

Total:

    L = lambda_pde L_PDE + lambda_pseudo L_pseudo + lambda_bc L_BC
        (+ optional deformation, smoothness, Jacobian regularizers, off by default)

## Gradient flow

`z` carries `requires_grad`; `x = NN_phi(z)` stays attached to the graph; the
pullback operators differentiate `phi` with respect to `z` and reuse the map's
Jacobian, so one backward pass reaches both `theta` (PINN) and `phi` (CPN). The
only intentional detach is `Psi_ideal`, evaluated on detached `x` so the target
is stationary. `gradient_checks.check_gradient_flow` verifies both groups receive
finite, nonzero gradients, and every run writes `gradient_diagnostics.json`.

## Training variations

| Variation | Description | Status |
|-----------|-------------|--------|
| A | Train PINN first (identity map), then optimize CPN with PINN frozen | full |
| B | Joint training with Jacobian regularization | scaffold |
| C | Joint CPN + PINN from the start | full |
| D | PINN-first with deformation/smoothness regularization | scaffold |
| E | Both pretrained, alternating schedule placeholder | scaffold |

Scaffolded variations run and are selectable but are not tuned.

## How to run

```bash
cd SPINN

python -m experiments.cpn_pinn_shape_optimization.run --variation C --steps 4000
python -m experiments.cpn_pinn_shape_optimization.run --variation A --steps 4000

python -m pytest experiments/cpn_pinn_shape_optimization/tests -q
```

## Where results are saved

`results/<variation_name>/` gets: `potential.png`, `pseudopotential.png`,
`pseudopotential_error.png`, `jacobian_determinant.png`, `deformation.png`,
`losses.png`, `metrics.json`, `gradient_diagnostics.json`.

## Interpreting weird transformations

With no Jacobian constraint, the map may fold (`det J` crosses zero), tear, or
collapse. Read `jacobian_determinant.png` and the `det_jacobian_min` field in
`metrics.json`: values at or below zero mean the map is no longer a valid
deformation, even if the potential still looks smooth. The `deformation.png`
electrode overlay shows whether electrodes bow inward (hyperbola-like) or distort
nonphysically. This is the signal that motivates the constraints listed under
Open questions.

## Architecture assumptions

- Pseudopotential convention: Ca-40 ion, 10.2 MHz RF, geometry in cm, value in
  eV, taken from the existing vanilla PINN and metrics code.
- The ideal target is a pure analytic quadratic near the center; there is no
  comparison against the DEVSIM hyperbolic trap in this pass.
- The CPN starts at identity (`zero_last_layer=True`) so `det J approx 1` at step
  zero and pathology develops gradually.
- Boundary voltages follow the project convention (East/West at the RF voltage,
  the rest grounded).

## Open questions

- Should the learned geometry be compared quantitatively against the existing
  DEVSIM hyperbolic trap (`DEVSIM/Hyperbolae_trap/`)?
- Which constraints to add next: Jacobian positivity, smoothness/curvature,
  boundary anchoring, electrode-shape regularization, symmetry, gap preservation,
  maximum displacement. The first pass exists to show which of these the failure
  modes actually demand.
- Whether variation A should fine-tune the PINN rather than freeze it
  (`pinn_frozen=False` flag is provided for the ablation).
- There is no supervised coordinate-pair target for the CPN, so the CPN is
  optimized only through the downstream physics and pseudopotential losses. No
  artificial coordinate target was invented.
