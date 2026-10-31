# exp

Multi-script experiment campaigns. Like `scratch/`, every script saves a plot
(here into the campaign's `results/` folder) and runs from the repository
root, for example:

```bash
python3 exp/harmonic_source_shape_search/10_boundary_shape_derivative.py
```

Scripts that tested a rejected idea are in `archive/exp_retired/`, under the
same campaign name.

## boundary_green_trainer

`trainer.py` is the end-to-end shape optimizer built from the pieces below. It
moves only the physical electrode surface and never warps coordinates.

1. Rods are described by a support function p(phi) = R + c4 cos 4phi + c8 cos 8phi. One rod is cloned to all four positions by the trap symmetry.
2. The field is a sum of log (Green-function) sources inside the electrodes, tied across rods with signs (+, +, -, -). Their coefficients come from a least-squares fit to the electrode voltages, so the field is harmonic exactly.
3. The shape gradient is the Hadamard boundary form: the sensitivity field equals -h dV/dn on the electrodes, solved in the same basis.
4. The outer loop is gradient descent or Gauss-Newton on the multipole coefficients q6 and q10.

On R alone it recovers the circular optimum R = 1.14516 r0 from three starts.
On (R, c4) it converges to one q6 = q10 = 0 electrode (R = 1.6463, c4 = 0.0453).
With a Gaussian PIG basis of the same parameter count, the electrode voltage
error rises from 1.5e-3 V to 5.7e-2 V, and the shape it converges to is worse
than the starting rod when re-read with the Green field.

Known limits:

- The electrode voltage error grows with rod size, so the cancelled multipoles are only as accurate as that fit.
- The c8 Hadamard derivative is off by up to 7% against finite differences.
- More sources per rod should help both.

## harmonic_source_shape_search

Exact-harmonic (MFS) electrode search. Two parts carry through to the trainer:

- `10_boundary_shape_derivative.py`: the Hadamard shape derivative matches full geometry re-solves to about 1e-6. Because only normal motion enters, tangential reparameterization drops out by construction, which removes the coordinate gauge problem without a mask.
- `11_green_pig_architecture.py` and `13_symmetry_orbit_green_pig.py`: Green-function atoms instead of Gaussian atoms, and symmetry-orbit tying, which cuts free coefficients from 321 to 80 at the same accuracy and drops the condition number from 1.5e16 to 2.5e10.

The rest:

- `01_source_conditioning.py` and `02_differentiable_radius_gradient.py` set up the MFS reference.
- `06`-`09` find and screen the support-function electrode that cancels both q6 and q10.
- `14_pixel_kernel_smoothness.py` is the analytic check behind the quintic PIXEL kernel.

## claude_tree_stress_test

This campaign started as an audit of an AI-generated tree of research ideas.
Most of its theorem-level claims did not hold, and those audit scripts are
retired. These scripts produced results worth keeping:

- `05`-`10`, conformal center-jet electrodes: D4-equivariant rational maps with poles inside the electrodes cancel q10 and q14 sequentially. `10` confirms the result with an independent fixed-enclosure solve.
- `11_manufacturing_tolerance.py`, fabrication tolerance: exact symmetry removes saddle drift, but makes allowed profile errors add coherently.
- `02_shape_visibility_spectrum.py`: center-only measurements see a very low-rank projection of the rod shape.
- `04`, `14`-`16`, `18`, `19`, `22`-`24`, trajectory-level objectives:
  - Lifetime is not monotone in dodecapole strength.
  - The tested beta = 0.025 electrode creates a small stable island rather than a broad gain.
  - Different lifetime objectives select different shapes.
  - Averaging over a fabrication distribution gives a usable gradient where pathwise autodiff does not.
  - `24` reads the `.npz` that `23` writes.
- `20`, `21`, conditioned boundary coordinates: SVD coordinates for boundary motion. Jet cancellation has to proceed in order, low to high.
- `30`-`32`, the exact-harmonic geometry head: an analytic harmonic decoder whose geometry-dependent coefficients are learned. It is the strongest central-field surrogate here, valid inside a tested geometry trust region.
- `51`-`54`, differentiating the MFS layer: implicit and then fully analytic derivatives of the exact MFS layer, which match fresh re-solves to about 1e-6. `52`-`54` load `51` (or `52`) through `runpy`.

Plots 35-38 and 40-50 in this campaign's notes have no saved scripts.

## coulomb_natural_units_stress_test

This tests rescaling each loss term by the Coulomb constant
C = k_e e = 1.44e-9 V m so that the potential, field and Laplace terms share
units before they are summed. The PDE itself is left unchanged.

- `experiment.py` tests the first formulation, which squared the potential as the network output. It solves a different PDE and is rejected.
- `loss_only_experiment.py` is the corrected version, which converts only the residuals. It is a valid unit convention.

Sharing units does not make the terms numerically comparable. At
initialization the converted terms differ by about eight orders of magnitude
in value and gradient norm. With equal weights the field error was 10.3x worse
than the ordinary dimensionless run. Fixed per-term weights set from each
term's initial gradient norm almost recovered the baseline, so the balancing
comes from gradient-norm weighting, not from the unit change.

`gradnorm_experiment.py` tests that directly. It uses the same problem, seed
and 400 Adam steps, with weights of 1/|grad L_i| refreshed every 20 steps by an
EMA. Each converted term is a constant multiple of its dimensionless term, so
these weights cancel the conversion. The Coulomb and dimensionless adaptive
runs end at the same Laplace MSE (1.11e-4), V MSE (2.4e-5) and E MAE (5.85e-3).
Their histories match except for transient float round-off. Against the plain
dimensionless run with alpha = 1, adaptive weighting:

- lowers the Laplace MSE 8.7x;
- lowers the V MSE 2.1x;
- lowers the E MAE 1.4x;
- slightly raises the worst boundary error, from 1.2e-2 to 1.4e-2.

So unit matching is harmless but does nothing under gradient-norm weighting.
The weighting is what helps.
