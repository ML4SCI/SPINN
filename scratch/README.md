# scratch

Short, single-question experiments. Each script asks one question, runs in
seconds to a few minutes, and saves one plot next to itself
(`scratch/NN_name.png`). Run them from the repository root:

```bash
python3 scratch/10_mfs_known_optimum.py
```

Most import from `spinn/`, so install the package first
(`python3 -m pip install -e .`). Many of the PIG and PIXEL scripts from 34
onward share helpers in `pig_gate_common.py`.

The numbering is chronological, so there are gaps. Scripts that tested an idea
which failed, or that a later script replaced, are in
`archive/scratch_retired/` (see `archive/README.md`).

## Coordinate gauge and the deformation mask

The original shape network's deformation mask was nonzero in the central
measurement region and at the enclosure. A coordinate warp could therefore
change the reported curvature objective without moving any electrode. These
scripts show the problem and test the corrected electrode-collar mask.

| Script | What it shows |
|---|---|
| `01_deformation_mask.py` | Where the old and corrected masks are nonzero. |
| `03_mask_eta_gauge_test.py` | With the field held at an exact quadrupole, the old mask lets a dilation change the reported eta; the corrected mask does not. |
| `04_mask_derivatives.py` | The corrected mask and its first two derivatives are exactly zero in the center anchor and finite elsewhere. |
| `05_collar_width_stress.py` | Jacobian quality under 0.10-0.15 r0 rod motions; a 0.35 r0 collar was the best tested width. |
| `06_collocation_jacobian_coverage.py` | Uniform sampling under-samples the low-Jacobian collar; stratified sampling hits it 4-5x more often. |
| `07_pullback_measure.py` | Integrals over a mapped domain need the det J weight. |
| `13_joint_vs_bilevel_gradient.py` | Once the center is anchored, the frozen-field shape derivative is exactly zero, so the simultaneous joint loss loses the physical response term. |

## Exact references and the bilevel shape loop

The method of fundamental solutions (MFS) with exact boundary points is the
reference every learned solver is checked against.

| Script | What it shows |
|---|---|
| `10_mfs_known_optimum.py` | MFS recovers the classical circular-rod optimum, a6/a2 = 0 at R/r0 = 1.14517. |
| `11_objective_gradient_gate.py` | The design-loss gradient converges under finite-difference refinement and changes sign at the optimum. |
| `12_one_parameter_electrode_optimization.py` | Gradient descent on R reaches the optimum from both sides. |
| `14_differentiable_inner_solve.py` | A differentiable inner solve gives the shape gradient to 4e-4 relative error. |
| `35_halfway_mfs_reference_audit.py` | Convergence audit of the MFS reference used by the PIG gates. |
| `38_direct_vs_total_shape_gradient.py` | Separates the direct shape partial from the total derivative through the re-equilibrated field. |
| `60_exact_nested_geometry_baseline.py` | Exact nested field-then-geometry loop; the first outer-loop gate for any learned inner solver. |

## Symmetry

The four-rod trap has known symmetry. Imposing it exactly at the output and on
the geometry works. Tying internal Gaussian features does not (those scripts
are retired).

| Script | What it shows |
|---|---|
| `16_mlp_symmetry_gate.py` | Exact quadrupole symmetry fixes the MLP's wrong-sign a6 but does not finish the solve. |
| `17_field_symmetry_modes.py` | Toggleable output symmetry modes (two-axis, one-axis) hold to 1.5e-7. |
| `18_pig_gaussian_orbits.py` | One anisotropic Gaussian generating its full symmetry orbit, with covariance transformed correctly. |
| `19_pig_orbit_gradient_norms.py` | How orbit normalization changes gradient scale (unnormalized orbits inflate it 7.9x). |
| `40_pig_one_axis_physical_gate.py` | One-axis symmetry helps only when the electrodes actually have that mirror. |
| `42_physical_symmetry_design_spaces.py` | What the one-axis and two-axis design spaces look like. |
| `43_toggleable_shape_symmetry_electrodes.py` | The symmetry toggle on the shape network controls which electrodes move together. |

## PIG solver recipe

| Script | What it shows |
|---|---|
| `29_pig_boundary_mode_diagnostic.py` | Why a6 drifts while a2 stays put: every r^m cos(m theta) term is harmonic, so only boundary data can fix the m=6 share, and plain voltage MSE barely sees that error. |
| `32_pig_adam_lbfgs_handoff.py` | Adam followed by full-batch L-BFGS is the first schedule to pass every field and harmonic gate. |
| `33_pig_lbfgs_three_radius_gate.py` | The same schedule gets the sign of a6/a2 right at R/r0 = 1.00, 1.145 and 1.25 with V and E errors below 0.1%. |
| `34_pig_lbfgs_seed_gate.py` | The schedule is stable across seeds. |
| `39_geometry_conditioned_pig.py` | Conditioning PIG on geometry restores the shape-derivative path that a per-geometry model lacks. |

The parity-polynomial conditioned PIG (plots 45, 46 and 48 in the logs) passed
its gates, but its scripts were never saved. They need to be rewritten before
that result can be reproduced.

## PIXEL and alternative physics losses

| Script | What it shows |
|---|---|
| `49_pixel_kernel_derivative_continuity.py` | The cosine PIXEL kernel is C1, not C2, so its Laplacian jumps at cell edges. |
| `50_pixel_c2_physical_gate.py` | A quintic C2 kernel improves a fixed-field run; now an option in `spinn/models/backends/pixel.py`. |
| `51_pixel_mean_value_physics.py` | A mean-value (derivative-free) physics loss carries real signal but should not replace the Laplacian loss. |
| `53_pixel_mixed_physics_loss.py` | Mixing it in at lambda = 0.1 is the conservative PIXEL setting. |
| `55_pig_mixed_physics_seeds.py` | The mixed loss improves PIG fixed-field accuracy over three seeds; the harmonic cleanliness gain is not consistent. |

## Objective audits

| Script | What it shows |
|---|---|
| `47_objective_physics_audit.py` | Maximizing eta alone rewards a physically worse trap. |
| `59_single_loss_shape_objective_cheat.py` | One weighted field-plus-shape loss lets the field network satisfy the design target by getting the field wrong. |
