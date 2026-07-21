# Paul-Trap Design Benchmark: PINN vs. Classical FEM

**Task:** maximize quadrupole efficiency at electrode distance `0.125` cm (interval `[0.11, 0.145]`).
**Reference (ground truth):** `devsim_fem`.

## Accuracy and cost

| Method | Available | Objective J | d(log J)/dd | Grad rel. err vs ref | Sign agrees | Setup s | Eval s | Deterministic |
|---|---|---|---|---|---|---|---|---|
| `devsim_fem` (ref) | yes | 0.9551 | -1.507 | reference | reference | 0.0 | 0.00 | yes |
| `coordinate_projection_pinn_seed0` | yes | 1.146 | 22.37 | 1584.26% | NO | 80.7 | 0.29 | no |
| `coordinate_projection_pinn_seed1` | yes | 1.161 | 2.215 | 246.93% | NO | 80.9 | 0.00 | no |
| `coordinate_projection_pinn_seed2` | yes | 1.378 | 21.68 | 1538.59% | NO | 79.9 | 0.00 | no |

## Verdict

- `coordinate_projection_pinn_seed0`: shape-gradient relative error 1584.26% vs FEM; sign DISAGREES; end-to-end cost 81.0s vs FEM 0.0s. => gradient NOT trustworthy, and not cheaper end-to-end.
- `coordinate_projection_pinn_seed1`: shape-gradient relative error 246.93% vs FEM; sign DISAGREES; end-to-end cost 80.9s vs FEM 0.0s. => gradient NOT trustworthy, and not cheaper end-to-end.
- `coordinate_projection_pinn_seed2`: shape-gradient relative error 1538.59% vs FEM; sign DISAGREES; end-to-end cost 79.9s vs FEM 0.0s. => gradient NOT trustworthy, and not cheaper end-to-end.

## Notes per method

- `devsim_fem`: finite-element ground truth; five-point finite-difference gradient
- `coordinate_projection_pinn_seed0`: loaded cached PINN; autograd gradient through the residual coordinate map
- `coordinate_projection_pinn_seed1`: trained PINN; autograd gradient through the residual coordinate map
- `coordinate_projection_pinn_seed2`: trained PINN; autograd gradient through the residual coordinate map
