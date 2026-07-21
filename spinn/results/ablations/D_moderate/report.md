# SPINN Experiment Results Summary

## Experiment setup
- Geometry used: circular_four_rod
- Backends compared: coordinate projection + MLP, coordinate projection + PIXEL, coordinate projection + PIG
- Number of collocation points: interior=2048, electrode boundary per electrode=256, outer boundary=512
- Boundary conditions: +V0/2 on x-axis RF electrodes, -V0/2 on y-axis RF electrodes, 0 on the outer boundary
- Optimizer settings: adam, lr_theta=0.001, lr_phi=2e-05
- Geometry/objective safeguards: jacobian_weight=100.0, displacement_weight=1.0, quadrupole_weight=10.0, valid_checkpoint_min_detJ>0.2
- Training steps/epochs represented in saved data: 299
- FEM/reference validation method: finite-difference reference via `fem_eta`, grid=120

## Main hypothesis
Better physics representations should give more accurate field derivatives, which should produce better coordinate-projection shape optimization and higher FEM-validated eta.

## Key metrics table
| Backend | eta_PINN_final | eta_FEM_final | best_valid_eta_FEM | best_valid_step | eta_error | PDE_residual_mean | BC_error_mean | min_detJ | min_gap | max_curvature | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.3587 | 0.9296 | 0.9296 | 0 | 0.5709 | 0.0083 | 0.1871 | 0.9956 | 0.7422 | 0.8764 | 54.2 |
| PIXEL | 1.0005 | 0.9296 | 0.9296 | 0 | 0.0709 | 0.0504 | 0.0000 | 1.0000 | 0.7435 | 0.8741 | 64.4 |
| PIG | 1.0265 | 0.9296 | 0.9296 | 0 | 0.0969 | 0.1273 | 0.0004 | 0.9999 | 0.7435 | 0.8741 | 200.1 |

## Plot checklist
- fixed_geometry_potential_fields.png: missing
- fixed_geometry_error_heatmaps.png: missing
- fixed_geometry_pde_residual_maps.png: missing
- fixed_geometry_eta_error_bar.png: missing
- eta_pinn_vs_epoch.png: missing
- eta_fem_vs_checkpoint.png: missing
- pinn_eta_vs_fem_eta_scatter.png: missing
- initial_vs_optimized_electrodes.png: missing
- deformation_field.png: missing
- jacobian_determinant_heatmap.png: missing
- loss_terms_vs_epoch.png: missing
- constraints_vs_epoch.png: missing
- pixel_feature_activity.png: missing
- pig_gaussian_centers.png: missing
- pig_gaussian_covariances.png: missing

## Warnings and missing data
- Fixed-geometry benchmark metrics are missing at `spinn/results/fixed/metrics.csv`; fixed-geometry potential/error/residual/eta-error plots were not generated.
- Saved joint comparison has 1 seed(s), not the planned 3-seed validation.
- Saved joint comparison is a short smoke-scale run; do not treat it as the final 20k-60k epoch experiment.
- PDE_residual_mean and BC_error_mean in the table are logged loss proxies from the final training row, not a full post-hoc statistical residual audit.

## Conclusion
All similar: the current FEM eta values are too close to identify the backend as the bottleneck.

The conclusion is based on the best valid FEM/reference checkpoint where available, not final PINN-predicted eta.
