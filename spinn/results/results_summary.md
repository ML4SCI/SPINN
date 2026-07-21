# SPINN Experiment Results Summary

## Experiment setup
- Geometry used: circular_four_rod
- Backends compared: coordinate projection + MLP, coordinate projection + PIXEL, coordinate projection + PIG
- Number of collocation points: interior=2048, electrode boundary per electrode=256, outer boundary=512
- Boundary conditions: +V0/2 on x-axis RF electrodes, -V0/2 on y-axis RF electrodes, 0 on the outer boundary
- Optimizer settings: adam, lr_theta=0.001, lr_phi=2e-05
- Geometry safeguards: jacobian_weight=500.0, displacement_weight=10.0, valid_checkpoint_min_detJ>0.05
- Training steps/epochs represented in saved data: 999
- FEM/reference validation method: finite-difference reference via `fem_eta`, grid=120

## Main hypothesis
Better physics representations should give more accurate field derivatives, which should produce better coordinate-projection shape optimization and higher FEM-validated eta.

## Key metrics table
| Backend | eta_PINN_final | eta_FEM_final | best_valid_eta_FEM | best_valid_step | eta_error | PDE_residual_mean | BC_error_mean | min_detJ | min_gap | max_curvature | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 2.5961 | 0.9296 | 0.9296 | 0 | 1.6665 | 0.1355 | 0.0001 | 0.9986 | 0.7431 | 0.8754 | 168.4 |
| PIXEL | 22.9353 | 0.8672 | 0.9296 | 250 | 22.0681 | 77.1545 | 0.0088 | 0.9512 | 0.7576 | 0.9050 | 197.1 |
| PIG | 21.9720 | 0.8672 | 0.9296 | 250 | 21.1048 | 89.5517 | 0.0076 | 0.9481 | 0.7597 | 0.9091 | 542.6 |

## Plot checklist
- fixed_geometry_potential_fields.png: missing
- fixed_geometry_error_heatmaps.png: missing
- fixed_geometry_pde_residual_maps.png: missing
- fixed_geometry_eta_error_bar.png: missing
- eta_pinn_vs_epoch.png: generated
- eta_fem_vs_checkpoint.png: generated
- pinn_eta_vs_fem_eta_scatter.png: generated
- initial_vs_optimized_electrodes.png: generated
- deformation_field.png: generated
- jacobian_determinant_heatmap.png: generated
- loss_terms_vs_epoch.png: generated
- constraints_vs_epoch.png: generated
- pixel_feature_activity.png: generated
- pig_gaussian_centers.png: generated
- pig_gaussian_covariances.png: generated

## Warnings and missing data
- Saved joint comparison has 1 seed(s), not the planned 3-seed validation.
- Saved joint comparison is a short smoke-scale run; do not treat it as the final 20k-60k epoch experiment.
- PDE_residual_mean and BC_error_mean in the table are logged loss proxies from the final training row, not a full post-hoc statistical residual audit.

## Conclusion
PINN eta is not validated by FEM for PIXEL: the internal eta rises far above the FEM/reference eta, so the optimizer is exploiting physics-backend error in this short run.

The conclusion is based on the best valid FEM/reference checkpoint where available, not final PINN-predicted eta.
