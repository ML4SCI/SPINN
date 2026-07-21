# SPINN Experiment Results Summary

## Experiment setup
- Geometry used: circular_four_rod
- Backends compared: coordinate projection + MLP, coordinate projection + PIXEL, coordinate projection + PIG
- Number of collocation points: interior=2000, electrode boundary per electrode=200, outer boundary=600
- Boundary conditions: +V0/2 on x-axis RF electrodes, -V0/2 on y-axis RF electrodes, 0 on the outer boundary
- Optimizer settings: adam, lr_theta=0.001, lr_phi=0.0002
- Geometry safeguards: jacobian_weight=10.0, displacement_weight=missing, valid_checkpoint_min_detJ>0.05
- Training steps/epochs represented in saved data: 999
- FEM/reference validation method: finite-difference reference via `fem_eta`, grid=120

## Main hypothesis
Better physics representations should give more accurate field derivatives, which should produce better coordinate-projection shape optimization and higher FEM-validated eta.

## Key metrics table
| Backend | eta_PINN_final | eta_FEM_final | best_valid_eta_FEM | best_valid_step | eta_error | PDE_residual_mean | BC_error_mean | min_detJ | min_gap | max_curvature | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 4.1735 | 1.0139 | 1.0139 | 999 | 3.1596 | 0.5244 | 0.0001 | 0.7506 | 0.7310 | 1.7329 | 178.7 |
| PIXEL | 44.2876 | 1.0210 | 1.0561 | 250 | 43.2666 | 1713298738258116608.0000 | 0.0069 | -1.3548 | 0.7104 | 3.0364 | 232.6 |
| PIG | 18.4101 | 0.4765 | 0.9296 | 0 | 17.9336 | 8.7191 | 0.0135 | 0.5382 | 1.3037 | 0.8791 | 721.4 |

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
