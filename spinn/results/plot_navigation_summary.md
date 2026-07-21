# SPINN Plot Navigation Summary

Use this as a quick guide while moving through the result slides. The current saved run is a constrained 1000-step, single-seed diagnostic comparison, so the safest language is still diagnostic rather than final.

## Overall Comparison Plots

### `eta_pinn_vs_epoch.png`

Shows the PINN-predicted geometric efficiency during joint training.

Say: all three backends overpredict eta internally. MLP rises to `2.596`, PIXEL to `22.935`, and PIG to `21.972`, so PINN eta alone is not a trustworthy design metric.

### `eta_fem_vs_checkpoint.png`

Shows FEM/reference-validated eta at saved checkpoints.

Say: this is the headline comparison. With stronger geometry safeguards, all three runs stay non-folded, but none produces a validated improvement over the starting FEM eta near `0.9296`. PIXEL and PIG actually drop to about `0.8672` by the final checkpoint.

### `pinn_eta_vs_fem_eta_scatter.png`

Shows whether PINN-predicted eta agrees with FEM/reference eta.

Say: points near the diagonal are trustworthy. PIXEL and PIG move far to the right while staying low on FEM eta, which shows objective hallucination clearly: the backend predicts high eta, but the validated geometry does not improve.

### `initial_vs_optimized_electrodes.png`

Compares starting circular electrodes with final optimized electrodes for MLP, PIXEL, and PIG.

Say: this plot makes the coordinate projection network visible as an electrode-shape optimizer. In this constrained run the deformations are conservative, which is consistent with the positive Jacobians.

### `deformation_field.png`

Shows the learned coordinate displacement field for each backend.

Say: this visualizes the map from reference coordinates to physical coordinates. It connects the equation `x = NN_phi(z)` to the actual displacement pattern.

### `jacobian_determinant_heatmap.png`

Shows the determinant of the coordinate-map Jacobian.

Say: positive determinant means the map is not folding or collapsing. The new constraints worked: final minimum detJ is about `0.999` for MLP, `0.951` for PIXEL, and `0.948` for PIG.

### `loss_terms_vs_epoch.png`

Shows PDE loss, boundary-condition loss, eta objective loss, displacement loss, and geometry-constraint losses.

Say: this plot helps diagnose whether optimization progress comes from solving the physics or from exploiting the objective. The displacement term is now included so we can see how much the coordinate map is allowed to move.

### `constraints_vs_epoch.png`

Tracks minimum determinant, minimum gap, maximum curvature, and FEM/reference eta.

Say: this is the "did the optimizer cheat?" plot. In this run the geometry remains valid, but FEM eta does not improve, so the problem is no longer folding; it is objective trustworthiness and loss balancing.

## MLP Plots

### `mlp_seed0/potential_mlp.png`

Shows the learned RF potential field for the vanilla MLP physics backend.

Say: this is the smooth global PINN baseline.

### `mlp_seed0/residual_mlp.png`

Shows where Laplace's equation is least satisfied.

Say: lower residual means a cleaner physics solve. MLP is the most conservative joint optimizer in this constrained run.

### `mlp_seed0/contours_mlp.png`

Shows initial versus optimized electrode contours for coordinate projection + MLP.

Say: this is the geometry produced by the baseline physics solver.

### `mlp_seed0/deformation_mlp.png`

Shows the coordinate displacement field learned with the MLP backend.

Say: MLP barely moves the geometry under the stronger constraints.

### `mlp_seed0/detJ_mlp.png`

Shows the Jacobian determinant for the MLP-backed coordinate map.

Say: the map stays very close to identity. The final minimum detJ is about `0.9986`.

### MLP Takeaway

MLP is the safest constrained result. FEM/reference eta stays at about `0.9296`, while PINN eta rises to `2.596`, so even the conservative baseline overpredicts internally.

## PIXEL Plots

### `pixel_seed0/potential_pixel.png`

Shows the learned RF potential from the fixed-grid local feature backend.

Say: this is the PIXEL version of the physics solver.

### `pixel_seed0/residual_pixel.png`

Shows the PDE residual for PIXEL.

Say: residual alone does not guarantee trustworthy eta; the validated eta can still move in the wrong direction.

### `pixel_seed0/contours_pixel.png`

Shows initial versus optimized electrode contours for coordinate projection + PIXEL.

Say: unlike the previous unconstrained run, PIXEL no longer folds the geometry.

### `pixel_seed0/deformation_pixel.png`

Shows the coordinate displacement field learned with PIXEL.

Say: PIXEL changes the geometry more than MLP, but the validated FEM result gets worse.

### `pixel_seed0/detJ_pixel.png`

Shows the Jacobian determinant for the PIXEL-backed coordinate map.

Say: the Jacobian constraint worked. The final minimum detJ is about `0.9512`, so the final geometry is valid.

### `pixel_seed0/pixel_feature_activity.png`

Shows the norm of the learned grid features.

Say: this is the PIXEL-specific diagnostic. It shows where the fixed local feature grid is using capacity.

### PIXEL Takeaway

PIXEL is now a clean objective-hallucination case rather than an invalid-folding case. Final PINN eta is `22.935`, but final FEM/reference eta is only `0.8672`; the best valid FEM checkpoint is basically the starting geometry at `0.9296`.

## PIG Plots

### `pig_seed0/potential_pig.png`

Shows the learned RF potential from adaptive Gaussian features.

Say: this is the PIG version of the physics solver.

### `pig_seed0/residual_pig.png`

Shows the PDE residual for PIG.

Say: PIG can produce strong internal eta signals without validated geometry improvement, so the FEM checkpoints are essential.

### `pig_seed0/contours_pig.png`

Shows initial versus optimized electrode contours for coordinate projection + PIG.

Say: this is the shape produced by the adaptive Gaussian backend.

### `pig_seed0/deformation_pig.png`

Shows the coordinate displacement field learned with PIG.

Say: PIG moves the geometry more than MLP while remaining topology-safe.

### `pig_seed0/detJ_pig.png`

Shows the Jacobian determinant for the PIG-backed coordinate map.

Say: the geometry remains valid. The final minimum detJ is about `0.9481`.

### `pig_seed0/pig_gaussian_centers.png`

Shows where the adaptive Gaussian centers ended up.

Say: this is the main PIG-specific visual. Ideally, centers should cluster near hard regions such as electrode edges and gaps.

### `pig_seed0/pig_gaussian_covariances.png`

Shows Gaussian covariance ellipses.

Say: this indicates whether PIG is learning local adaptive resolution or spreading capacity broadly.

### PIG Takeaway

PIG also stays non-folded after the constraint changes, but it does not win on FEM. Final PINN eta is `21.972`, final FEM/reference eta is `0.8672`, and the best valid FEM checkpoint is essentially the starting geometry at `0.9296`.

## Safe Final Slide Conclusion

The stronger Jacobian penalty, lower shape learning rate, and displacement regularization fixed the folding problem. However, the constrained 1000-step run still does not show validated shape improvement: PIXEL and PIG produce high PINN eta but lower FEM eta. The next experiment should focus on making the eta objective more trustworthy, for example by improving backend field accuracy near the center, reducing eta-objective weight, or selecting checkpoints only by FEM eta.

## Follow-Up Ablation: Central Quadrupole Loss

### What changed

The next experiment added a central quadrupole-shape loss, reduced the single-point eta objective weight from `0.05` to `0.005`, and selected valid checkpoints using `min detJ > 0.2`.

Say: this directly tests whether the previous failure was caused by optimizing a single curvature derivative at the center.

### `spinn/results/ablations/ablation_summary.md`

Summarizes the five 300-step ablation settings:

- A_safe: `jacobian=500`, `displacement=10`, `lr_phi=2e-5`
- B_more_displacement: `jacobian=500`, `displacement=1`, `lr_phi=2e-5`
- C_weaker_jacobian: `jacobian=100`, `displacement=10`, `lr_phi=2e-5`
- D_moderate: `jacobian=100`, `displacement=1`, `lr_phi=2e-5`
- E_more_shape_lr: `jacobian=500`, `displacement=1`, `lr_phi=5e-5`

Say: the central quadrupole-shape loss suppresses the eta hallucination. PIXEL and PIG stay near eta_PINN around 1 instead of exploding to 20-40.

### Ablation Takeaway

The best short-run validated result is `E_more_shape_lr` with MLP: FEM eta improves from `0.9296` to `0.9423` while `min detJ = 0.9554`. PIXEL and PIG remain valid and no longer hallucinate large eta, but they do not improve FEM eta in the 300-step grid.

Safe wording: the central quadrupole loss improves objective trustworthiness, but a longer run or better backend tuning is still needed before claiming PIXEL/PIG shape-optimization gains.

## Full Follow-Up: `E_more_shape_lr` 1000 Steps

### `spinn/results/e_more_shape_lr_full/E_more_shape_lr/e_more_shape_lr_1000_summary.md`

Shows the longer run using the best short-ablation setting:

- `jacobian=500`
- `displacement=1`
- `lr_phi=5e-5`
- central quadrupole loss kept on
- checkpoint selection by best valid FEM eta with `min detJ > 0.2`

Say: this is the first run in this sequence with a clear FEM-validated shape improvement.

### Headline result

MLP reaches best valid `eta_FEM = 1.0436` at step 500, with `min detJ = 0.8538`. The final MLP checkpoint remains valid but drops slightly to `eta_FEM = 1.0268`, so best-checkpoint selection matters.

PIXEL and PIG remain stable and no longer hallucinate large eta, but they do not improve FEM eta in this setting. Both stay near the initial FEM value `0.9296`.

Safe wording: the central quadrupole-shape loss and higher shape-network learning rate can produce a validated MLP geometry improvement. The current evidence still does not support a PIXEL/PIG advantage for joint shape optimization.

## Shape-Coupling Diagnostic

### `spinn/results/shape_coupling_diagnostic/shape_coupling_summary.md`

Tests whether PIXEL/PIG fail because they do not send useful geometry gradients into `NN_phi`.

Say: the diagnostic logs `||grad_phi L_eta||`, `||grad_phi L_quad||`, mean deformation, and max deformation.

### `shape_coupling_vs_epoch.png`

Shows shape-gradient coupling and deformation over time.

Say: in the pre-fix diagnostic, `||grad_phi L_eta||` was zero because the single-point eta loss was computed directly from backend curvature at the origin, not through the coordinate map. The code has since been changed so eta is evaluated through `V_theta(NN_phi(z))`.

### Diagnostic Takeaway

MLP moves the shape and improves FEM eta to `0.9497`, with final mean deformation `0.0208`. PIXEL and PIG remain almost identity maps despite higher `lr_phi` and lower displacement penalty: PIXEL final mean deformation is `0.0013`, and PIG final mean deformation is `0.0007`.

Safe wording: the current PIXEL/PIG bottleneck is weak backend-to-shape coupling, not folding or FEM validation instability.

## Backend Differentiability Diagnostic

### `spinn/results/backend_differentiability/backend_differentiability_summary.md`

Checks whether MLP, PIXEL, and PIG are differentiable with respect to input coordinates `x`, and whether backend gradients can flow through `x = NN_phi(z)`.

Say: PIXEL and PIG are technically differentiable; their gradients are not detached. The problem is gradient scale. PIXEL has mean `||dV/dx|| = 4.42e-5`, while MLP has `1.64e-1`. PIG is in between at `8.28e-3`.

### Diagnostic Takeaway

The PIXEL/PIG problem is not hard non-differentiability. It is weak coordinate-gradient coupling into the shape network, especially for PIXEL. The next backend-specific fix should strengthen gradient scale, likely through better fixed-geometry pretraining, larger backend capacity, or adding coordinate passthrough features to the decoder.
