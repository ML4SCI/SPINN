# Joint Shape Ablation Summary

This 300-step single-seed ablation keeps the central quadrupole-shape loss enabled, reduces the single-point eta objective weight, and selects checkpoints by best FEM eta subject to `min detJ > 0.2`.

## Ablation Settings

| Run | Jacobian weight | Displacement weight | Shape LR | Purpose |
|---|---:|---:|---:|---|
| A_safe | 500 | 10 | 2e-5 | Current safe baseline |
| B_more_displacement | 500 | 1 | 2e-5 | Allow more deformation |
| C_weaker_jacobian | 100 | 10 | 2e-5 | Weaker Jacobian penalty |
| D_moderate | 100 | 1 | 2e-5 | Moderate freedom |
| E_more_shape_lr | 500 | 1 | 5e-5 | More shape movement |

## Results

| Run | Backend | Final eta_PINN | Final eta_FEM | Best valid eta_FEM | Best valid step | Min detJ | eta mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| A_safe | MLP | 0.3585 | 0.9296 | 0.9296 | 0 | 0.9956 | 0.9279 |
| A_safe | PIXEL | 0.9998 | 0.9296 | 0.9296 | 0 | 1.0000 | 0.7962 |
| A_safe | PIG | 1.0264 | 0.9296 | 0.9296 | 0 | 0.9999 | 0.9387 |
| B_more_displacement | MLP | 0.3587 | 0.9296 | 0.9296 | 0 | 0.9956 | 0.9279 |
| B_more_displacement | PIXEL | 1.0005 | 0.9296 | 0.9296 | 0 | 1.0000 | 0.7962 |
| B_more_displacement | PIG | 1.0265 | 0.9296 | 0.9296 | 0 | 0.9999 | 0.9387 |
| C_weaker_jacobian | MLP | 0.3585 | 0.9296 | 0.9296 | 0 | 0.9956 | 0.9279 |
| C_weaker_jacobian | PIXEL | 0.9998 | 0.9296 | 0.9296 | 0 | 1.0000 | 0.7962 |
| C_weaker_jacobian | PIG | 1.0264 | 0.9296 | 0.9296 | 0 | 0.9999 | 0.9387 |
| D_moderate | MLP | 0.3587 | 0.9296 | 0.9296 | 0 | 0.9956 | 0.9279 |
| D_moderate | PIXEL | 1.0005 | 0.9296 | 0.9296 | 0 | 1.0000 | 0.7962 |
| D_moderate | PIG | 1.0265 | 0.9296 | 0.9296 | 0 | 0.9999 | 0.9387 |
| E_more_shape_lr | MLP | 0.2642 | 0.9423 | 0.9423 | 299 | 0.9554 | 0.6781 |
| E_more_shape_lr | PIXEL | 1.0012 | 0.9296 | 0.9296 | 299 | 0.9999 | 0.0716 |
| E_more_shape_lr | PIG | 1.0298 | 0.9296 | 0.9296 | 0 | 0.9997 | 0.9387 |

## Interpretation

The central quadrupole-shape loss successfully suppresses the previous objective hallucination. PIXEL and PIG no longer drive eta_PINN to very large values in this short grid; they stay near eta_PINN around 1 while maintaining positive Jacobians.

Most ablations are still too conservative to move the validated geometry. Runs A-D remain at the initial FEM eta near `0.9296`.

The one useful validated improvement is `E_more_shape_lr` with MLP: eta_FEM rises from `0.9296` to `0.9423` while `min detJ = 0.9554`, so the geometry remains valid. This suggests the next full run should keep the quadrupole-shape loss and use the higher shape-network learning rate, while continuing to select checkpoints by best valid FEM eta.

## Slide-Safe Conclusion

Adding the central quadrupole-shape objective fixes the most obvious PINN-eta hallucination: the learned backends no longer produce huge internal eta values while FEM remains low. The short ablation does not yet show a PIXEL/PIG advantage, but it identifies a safer optimization regime. The best short-run validated result is MLP with higher shape-network learning rate, which improves FEM eta to `0.9423` without folding.
