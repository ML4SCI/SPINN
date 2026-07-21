# E_more_shape_lr 1000-Step Summary

This run keeps the central quadrupole-shape loss enabled and uses the `E_more_shape_lr` ablation setting:

- `jacobian = 500`
- `displacement = 1`
- `lr_phi = 5e-5`
- `eta = 0.005`
- `quadrupole = 10`
- valid checkpoint threshold: `min detJ > 0.2`

## Result Table

| Backend | Best valid step | Best eta_FEM | eta_PINN at best | min detJ at best | Final eta_FEM | Final eta_PINN | Final min detJ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 500 | 1.0436 | 0.8600 | 0.8538 | 1.0268 | 1.1036 | 0.8554 |
| PIXEL | 999 | 0.9296 | 1.0003 | 0.9991 | 0.9296 | 1.0003 | 0.9991 |
| PIG | 0 | 0.9296 | -0.0091 | 1.0000 | 0.9296 | 0.9962 | 0.9998 |

## Interpretation

This is the first joint run in this sequence with a clear FEM-validated improvement above the initial circular trap value. The MLP backend reaches best valid `eta_FEM = 1.0436` at step 500 while staying safely non-folded with `min detJ = 0.8538`.

PIXEL and PIG remain stable and no longer hallucinate very large eta values, but they do not move the FEM-validated geometry in this setting. Their best valid FEM eta remains the initial value near `0.9296`.

## Slide-Safe Conclusion

The central quadrupole-shape loss plus higher shape-network learning rate produces a valid MLP shape improvement. The result supports the idea that objective alignment was the bottleneck, but it does not yet show a PIXEL or PIG advantage in joint shape optimization.
