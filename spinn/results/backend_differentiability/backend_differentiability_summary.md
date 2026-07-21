# Backend Differentiability Diagnostic

This diagnostic checks whether each physics backend is differentiable with respect to input coordinates `x`, and whether that signal can propagate through the composed map:

```text
z -> NN_phi(z) -> V_theta(x)
```

The code now computes eta through the composed field instead of calling the backend directly at the raw origin, so `L_eta` is no longer structurally disconnected from the shape network. The central quadrupole loss remains the main shape objective.

## Results at Initialization

| Backend | mean `||dV/dx||` | mean second-derivative norm | nonzero `dV/dx` fraction | `||grad_phi mean(V)||` | `||grad_phi L_eta||` | `||grad_phi L_quad||` |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 1.639e-1 | 5.418e-2 | 1.0 | 4.181e-4 | 0.000e0 | 6.874e-1 |
| PIXEL | 4.421e-5 | 1.994e-4 | 1.0 | 2.852e-8 | 2.970e-6 | 9.411e-2 |
| PIG | 8.281e-3 | 2.021e-2 | 1.0 | 2.669e-5 | 3.030e-4 | 1.481e-1 |

## Interpretation

PIXEL and PIG are differentiable with respect to input coordinates; the gradients are not completely broken or detached. However, PIXEL's input-coordinate gradient is extremely small compared with MLP, and its gradient from backend output into shape parameters is also tiny.

PIG has stronger input derivatives than PIXEL, but it is still weaker than MLP in the direct backend-to-shape signal.

The most likely issue is therefore not a hard detach, but gradient scale and coupling strength. PIXEL in particular has very small `dV/dx` at initialization, so shape updates receive little useful signal even when the geometry constraints are loosened.

## Slide-Safe Conclusion

PIXEL/PIG are technically differentiable in input coordinates, but their coordinate-gradient signal into `NN_phi` is much weaker than MLP. The next backend-specific fix should improve gradient scale and coupling, for example by increasing backend capacity, adding coordinate passthrough features to the decoder, or pretraining PIXEL/PIG to a stronger fixed-geometry solve before joint shape optimization.
