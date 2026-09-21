# Shape-Coupling Diagnostic Summary

This diagnostic keeps the stable constrained setup and asks whether the physics backend sends a useful shape-optimization signal into `NN_phi`.

## Settings

- MLP control: `lr_phi = 5e-5`, `displacement = 1.0`
- PIXEL/PIG diagnostic: `lr_phi = 1e-4`, `displacement = 0.3`
- Central quadrupole loss kept on
- `jacobian = 500`
- `eta = 0.005`
- `quadrupole = 10`
- valid checkpoint threshold: `min detJ > 0.2`
- run length: 600 steps

## Results

| Backend | Best valid step | Best eta_FEM | Final eta_FEM | Final eta_PINN | Final `||grad_phi L_eta||` | Final `||grad_phi L_quad||` | Final mean displacement | Final max displacement | Final min detJ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 250 | 0.9497 | 0.9497 | 1.0835 | 0.0000 | 0.0095 | 0.0208 | 0.0429 | 0.9007 |
| PIXEL | 599 | 0.9296 | 0.9296 | 0.9997 | 0.0000 | 0.000001 | 0.0013 | 0.0038 | 0.9972 |
| PIG | 599 | 0.9296 | 0.9296 | 1.0052 | 0.0000 | 0.000030 | 0.0007 | 0.0022 | 0.9984 |

## Interpretation

The single-point eta objective provides no direct gradient into the shape network:

```text
||grad_phi L_eta|| = 0
```

for MLP, PIXEL, and PIG. This happens because `L_eta` is currently computed directly from the backend curvature at the origin, not through the coordinate map `NN_phi`.

The useful shape signal is coming from the central quadrupole loss and the physics/boundary terms. Under the same stable constraints, MLP receives a much stronger quadrupole gradient into `NN_phi` and develops a visibly larger deformation. PIXEL and PIG receive much smaller quadrupole gradients and barely deform, even after increasing their shape-network learning rate and lowering the displacement penalty.
