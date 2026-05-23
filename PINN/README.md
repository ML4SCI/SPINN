# Physics-Informed Neural Networks for Paul Trap Design

A PINN-based companion to the [DEVSIM](../DEVSIM/) finite-element baseline. The PINN solves the same Laplace boundary-value problem the DEVSIM code does, but the solution is a continuous differentiable function of position (and, eventually, of electrode shape parameters). This makes the PINN suitable as the inner loop of a **shape-optimization** workflow — gradients on shape parameters flow naturally through the PINN's autograd graph and through the trap-quality objective.

## Approach

For each geometry we train one PINN per **DC base function** plus one PINN for the **RF amplitude**. Each PINN learns one scalar potential `φ(x, y, z)` that satisfies Laplace's equation `∇²φ = 0` in the air region with Dirichlet boundary conditions specific to that base function. The pseudopotential at evaluation time is

```
ψ_RF(x) = e² / (4 M Ω²) · |∇φ_RF(x)|²
Ψ(x)    = Σ_i U_i · φ_DC_i(x) + ψ_RF(x)
```

with `∇φ_RF` computed by autograd through the network. Because Poisson's equation is linear, the DC base-function PINNs and the RF PINN are trained **independently** and superposed at evaluation, mirroring the base-function pattern already in the DEVSIM code.

## Key design choices

- **SIREN** (`sin` activations + Sitzmann initialization) instead of plain MLP/GELU. Vanilla MLPs have a strong spectral bias toward low frequencies; SIREN resolves sharp boundary-layer field structure with far fewer parameters.
- **Hard boundary conditions** via Shepard-style factorization:
  ```
  φ(x) = V_RF · d_B(x) / (d_A(x) + d_B(x))   +   d_A(x) · d_B(x) / (d_A(x) + d_B(x)) · NN(x)
  ```
  where `d_A`, `d_B` are minimum distances to the RF and ground boundary point clouds. The first term is a smooth lift that satisfies the BCs exactly; the second is a "bump" that vanishes on every boundary and absorbs any NN output. The PINN only has to learn the bulk Laplace residual — the BCs are exact for any NN weights.
- **Normalized scale**: the PINN solves for `φ / V_RF` (BCs of 1 and 0). Multiplied by `V_RF` at evaluation. Keeps the residual loss at O(1/L²) instead of O(V_RF²/L²).
- **DEVSIM as ground truth**: validation script compares PINN output to the DEVSIM `paul_trap_basis_RF.dat` Tecplot at common node coordinates.

## Layout

```
PINN/
  Classical_3D_Paul/        # Phase 0: validate the workflow here
    geometry.py             # trap parameters (r0, z0, world size, V_RF)
    boundaries.py           # sample boundary point clouds + air collocation points
    model.py                # SIREN + hard-BC wrapper
    train_rf.py             # train the RF-only PINN
    validate_rf.py          # compare against DEVSIM
```

## Phase plan

0. **Phase 0** — RF-only PINN for Classical 3D Paul, validate against DEVSIM. *(in progress)*
1. **Phase 1** — extend to multiple DC base functions; verify superposition matches DEVSIM.
2. **Phase 2** — wire up trap-quality objective (depth, isotropy, anharmonicity, controllability, existence penalty).
3. **Phase 3** — explicit shape optimization (small parameter vector); recover ideal Paul ratio `r₀² = 2z₀²` from a generic starting shape.
4. **Phase 4** — neural-diffeomorphism shape parameterization (isotopy/homology preserved by construction).

## Running

```bash
cd PINN/Classical_3D_Paul
python -m PINN.Classical_3D_Paul.train_rf    --steps 5000 --batch_size 512 --lr 1e-3
python -m PINN.Classical_3D_Paul.validate_rf --n_eval 20000
```

Trained parameters land in `pinn_rf_params.pkl`; the comparison plot in `pinn_vs_devsim.png`.
