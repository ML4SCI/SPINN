# Trap-quality metrics across PINN models

V_RF = 300 V, Ca-40 ion, f_RF = 10.2 MHz, trap depth measured at r = 0.05 cm.

| Model | Architecture | r0 (cm) | z0 (cm) | Paul ratio | Isotropy | Secular freqs (MHz) | Depth (eV) | Locus offset (µm) | Stable |
|---|---|---:|---:|---:|---:|---|---:|---:|:---:|
| `pinn_rf_seeded_params.pkl` | Seeded-holes discovery (deformed torus + blobs) | nan | nan | nan | 0.247 | 2.91, 2.95, 5.85 | 60.58 | 0.0 | ✓ |
| `pinn_vanilla_50k.pkl` | Vanilla (tanh + soft BC) | 0.0992 | 0.0707 | 1.969 | 0.250 | 1.27, 1.29, 2.53 | 13.38 | 3.2 | ✓ |

*Paul ratio = r0²/z0² — ideal Paul trap is 2.0.*
*Isotropy = min(Hessian eigenvalue) / max(Hessian eigenvalue). Ideal Paul = 0.25 (axial freq = √2 × radial).*
