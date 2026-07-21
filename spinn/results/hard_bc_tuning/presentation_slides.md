# Constraint Placement in Physics-Informed Networks
## Hard vs. Soft Enforcement for Paul-Trap Field Solvers, with Consequences for the PIXEL/PIG Benchmark

---

# SECTION 1 — CONTEXT

## Slide 0 — So... Why Did We Even Look Into This? (the casual backstory)

> Speaker note: deliberately informal — this is the "how it started" slide.

- Our config file has a line that says `bc: 100.0`. That's the penalty weight telling the network "please respect the electrode voltages, times one hundred." Why 100? Nobody remembers. It just... works, mostly. Every time training got weird, this was suspect #1 — are the physics loss and the boundary loss fighting each other again?
- The thing we actually care about, $\eta$, is a *second derivative at a single point*. It's a tiny, delicate number. If the network fits the boundary voltages even slightly wrong, that error can quietly leak into the middle of the trap where we measure — and we'd have no way to tell. So the thought was: what if we made it *mathematically impossible* to get the boundary wrong? Then whatever $\eta$ we measure, at least the boundaries aren't the ones lying to us.
- Also, the literature kept nudging us. Every other PINN paper says some version of "constraints should be baked into the architecture, penalty terms are inelegant." It sounds obviously right! Exact is better than approximate, fewer loss weights is better than more. But nobody ever showed receipts for a problem shaped like ours.
- And honestly — it looked like a quick afternoon win: delete a hyperparameter, gain a guarantee, move on.
- It was not a quick afternoon win. It turned into a multi-day detective story (a fake convergence, an identical-number smoking gun, a seed lottery, and a saturated metric) that ended up rewriting how we're going to run the whole PIXEL/PIG benchmark. Easily the most productive "failed" experiment of the project — which is kind of the point of this talk.

## Slide 1 — Hypothesis and Research Question

- System: 2D four-rod linear Paul trap cross-section; dimensionless units $r_0 = 1$, $V_0 = 1$, rod radius $R_{\mathrm{rod}} = 1.145\,r_0$, rod centers at $\pm(r_0 + R_{\mathrm{rod}})$ on both axes, grounded shield $R_{\mathrm{out}} = 5\,r_0$.
- Governing physics: $\nabla^2 V = 0$ in vacuum; Dirichlet data $V = +V_0/2$ (x-rods), $-V_0/2$ (y-rods), $0$ (shield).
- Headline metric: geometric efficiency $\eta = (r_0^2/V_0)\,\partial^2 V/\partial x^2 \,\big|_{(0,0)}$; ideal hyperbolic quadrupole $\Rightarrow \eta = 1$.
- Initial hypothesis $H_A$: architectural (hard) enforcement of boundary conditions and symmetry outperforms penalty (soft) enforcement for PINN field solvers.
- Outcome preview: $H_A$ rejected for boundary conditions, retained for symmetry; two additional findings alter the design of the downstream PIXEL/PIG benchmark.

## Slide 2 — Theoretical Grounding

- Hard-BC ansatz class: Lagaris / transfinite interpolation, $V = G + D\,N$ with $G$ interpolating boundary data exactly and $D$ vanishing on all Dirichlet boundaries; constraint satisfaction independent of network weights.
- Symmetry class: field transforms in the sign character of $D_4$ — even under axis mirrors, odd under diagonal swap and $\pm 90^\circ$ rotation; enforceable via group averaging (Reynolds operator) or invariant-feature factorization.
- Validation principle: PINN-independent finite-difference solver (5-point stencil) recomputes $\eta$ from rasterized geometry; agreement PINN $\leftrightarrow$ FEM is the trust criterion.

## Slide 3 — Scope

- Fixed reference geometry only (no shape optimization executed); backend architecture: symmetrized MLP; comparisons at matched architecture, optimizer, and step budget.
- All artifacts: `spinn/scripts/tune_hard_bc.py`, `spinn/results/hard_bc_tuning/` (metrics CSVs, models, Optuna DB, logs).

---

# SECTION 2 — METHODOLOGY & TESTING

## Slide 4 — Experimental Design: Variables

- Independent variables (search phase): symmetrization $\in$ {reynolds, prefactor}; depth $\in [2,5]$; width $\in [32,192]$; activation $\in$ {tanh, sin, gelu} ($C^\infty$ required — residual contains second derivatives); Fourier features $\in \{0,16,32,64\}$ with bandwidth $\sigma \in [0.5, 4]$; lr $\in [10^{-4}, 10^{-2}]$ (log); optimizer $\in$ {Adam, AdamW}; scheduler $\in$ {cosine, none}; batch $\in \{1024, 2048, 4096\}$; center-boost fraction $\in [0, 0.5]$.
- Independent variable (A/B phase): BC mechanism only — hard (carrier ansatz) vs. soft (penalty $\lambda_{bc} = 100$); architecture/protocol otherwise identical.
- Dependent variables: validation residual MSE on a fixed held-out collocation set (4096 points, identical across all trials/runs); $\eta_{\mathrm{fit}}$ on center disk $r \le 0.25$; boundary RMSE; wall time.
- Controls: fixed torch seed across search trials (initialization noise removed from ranking); FEM reference computed independently at grids 256/384/512.

## Slide 5 — Test Protocol

- Phase 1: Optuna TPE study, 20 trials $\times$ 4{,}000 steps, MedianPruner (5 startup trials, 2 warmup reports); objective = held-out residual MSE.
- Phase 2: external validation of the phase-1 winner against FEM; diagnosis experiments (network-deleted control $V \equiv G$; least-squares carrier augmentation).
- Phase 3: budget test — winner configuration, 20{,}000 steps, $\eta$ and residual logged every 500 steps.
- Phase 4: $3 \times 2$ matrix — {hard, soft} $\times$ seeds {0, 1, 2}, 20{,}000 steps each; region-resolved $L_2$ error vs. FEM with FEM self-convergence (256 vs. 384) as the trust limit.
- Phase 5: FEM survey of candidate perturbed geometries (stage-D redesign input).

## Slide 6 — Constraint Verification (pre-training)

- Untrained-network exactness (float64): boundary error $3.0\times10^{-14}$ (reynolds), $3.7\times10^{-12}$ (prefactor); mirror symmetry error $1.7\times10^{-16}$; swap antisymmetry error $1.4\times10^{-16}$; Laplacian finite at all sampled points including the origin.
- Estimator calibration: $\eta_{\mathrm{fit}} = 1.000000000$ on the analytic quadrupole $V = \tfrac{V_0}{2r_0^2}(x^2 - y^2)$.
- Implementation note: geometry buffers must be constructed in float64 before dtype cast; float32 construction injects $\mathcal{O}(10^{-7})$ center errors that break float64 boundary exactness.

---

# SECTION 3 — MATHEMATICAL FOUNDATIONS

## Slide 7 — Hard-Constraint Ansatz

$$V(\mathbf{x}) = G(\mathbf{x}) + \hat{D}(\mathbf{x})\, N_{\mathrm{sym}}(\mathbf{x})$$

- Polynomial boundary functions (5 components: 4 rods + shield), normalized to unity at the origin:
$$\varphi_i(\mathbf{x}) = \frac{|\mathbf{x} - \mathbf{c}_i|^2 - R_i^2}{2 R_i\, \varphi_i^{(0)}}, \qquad \varphi_{\mathrm{out}}(\mathbf{x}) = \frac{R_{\mathrm{out}}^2 - |\mathbf{x}|^2}{2 R_{\mathrm{out}}\, \varphi_{\mathrm{out}}^{(0)}}$$
- Polynomial form is mandatory: the true distance $R_{\mathrm{out}} - |\mathbf{x}|$ has $\nabla^2(R - |\mathbf{x}|) = -1/|\mathbf{x}|$, singular at the origin — the $\eta$ measurement point.
- Transfinite interpolant (exact on every boundary; convex $\Rightarrow |G| \le V_0/2$):
$$G = \sum_i V_i\, w_i, \qquad w_i = \frac{\prod_{j \ne i} \varphi_j}{\sum_k \prod_{j \ne k} \varphi_j}$$
- Bounded vanishing factor: $D = \prod_i \varphi_i$, $\hat{D} = D/(1 + D) \in [0, 1)$; $\hat D = 0$ exactly on all boundaries.

## Slide 8 — Symmetrization Operators

- Reynolds projection onto the sign character of $D_4$ (order 8):
$$N_{\mathrm{sym}}(\mathbf{x}) = \frac{1}{8} \sum_{g \in D_4} \chi(g)\, N(g^{-1}\mathbf{x}), \qquad \chi = \begin{cases} +1 & g \in \{e,\ \sigma_x,\ \sigma_y,\ C_2\} \\ -1 & g \in \{C_4,\ C_4^3,\ \sigma_d,\ \sigma_{d'}\} \end{cases}$$
- Invariant-prefactor factorization (restricts spectrum to $m \equiv 2 \pmod 4$):
$$N_{\mathrm{sym}}(\mathbf{x}) = (x^2 - y^2)\; F\!\left(r^2,\ x^2 y^2,\ (x^2 - y^2)^2\right)$$
- Structural consequence of Reynolds: constant components of $N$ are annihilated ($\sum_g \chi(g) = 0$), hence $N \to \text{const}$ collapse yields $N_{\mathrm{sym}} \equiv 0$, i.e. $V \equiv G$ exactly.

## Slide 9 — Losses, Estimators, References

- Hard arm loss: $\mathcal{L} = \mathbb{E}\big[(\nabla^2 V)^2\big]$ (residual only). Soft arm: $\mathcal{L} = \mathbb{E}\big[(\nabla^2 V)^2\big] + \lambda_{bc}\, \mathbb{E}\big[(V - V_b)^2\big]$, $\lambda_{bc} = 100$.
- $\eta$ estimator: least squares of $V \approx a(x^2 - y^2) + bxy + cx + dy + e$ on $r \le 0.25$; $\eta_{\mathrm{fit}} = 2a\, r_0^2 / V_0$.
- FEM continuum reference by Richardson extrapolation, $\eta(h) = \eta_0 - C h^2$, $h \propto (N_{\mathrm{grid}}-1)^{-1}$:
$$\eta_0 = \eta_{512} + (\eta_{512} - \eta_{384}) \frac{h_{512}^2}{h_{384}^2 - h_{512}^2} = 0.99737$$
- Asymmetric-trap quantities (stage-D metric basis): Hessian $H = \begin{pmatrix} 2a & b \\ b & -2a \end{pmatrix}$; null position $\mathbf{x}_{\mathrm{null}} = -H^{-1}(c, d)^{\mathsf T}$; principal quadrupole strength $\eta_p = \sqrt{4a^2 + b^2}$.

---

# SECTION 4 — RESULTS

## Slide 10 — Finding 1: Short-Budget Search Collapse (G-Floor)

- Phase-1 study: all completed reynolds trials clustered at residual $0.178$–$0.211$; 8 of 8 reported $\eta = 0.69134$ to 5 decimals across distinct depths/widths/activations.
- Network-deleted control ($V \equiv G$): residual MSE $= 0.1977$, $\eta = 0.6911$ — identical to the trial plateau. Every 4k-step trial converged to the carrier, not to a solution.
- FEM contradiction: $\eta_{\mathrm{FEM}} = 0.958$ (256) / $0.983$ (384) vs. PINN $0.692$; full-field relative $L_2 = 26.9\%$.
- Carrier augmentation dead end: least-squares fit of 12 $\chi$-symmetric polynomial modes reduced source MSE $0.191 \to 0.156$ ($1.2\times$) — residual structure is near-electrode and not low-order.
- Prefactor arm: residuals $9.7$–$5029$, all above the G-floor; categorically inferior.
- Protocol consequence: hyperparameter rankings obtained below the plateau horizon are noise.

## Slide 11 — Finding 2: Plateau Escape Is Budget-Dependent and Seed-Fragile

- Winner configuration at 20k steps (seed 0): residual $0.198 \to 0.078$; $\eta$ flat at $\approx 0.69$ for 14{,}500 steps, then $0.69 \to 0.98$ between steps 15k–18k, coincident with cosine-schedule lr decay.
- Seed replication (hard arm, identical config): seed 1 — no escape in 20k steps, final $\eta = 0.7053$; seed 2 — escape, final $\eta = 1.0170$ with late divergence ($1.0095 \to 1.0170$ over the final 2k steps while residual improved $0.072 \to 0.053$): residual improvement and center-field accuracy decouple.
- Three outcome grades from seed alone: no escape / escape-biased / escape-accurate.

## Slide 12 — Finding 3: Soft BC + Hard Symmetry Dominates (3×2 Matrix)

| Arm | $\eta$ (seeds 0/1/2) | Seed spread | Residual MSE | Boundary RMSE |
|---|---|---|---|---|
| Hard BC | 0.9804 / 0.7053 / 1.0170 | $0.31$ | 0.053–0.147 | $6\times10^{-8}$ V |
| Soft BC | 1.00280 / 1.00276 / 1.00272 | $8\times10^{-5}$ | $1\times10^{-6}$ | $1.2$–$1.8\times10^{-5}$ V |

- Reference: $\eta_0 = 0.9974$ (Richardson; bracketing PINN value suggests true $\eta \in [0.997, 1.003]$).
- Region-resolved $L_2$ vs. FEM(384) [soft / hard / FEM self-error]: $r<1$: 2.0% / 1.3% / 2.5%; $2 \le r < 3$: 1.4% / 9.6% / 1.0%; $3 \le r < 5$: 0.8% / 34.2% / 0.5%; global: 1.2% / 28.6% / 0.8%.
- Interpretation: hard BC retains only its definitional advantage (boundary exactness, $\sim 250\times$ lower BC error); soft BC wins residual ($\sim 10^5\times$), $\eta$ accuracy, global field accuracy ($24\times$), and reproducibility ($\sim 4000\times$ tighter).
- Design principle: hard constraints on geometry/symmetry (cost-free: Reynolds trained identically in both arms); soft constraints on field boundary data.

## Slide 13 — Finding 4: Headline Metric Is Saturated on the Reference Geometry

- Unoptimized reference trap: $\eta \in [0.997, 1.003]$; theoretical ceiling $\eta \approx 1$; available headroom $\lesssim 0.005$.
- Existing victory condition (plan §7): PIG wins at $\ge 0.03$ absolute or $\ge 5\%$ relative margin — $6$–$10\times$ larger than total headroom.
- Consequence: the planned $6 \times 60\mathrm{k}$-step comparison would return $H_0$ regardless of backend quality; a null result by metric saturation, not by physics.

## Slide 14 — Finding 5: $\eta$ Rewards Damage Off-Symmetry (Perturbation FEM Survey)

| Perturbed geometry (grid 384; ref. 0.9834) | $\eta_x = 2a$ | $\|\mathbf{x}_{\mathrm{null}}\|/r_0$ | $b_{xy}$ |
|---|---|---|---|
| Repo stage-D ($\pm 5$–6%, all rods) | 1.0140 | 0.063 | $\approx 0$ |
| East $+10\%$ outward, North $+5^\circ$ | 0.8883 | 0.068 | $+0.042$ |
| East $-15\%$ inward, North $+10^\circ$ | 1.1543 | 0.165 | $+0.079$ |
| 4-rod composite, all mirrors broken | 0.9925 | 0.265 | $+0.086$ |

- Inward rod displacement raises origin curvature: $\eta_x$ increases up to $+17\%$ while the RF null displaces $0.17\,r_0$ and cross/odd harmonics grow — the metric improves as the trap degrades.
- A shape optimizer maximizing $\eta_x$ on asymmetric geometry is rewarded for pushing electrodes inward, not for restoring field quality.
- Diagnostic gap identified: `harmonic_purity` fits even orders only (valid under $D_4$); dipole $m{=}1$ and hexapole $m{=}3$ — the leading perturbation signatures — are invisible to it.

## Slide 15 — Results Synthesis

- Constraint placement, measured: exactness–trainability trade quantified; hard-BC exactness costs $\sim 10^5$ in residual and converts convergence into a seed-level Bernoulli event.
- Validation asymmetry: PINN (soft) and FEM now mutually confirm to within FEM's own discretization error everywhere in the domain.
- Two benchmark-invalidating artifacts caught pre-benchmark: metric saturation (Finding 4) and objective mis-specification off-symmetry (Finding 5); both discovered by the FEM-first workflow at negligible compute cost.

---

# SECTION 5 — FUTURE DIRECTIONS & GAPS

## Slide 16 — Open Gaps

- FEM reference order ambiguity: Richardson assumed $\mathcal{O}(h^2)$; staircase Dirichlet rasterization is formally between $\mathcal{O}(h)$ and $\mathcal{O}(h^2)$; $\eta_0$ uncertainty $\approx \pm 0.005$. Resolution: grid-768 solve or boundary-fitted solver.
- Soft-arm $\eta$ offset ($+0.005$ from $\eta_0$) unattributed: candidates — BC penalty bias vs. extrapolation-order error; a maximum-principle bound linking boundary RMSE ($1.5\times10^{-5}$ V) to $\eta$ bias is derivable and outstanding.
- A/B architecture inherited from an invalid (4k-step) search; both arms shared it, so the comparison is internally fair but the configuration is not certified optimal.
- $\lambda_{bc} = 100$ inherited untested at valid budgets; prefactor underperformance not mechanistically isolated (amplitude scaling vs. spectral restriction); SIREN untested under proper $\omega_0$ initialization (bar: residual $10^{-6}$, $\eta$ error $0.5\%$, far-field $0.8\%$).

## Slide 17 — Redesigned Benchmark (Actionable Sequence)

1. Metric layer: extend `harmonic_purity` to odd orders; add null-shift fit; define fidelity objective, e.g. $\Phi = \mathrm{purity}_{c_2}(\text{all } m) \cdot \exp(-\|\mathbf{x}_{\mathrm{null}}\|/\ell)$ with $\eta_p = \sqrt{4a^2 + b^2}$ reported; freeze rubric before any backend run.
2. Arenas: two perturbed geometries — curvature-degraded (outward, $\eta_x = 0.888$) and purity-degraded (inward/rotated, $\eta_x \ge 1$ with $\|\mathbf{x}_{\mathrm{null}}\| = 0.17\,r_0$); broken $D_4$ removes the symmetric-baseline confound structurally. Geometric safety envelope: adjacent-rod gap $0.74\,r_0$ at reference; single-rod rotation safe to $\sim 25^\circ$; maintain min-gap $\ge 0.3$–$0.4\,r_0$ during optimization.
3. Scoring: wall-time-to-fidelity-threshold, residual at convergence, $\eta_{\mathrm{PINN}}$–$\eta_{\mathrm{FEM}}$ gap, cross-seed CV, and seed failure rate; final-$\eta$ margin reported, not scored.
4. Protocol (non-negotiable): $\ge 3$ seeds, $\ge 15$–20k steps per run, FEM validation per checkpoint.
5. Backend integration: soft-BC symmetrized MLP enters as the fourth backend vs. MLP / PIXEL / PIG; shape phase adopts hard-constrained Fourier-$D_4$ boundary parameterization $r(\theta) = R_{\mathrm{rod}}\big(1 + \sum_k c_{\max} \tanh(\hat{c}_k) k^{-p} \cos(4k\theta + \phi_k)\big)$ with QP-projected gap constraints.
