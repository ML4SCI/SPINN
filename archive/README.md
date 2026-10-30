# Archive

Superseded code, kept for reference and provenance. None of it is imported by
`spinn/`, the active package — cross-checked before archiving (see
`../Resources/Project_Audit_and_Roadmap.md`, 2026-06-09, for the audit that
motivated `spinn/`'s design). Nothing here is deleted, only moved; treat it as
read-only history rather than a place to add new work.

| Directory | What it was | Last touched | Superseded by |
|---|---|---|---|
| `PINN/` | First configurable 2D vanilla PINN (`Vanilla_Paul_Trap_PINN`) plus early coordinate-projection prototypes (`Coordinate_Projection`, `Coordinate_Projection_Network`) | 2026-06-22 | `spinn/models/shape_network.py`, `spinn/physics/` |
| `experiments/` | Three overlapping architecture-comparison studies (`coordinate_architecture_comparison`, `cpn_pinn_shape_optimization`, `potential_pinn_architectures`) exploring MLP/PIXEL/PIG backends ad hoc | 2026-06-23 | `spinn/physics_backend` (config-driven MLP/PIXEL/PIG in one harness) |
| `benchmark/` | A benchmark harness comparing an "instructor" coordinate-projection implementation against DEVSIM and a PINN baseline | 2026-06-23 | `spinn/eval/compare_backends.py`, `spinn/physics/fem_reference.py` |
| `Hyperparameter/` | Manual one-factor-at-a-time sweep tooling (width/depth/activation/boundary-weight/optimizer/lr) with custom report generation | 2026-06-16 | `spinn/scripts/tune_hard_bc.py` (Optuna-based) and `spinn/config.py` |
| `tests/` | Tests for the above (`test_coordinate_projection.py`, `test_instructor_coordinate_projection.py`, `test_metrics.py`) | 2026-06-23 | `spinn/tests/` |

If you need something from here, prefer porting the specific piece into
`spinn/` over reviving the directory wholesale — the overlapping generations
above are exactly the duplication `spinn/` was built to consolidate.

## Retired experiment scripts

`scratch_retired/` and `exp_retired/` hold experiment scripts whose idea was
tested and rejected, or which a later script replaced. Their findings are
still recorded in the experiment logs, and the scripts are kept so those
results can be reproduced. Run them from the repository root with the root
and `scratch/` on the path, since some import `pig_gate_common`:

```bash
PYTHONPATH=.:scratch python3 archive/scratch_retired/08_known_circular_rod_optimum.py
```

Plots still go to their original folders (`scratch/`, `exp/<campaign>/results/`).

| Scripts | Why retired | Replaced by |
|---|---|---|
| `scratch_retired/02` | Compares the old saved MLP/PIXEL/PIG checkpoints; a historical smoke test. | `scratch/33`, `scratch/35` |
| `scratch_retired/08`, `09` | Raster finite-difference objective missed the 1.145 optimum because of rasterization error. | `scratch/10` (exact boundaries) |
| `scratch_retired/15` | 500-step soft-boundary MLP got the a6 sign wrong. The finding stands, and the setup is superseded. | `scratch/16`, `scratch/32`-`34` |
| `scratch_retired/20` | First output-symmetric PIG field gate. | `scratch/32`-`34` |
| `scratch_retired/21`-`24`, `41` | Tying Gaussian features or restricting centers to a quadrant failed the field gates. Symmetry belongs at the output. | `scratch/17`, `scratch/40` |
| `scratch_retired/25`-`28`, `30`, `31` | PIG stabilization attempts: longer runs, checkpoint selection, resampling, boundary quadrature and a tangential penalty. None fixed the endpoint. | `scratch/32` (Adam then L-BFGS) |
| `scratch_retired/36`, `37` | Per-geometry PIG shape response; it lacks the derivative path. | `scratch/38`, `scratch/39` |
| `scratch_retired/44` | Three-slice conditioned surrogate gave a convincing but false null-gradient curve. | Denser geometry collocation (plots 45, 46) |
| `scratch_retired/52`, `54`, `56` | Mean-value loss variants (circle radius, single-seed transfer, resampled quadrature). | `scratch/53`, `scratch/55` |
| `scratch_retired/57`, `58` | Harmonic measurement core; the high multipole was weakly identified. | `exp/claude_tree_stress_test/30`-`32` |
| `exp_retired/harmonic_source_shape_search/03`-`05` | Elliptical electrodes increased nonlinearity. | Support-function search (`06`-`09`) |
| `exp_retired/harmonic_source_shape_search/12` | Triple-zero c8 electrode left only 0.061 r0 enclosure clearance. | None; rejected in this enclosure |
| `exp_retired/claude_tree_stress_test/01`, `13` | Audits that disproved claims in the source idea tree. | None needed |
| `exp_retired/claude_tree_stress_test/03`, `17` | Sparse lifetime scans. | `04`, `18`, `19` |
| `exp_retired/claude_tree_stress_test/25` | Potential-only geometry-conditioned PIG; it fit V well but got beta wrong. | Moment-supervised gates and `30`-`32` |
| `exp_retired/claude_tree_stress_test/34`, `39` | Purely additive geometry heads were rejected. | `30`-`32` |
