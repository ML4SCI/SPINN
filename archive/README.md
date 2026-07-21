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
