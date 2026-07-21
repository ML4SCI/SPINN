# Results index

Run outputs, in rough chronological/topical order. Directories prefixed
`archive_YYYYMMDD_HHMMSS` are superseded runs kept for provenance (own
existing convention -- follow it for future superseded runs rather than
deleting).

| Path | Content |
|---|---|
| `fixed/`, `joint/` | Main fixed-geometry and joint shape-optimization benchmark runs (plan §6/§8: `metrics.csv`, `checkpoints.csv`) |
| `figures/` | Headline plots (`eta_fem_vs_checkpoint.png`, `eta_pinn_vs_fem_scatter.png`, etc.) |
| `backend/`, `backend_differentiability/` | Backend-comparison and differentiability diagnostics |
| `ablations/`, `shape_coupling_diagnostic/`, `e_more_shape_lr_full/` | Targeted ablation/diagnostic runs |
| `hard_bc_tuning/` | Hard-constraint-ansatz Optuna study, budget test, and the hard-vs-soft BC 3-seed A/B (see `presentation_slides.md` there for the full writeup) |
| `archive_*/` | Superseded runs, kept for provenance |
| `report.md`, `report_tables.md`, `results_summary.md`, `plot_navigation_summary.md`, `backend_reference_audit.md` | Generated summaries -- see `plot_navigation_summary.md` first if you're looking for a specific figure |
