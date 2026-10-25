# SPINN

The consolidated scientific results, claim status, negative experiments, and artifact index are maintained in [the master RESULTS document](exp/harmonic_source_shape_search/RESULTS.md). The open-ended audit of the ten-branch Claude artifact, including the new conformal center-jet electrode family, is in [its dedicated experiment record](exp/claude_tree_stress_test/RESULTS.md).

Differentiable Paul-trap design: physics-informed field solvers and
coordinate-projection shape optimization for a 2D four-rod linear Paul trap,
validated against an independent finite-difference solver.

## Layout

```text
SPINN/
  spinn/        the active package -- start here (spinn/README.md)
  scratch/      short numbered exploration scripts
  exp/          multi-script experiment campaigns
  prototypes/   standalone single-file PINN / PIG / PIXEL / CPN / SPINN scripts
                (Dale's fixed-geometry line); independent of spinn/
  DEVSIM/       reference Gmsh/DEVSIM finite-element pipelines (circular,
                rectangular, inline, hyperbolic trap cross-sections); used as
                an independent cross-check, not a PINN implementation
  external/     vendored comparison codebases (PIXEL, Physics-Informed-Gaussians)
                pulled in for the backend benchmark; not maintained here
  archive/      superseded PINN/experiment/benchmark/sweep code, kept for
                provenance -- see archive/README.md before reviving anything
```

Only code, configs and trained model weights (`*.pt`) are version-controlled.
Plots, results, reports, the paper, slides and agent notes are generated or
kept locally and ignored by git (see `.gitignore`).

## Where things are

* **Current work** (models, losses, training, tuning, results): `spinn/` --
  see `spinn/README.md` for the module map and how to run it.
* **Independent physics validation**: `DEVSIM/` (finite-element) and
  `spinn/physics/fem_reference.py` (finite-difference, called from `spinn/`
  directly -- no need to touch `DEVSIM/` for day-to-day PINN validation).
* **Vendored backend implementations** referenced by `spinn/physics_backend`
  configs (PIXEL, PIG): `external/`.
* **Anything not listed above**: probably in `archive/`. Check there before
  writing something from scratch.

## Tests

Install the active package and its test dependency once:

```bash
python3 -m pip install -e '.[dev]'
```

Then run:

```bash
python3 -m pytest
```
