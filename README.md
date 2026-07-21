# SPINN

Differentiable Paul-trap design: physics-informed field solvers and
coordinate-projection shape optimization for a 2D four-rod linear Paul trap,
validated against an independent finite-difference solver. See
`../Resources/Project_Audit_and_Roadmap.md` for the research thesis and
`../Resources/SPINN_Project_Reference.md` for background.

## Layout

```
SPINN/
  spinn/        the active package -- start here (spinn/README.md)
  DEVSIM/       reference Gmsh/DEVSIM finite-element pipelines (circular,
                rectangular, inline, hyperbolic trap cross-sections); used as
                an independent cross-check, not a PINN implementation
  external/     vendored comparison codebases (PIXEL, Physics-Informed-Gaussians)
                pulled in for the backend benchmark; not maintained here
  archive/      superseded PINN/experiment/benchmark/sweep code, kept for
                provenance -- see archive/README.md before reviving anything
```

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

```bash
python -m pytest    # runs spinn/tests via pytest.ini
```
