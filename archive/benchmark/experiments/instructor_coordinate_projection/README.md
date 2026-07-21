# Instructor Coordinate Projection Benchmark

An isolated experiment testing whether the instructor-generated coordinate
projection model fixes, improves, or helps diagnose the wrong-sign shape
gradient seen in the main benchmark. It is kept separate from the main model and
benchmark code so it can be deleted, merged, or promoted without disturbing
them.

## 1. The original negative-gradient issue

The main benchmark (`SPINN/benchmark/`) optimizes one design variable, the
electrode-center distance `d`, and compares the shape gradient `dJ/dd`, where
`J` is the quadrupole efficiency at the trap center.

The DEVSIM finite-element reference gives

```
dJ/dd ~ -1.44
```

meaning efficiency decreases as the electrodes move apart. The existing
coordinate-projection PINN instead reports `dJ/dd > 0` across seeds: the wrong
sign. A shape optimizer consumes the gradient, so a wrong-sign gradient drives
the geometry in the wrong direction. This is the problem the experiment probes.

## 2. What the instructor-generated model is

The instructor model (`PINN/Coordinate_Projection_Network/CPN.py`) is a residual
coordinate map

```
NN_phi(z) = z + NNhat_phi(z)
```

where a 6-layer, width-50 tanh MLP learns the displacement `NNhat_phi`. It is
trained on the trap geometry with a mean Euclidean coordinate loss to approximate
the identity map, so the learned displacement should be near zero. Its
displacement-field plot looks reasonable.

## 3. How it differs from the existing coordinate-projection PINN

| | Existing PINN (`PINN/Coordinate_Projection/`) | Instructor model |
|---|---|---|
| Output | potential field, composed as `NN_theta(NN_phi(z))` | coordinate map only |
| Training signal | pulled-back PDE residual + boundary + Jacobian constraints | mean Euclidean coordinate error toward identity |
| Distance parameter | yes, conditions on `d` | no |
| Produces `dJ/dd` | yes (autograd through the field network) | no |
| Physics-informed | yes | no |

The key consequence: the instructor model is the *shape-map half* of the
architecture. It has no potential field and no distance parameter, so it cannot
by itself produce `dJ/dd`. It is therefore benchmarked on map quality, not on a
shape gradient.

## 4. Files added

```
benchmark/methods/instructor_coordinate_projection.py     ShapeMethod wrapper
benchmark/experiments/instructor_coordinate_projection/
  __init__.py
  model.py            instructor MLP + CoordinateProjectionNet, importable (no script side effects)
  diagnostics.py      identity fidelity + Jacobian statistics on the trap domain
  run_comparison.py   Experiment 3 driver
  run_step_sweep.py   Experiment 4 driver
  README.md           this file
  results/            JSON and figures
tests/test_instructor_coordinate_projection.py            smoke tests
```

## 5. Files reused

* `benchmark/task.py`, `benchmark/methods/base.py` for the method interface.
* `benchmark/methods/devsim.py` and the main scorecard for the reference gradient.
* `benchmark/methods/pinn.py` for the step-count sweep of the existing PINN.
* `PINN/Coordinate_Projection/constraints.py` (`jacobian_constraints`) and the
  operators it builds on, for the Jacobian diagnostics.

The instructor model's architecture, residual form, loss, and zero-last-layer
option are taken verbatim from `CPN.py`; only the script-level execution,
globals, and plotting were removed so the module imports cleanly.

## 6. How to run

```bash
cd SPINN

# Experiment 3: instructor map diagnostics (three seeds)
python -m benchmark.experiments.instructor_coordinate_projection.run_comparison \
    --steps 6000 --seeds 0 1 2

# Experiment 4: step-count sweep for both methods
python -m benchmark.experiments.instructor_coordinate_projection.run_step_sweep \
    --steps 6000 20000 50000 --seeds 0

# Just the existing PINN sweep, or just the instructor sweep
python -m benchmark.experiments.instructor_coordinate_projection.run_step_sweep \
    --steps 6000 20000 50000 --skip-instructor
python -m benchmark.experiments.instructor_coordinate_projection.run_step_sweep \
    --steps 6000 20000 50000 --skip-pinn
```

The main DEVSIM and existing-PINN benchmark is run separately:

```bash
python -m benchmark.run_benchmark --pinn-seeds 0 1 2 --plot
```

## 7. Interpreting gradient sign and scale

For the existing PINN, the gradient is correct only if its sign matches DEVSIM
(negative). Magnitude matters second: the main benchmark uses the scale-invariant
`d(log J)/dd` for cross-method comparison because the PINN works in a normalized
potential. For the instructor map there is no gradient; the relevant quantities
are RMS identity error (should be small) and the Jacobian determinant (should be
positive and near 1, with a low condition number). A clean, positive Jacobian
means the map *could* support a correct shape gradient; a degenerate one would
corrupt any gradient built on top of it.

## 8. Assumptions

* The instructor model is evaluated on the same trap geometry as the main
  benchmark (distance 0.125, radius 0.05, world 2.5 x 2.0).
* Jacobian diagnostics use interior points only, excluding electrode disks,
  matching where the field problem is defined.
* The Jacobian determinant is computed in float64 for numerical stability.
* The instructor model is treated as a coordinate map, not a physics solver; it
  is never labelled physics-informed.

## 9. Unresolved

* Whether composing the instructor map with a trained potential network would
  produce a correct-sign `dJ/dd`. That is a deliberate next step (a hybrid model),
  not done here, because it would no longer test the instructor model in isolation.
* Whether the existing PINN's wrong sign is under-training or architectural; the
  step-count sweep addresses this and its result is recorded in `results/`.
