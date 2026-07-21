# Paul-Trap Design Benchmark: PINN vs. Classical FEM

This benchmark answers one question honestly: **for differentiable Paul-trap
shape design, is a physics-informed neural network actually the right tool, or
does a traditional finite-element solver win?**

It is the concrete follow-up to the critique in *Progress and Perils of PINNs*
(Hyde et al., 2026), whose central warnings — soft constraints, "low residual
≠ low error", non-convergence, brittleness, and hidden training/data cost — are
turned here into measurements rather than opinions.

## The idea

Run the **same** design task through interchangeable methods and score them on
**one shared scorecard**. The classical DEVSIM finite-element solve is ground
truth; every other method's shape gradient is measured against it.

```
benchmark/
  task.py                # the design task + scorecard data model (method-agnostic)
  methods/
    base.py              # ShapeMethod interface every method implements
    devsim.py            # classical FEM + finite-difference gradient  (the reference)
    pinn.py              # coordinate-projection PINN + autograd gradient (under test)
  run_benchmark.py       # driver: runs methods, scores them, writes results
  results/               # scorecard.json, scorecard.md, gradient_comparison.png
```

Adding a method (e.g. a neural operator) is one new file in `methods/`
implementing `ShapeMethod.evaluate`; nothing else changes.

## The task

* **Design variable:** electrode-center `distance` (1-D first, so the classical
  finite-difference gradient is cheap and unambiguous).
* **Objective `J`:** local quadrupole efficiency at the trap center — the same
  scalar used throughout the coordinate-projection code.
* **Reference gradient:** centered five-point finite difference of the DEVSIM
  objective. DEVSIM exposes no analytic adjoint, so this finite-difference
  sensitivity (accurate to the mesh tolerance) is the honest classical gradient.

## The metrics

| Block | What it measures | Maps to PINN critique |
|---|---|---|
| **Accuracy** | shape-gradient relative error & sign vs. FEM | "low residual ≠ low error" |
| **Cost** | training + data-gen + eval wall clock, end-to-end | hidden cost of PINNs |
| **Robustness** | gradient error as the design nears the trained edge; seed-to-seed variance | brittleness, non-determinism |

Because the PINN works in a *normalized* potential and DEVSIM in volts, the
headline accuracy number is the **scale-invariant relative sensitivity**
`d(log J)/d(distance)`, which compares the two fairly. Raw gradients are also
recorded in `scorecard.json`.

## Run it

```bash
cd SPINN

# Fast wiring check (PINN barely trained — not a real verdict):
python -m benchmark.run_benchmark --training-steps 50 \
    --world-mesh-size 0.06 --electrode-mesh-size 0.008

# Real run: trained PINN, three seeds, comparison plot:
python -m benchmark.run_benchmark --training-steps 6000 --pinn-seeds 0 1 2 --plot
```

* DEVSIM solves are **cached** by geometry+mesh hash, so the expensive FEM work
  runs once; reruns are instant.
* Trained PINNs are **cached** per (seed, steps); rerunning does not retrain.
* If DEVSIM/gmsh are not installed, that method reports `available=false` and
  the run still completes (without an accuracy verdict).

## Reading the result

`results/scorecard.md` has the table and an explicit verdict line per method:
whether its gradient is **usable** (relative error < 5% *and* sign agrees) and
whether it is **cheaper end-to-end** than the FEM solver. `d(log J)/dd` with the
wrong sign means a PINN-based optimizer would step the geometry the wrong way —
the single most important thing to check.

## Honest scope

* This settles the **1-D, fixed-topology** case. The coordinate map represents
  smooth deformations only; adding/removing electrodes (a topology change) is
  outside what the PINN method can express and is called out, not faked.
* A neural-operator baseline (the paper's suggested alternative for genuine
  cross-geometry generalization) is a natural next `methods/` entry once the
  1-D accuracy/cost verdict is in.
