# SPINN Paul-trap shape-optimization report

## Fixed-geometry solver quality

| backend | eta_pinn | eta_fem | \|error\| |
|---|---|---|---|
| mlp | 0.9838 | 0.9296 | 0.0541 |
| pig | 1.0088 | 0.9296 | 0.0792 |
| pixel | 0.9946 | 0.9296 | 0.0650 |


## Joint shape-optimization result

| backend | best valid eta_fem (mean) | final eta_fem (mean) | eta mismatch | min detJ | valid |
|---|---|---|---|---|---|
| mlp | 1.0139 +/- 0.0000 | 1.0139 +/- 0.0000 | 3.1596 | 0.7506 | True |
| pig | 0.9296 +/- 0.0000 | 0.4765 +/- 0.0000 | 0.9343 | 0.5382 | True |
| pixel | 1.0561 +/- 0.0000 | 1.0210 +/- 0.0000 | 15.3600 | -1.3548 | False |


## Hypothesis verdict (plan sec. 7)

```
pig_eta_fem: 0.9296054443715414
best_baseline_eta_fem: 1.0560655184574776
absolute_gain: -0.1264600740859363
relative_gain: -0.11974642848925494
H1_supported: False
verdict: H0 not rejected
```
