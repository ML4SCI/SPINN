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
| mlp | 0.9296 +/- 0.0000 | 0.9296 +/- 0.0000 | 0.9278 | 0.9986 | True |
| pig | 0.9296 +/- 0.0000 | 0.8672 +/- 0.0000 | 6.5761 | 0.9481 | True |
| pixel | 0.9296 +/- 0.0000 | 0.8672 +/- 0.0000 | 38.7866 | 0.9512 | True |


## Hypothesis verdict (plan sec. 7)

```
pig_eta_fem: 0.9296059720423698
best_baseline_eta_fem: 0.9296059369001108
absolute_gain: 3.514225899170498e-08
relative_gain: 3.780339345603935e-08
H1_supported: False
verdict: H0 not rejected
```
