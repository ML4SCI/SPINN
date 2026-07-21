# SPINN Hyperparameter Results Speaker Guide

## Main Readout
- Best DEVSIM agreement came from bc_weight_1000: Phi MSE 0.619, Psi MSE 0.0388.
- That is about 95x lower RF-potential MSE than the baseline.
- ReLU is the cautionary case: near-zero training loss, but the worst DEVSIM field error.
- Among optimizer-only variants, RMSprop had the lowest DEVSIM error: Phi MSE 14.5.

## Graph Analysis
### Normalized convergence
- Normalized loss lets runs with different starting losses be compared on the same scale.
- Fast normalized convergence did not always mean physical accuracy.
- ReLU ended at 3.58e-08 normalized loss but failed the DEVSIM comparison.
- BC weight 1000 also converged strongly and was physically accurate: Phi MSE 0.619.

### RF potential error
- Lowest RF-potential error: bc_weight_1000 at Phi MSE 0.619.
- Largest RF-potential error: activation_relu at Phi MSE 7.2e+03.
- Boundary-condition weighting dominated field accuracy in this first sweep.
- Wider networks and sinusoidal activations were the strongest architecture-side options.

### RF pseudopotential error
- Lowest pseudopotential error: bc_weight_1000 at Psi MSE 0.0388.
- Pseudopotential depends on gradients, so it punishes field-shape errors more than voltage MSE alone.
- The same variants that improved Phi generally improved Psi, but high-frequency instability can still appear in gradients.

### Final normalized loss
- Final training loss alone is not a reliable model-selection criterion here.
- ReLU is artificially favored because the second derivative of ReLU is zero almost everywhere.
- The best physical run, bc_weight_1000, also had a low final normalized loss, so it is the most credible winner.

### Training loss vs. RF potential error
- Ideal runs sit toward the lower-left: low training loss and low DEVSIM error.
- ReLU sits far from the physical winner despite its tiny training loss, exposing a residual-definition failure.
- BC weight 1000 is the cleanest point because it pairs low loss with the lowest field error.
- Use this graph as the model-selection warning slide: optimization metrics need ground-truth validation.

## Sweep Notes
### baseline
- Baseline gives a useful reference: Phi MSE 58.9, Psi MSE 6.89.
- Several variants beat the baseline decisively, so the PINN is hyperparameter-sensitive.

### width
- Increasing width helped the physical field much more than it helped final training loss.
- Width 256 was best: Phi MSE 10.4, compared with 334 for the worst width run.

### depth
- Depth variants were unstable in this setup; deeper did not mean better.
- The best depth-only run was depth_6, but it still lagged the baseline by DEVSIM error.

### activation
- Sinusoidal activation was the best activation-only variant for field accuracy.
- ReLU produced misleadingly tiny loss because the Laplace residual becomes degenerate under second derivatives.

### bc_weight
- Boundary-condition weight was the strongest lever in this sweep.
- Too little BC weight let boundary mismatch dominate; 1000 gave the cleanest DEVSIM match.

### optimizer
- Optimizer choice changed both convergence and physical accuracy.
- RMSprop had the best optimizer-only DEVSIM error; SGD plateaued with large boundary error.

### learning_rate
- Lower learning rates were smoother but too slow for this fixed training budget.
- The high-rate run moved fastest but had large PDE spikes; it still beat the lower-rate runs by DEVSIM error.

## Final Discussion Summary
- Use the final slide to separate optimization speed from physical correctness: ReLU reached normalized loss below 1e-3 by step 2900, but it had the worst DEVSIM mismatch, with Phi MSE 7.2e+03 and Psi MSE 3.97e+03.
- Among credible runs, RMSprop reached normalized loss below 1e-2 first at step 7700, while bc_weight_1000 reached below 1e-3 by step 11000 and had the best total PINN-minus-ground-truth MSE: Phi MSE 0.619 and Psi MSE 0.0388.
- The strongest architecture-side options were width_256 and activation_sinusoidal; the final slide table gives every variant's convergence threshold, final normalized loss, Phi MSE, and Psi MSE.
- The raw PINN-minus-DEVSIM plots in the appendix should be used as the visual check: the best runs have localized, low-amplitude residuals, while poor runs show broad field-shape mismatch.
