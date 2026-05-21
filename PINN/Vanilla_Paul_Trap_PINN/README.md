# Vanilla Paul Trap PINN

This PINN solves for a Paul Trap with circular electrodes.

## Overview

The PINN solves directly for the "RF" DC potential ($\Phi_{RF}$) by placing the RF potential at a constant electric potential as specified in `config["boundary_conditions"]["rf_voltages"]` and solving Poisson's equation. The pseudopotential $\psi_{RF}$ is then calculated via:

$$
\psi_RF = \frac{e^2}{4M\Omega^2_{RF}}|\nabla \Phi_{RF}|^2
$$

During post-processing.