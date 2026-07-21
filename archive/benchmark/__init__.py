"""Head-to-head benchmark of PINN vs. classical methods for Paul-trap design.

The benchmark answers a single question: for differentiable Paul-trap shape
design, is a physics-informed neural network actually the right tool compared
with a traditional finite-element solver?

It does so by running the *same* design task through several interchangeable
methods (see :mod:`benchmark.methods`) and scoring them on one shared
scorecard (see :mod:`benchmark.task`):

* accuracy of the forward objective and of the shape gradient,
* end-to-end cost (including any training / data-generation),
* robustness as the design point moves toward the edge of the trained region.

The classical DEVSIM solve is treated as ground truth; its finite-difference
shape gradient is the reference the PINN gradient is measured against.
"""

from .task import (
    DesignTask,
    GradientComparison,
    MethodResult,
    Scorecard,
)

__all__ = [
    "DesignTask",
    "GradientComparison",
    "MethodResult",
    "Scorecard",
]
