"""Explicit gradient-flow diagnostics.

Differentiable shape optimization only works if a single backward pass delivers
finite, nonzero gradients to BOTH the PINN and the CPN parameter groups. These
checks make that property testable rather than assumed.
"""

from __future__ import annotations

import torch


def _group_has_finite_nonzero_grad(parameters) -> bool:
    saw_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
        if torch.any(parameter.grad != 0):
            saw_grad = True
    return saw_grad


def check_gradient_flow(coupled_model, loss: torch.Tensor) -> dict[str, bool]:
    """Backpropagate ``loss`` and report per-group gradient health.

    The model's gradients are zeroed first so the result reflects only this
    backward pass.
    """

    coupled_model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    return {
        "pinn_grads_finite_nonzero": _group_has_finite_nonzero_grad(coupled_model.pinn_parameters()),
        "cpn_grads_finite_nonzero": _group_has_finite_nonzero_grad(coupled_model.cpn_parameters()),
    }
