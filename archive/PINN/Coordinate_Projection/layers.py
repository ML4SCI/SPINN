"""Shared building blocks for the coordinate-projection networks."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class Sine(nn.Module):
    """Sinusoidal activation used by SIREN-style networks."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(inputs)


def validate_second_order_activation(name: str) -> None:
    if name.lower() in {"relu", "leaky_relu", "hardtanh"}:
        raise ValueError(
            f"{name} is unsuitable for a strong-form second-order PINN because "
            "its second derivative is zero or undefined almost everywhere"
        )


def resolve_activation(name: str) -> Callable[[], nn.Module]:
    """Map an activation name to a factory producing fresh module instances."""

    key = name.lower()
    if key == "tanh":
        return nn.Tanh
    if key in {"sine", "sin", "sinusoidal"}:
        return Sine
    if key == "gelu":
        return nn.GELU
    raise ValueError(f"unsupported activation: {name}")


def build_mlp(
    in_dim: int,
    out_dim: int,
    *,
    width: int,
    depth: int,
    activation_factory: Callable[[], nn.Module],
    zero_initialize_output: bool = False,
) -> nn.Sequential:
    """Build a fully connected network with ``depth`` hidden layers.

    The output layer is linear. When ``zero_initialize_output`` is set its
    weight and bias are zeroed, which lets a residual map start as the exact
    identity.
    """

    layers: list[nn.Module] = [nn.Linear(in_dim, width), activation_factory()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), activation_factory()])
    output = nn.Linear(width, out_dim)
    if zero_initialize_output:
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)
