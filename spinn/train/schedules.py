"""Learning-rate schedules. Plan optimizer table: cosine or exponential decay."""

from __future__ import annotations

import math

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, kind: str, total_steps: int):
    kind = (kind or "none").lower()
    if kind in {"none", ""}:
        return None
    if kind in {"cosine", "cosine_decay"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    if kind in {"exp", "exponential", "exponential_decay"}:
        # decay to ~1% of the initial rate over the run
        gamma = math.exp(math.log(0.01) / max(total_steps, 1))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    raise ValueError(f"unknown scheduler {kind!r}")
