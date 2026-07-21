"""Coordinate-projection PINN: autograd objective and shape gradient.

This is the method under test. A parametric potential network is trained
against the pulled-back physics and geometry constraints, after which the
quadrupole efficiency and its derivative with respect to electrode distance
are obtained by automatic differentiation -- exactly the gradient a PINN-based
shape optimizer would consume.

Training cost is reported honestly as ``setup_seconds`` so the benchmark can
weigh it against the classical solver, per the central cost critique of PINNs.
The trained network is cached so repeated benchmark runs do not retrain.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from PINN.Coordinate_Projection.train import (
    build_components,
    efficiency_and_gradient,
    load_config,
    train_model,
    deep_update,
)

from ..task import DesignTask, MethodResult
from .base import ShapeMethod


class CoordinateProjectionPINNMethod(ShapeMethod):
    """Train a coordinate-projection PINN, then differentiate the objective."""

    name = "coordinate_projection_pinn"
    is_reference = False

    def __init__(
        self,
        cache_dir: Path,
        *,
        training_steps: int = 4000,
        seed: int = 0,
        config_overrides: dict | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.training_steps = training_steps
        self.seed = seed
        self.config_overrides = config_overrides or {}

    def evaluate(self, task: DesignTask) -> MethodResult:
        config = self._build_config(task)
        model, coordinate_map, setup_seconds, retrained = self._load_or_train(config)

        start = time.perf_counter()
        efficiency, gradient, _ = efficiency_and_gradient(
            model, coordinate_map, task.distance, config["geometry"]
        )
        eval_seconds = time.perf_counter() - start

        return MethodResult(
            name=self.name,
            available=True,
            objective=float(efficiency.detach()),
            gradient=float(gradient.detach()),
            setup_seconds=setup_seconds,
            eval_seconds=eval_seconds,
            deterministic=False,
            notes=(
                f"{'trained this run' if retrained else 'loaded from cache'} PINN; "
                f"autograd gradient through the residual coordinate map"
            ),
            extra={
                "training_steps": self.training_steps,
                "seed": self.seed,
                "freshly_trained": retrained,
                "training_seconds": setup_seconds,
            },
        )

    # -- internals ----------------------------------------------------------

    def _build_config(self, task: DesignTask) -> dict:
        config = load_config(None)
        config = deep_update(config, {
            "seed": self.seed,
            "geometry": {
                "reference_distance": task.distance,
                "distance_min": task.distance_min,
                "distance_max": task.distance_max,
                "radius": task.radius,
                "world_size_x": task.world_size_x,
                "world_size_y": task.world_size_y,
            },
            "training": {"steps": self.training_steps},
        })
        return deep_update(config, self.config_overrides)

    def _checkpoint_path(self) -> Path:
        return self.cache_dir / f"pinn_seed{self.seed}_steps{self.training_steps}.pt"

    def _load_or_train(
        self, config: dict
    ) -> tuple[torch.nn.Module, torch.nn.Module, float, bool]:
        path = self._checkpoint_path()
        model, coordinate_map = build_components(config)

        if path.exists():
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(bundle["model_state_dict"])
            model.eval()
            return model, coordinate_map, bundle.get("setup_seconds", 0.0), False

        torch.manual_seed(config["seed"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        train_model(model, coordinate_map, config, self.cache_dir)
        setup_seconds = time.perf_counter() - start
        model.eval()

        torch.save(
            {"model_state_dict": model.state_dict(), "setup_seconds": setup_seconds},
            path,
        )
        return model, coordinate_map, setup_seconds, True
