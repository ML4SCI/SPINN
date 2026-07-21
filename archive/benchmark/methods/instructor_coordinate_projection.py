"""Benchmark wrapper for the instructor-generated coordinate projection model.

The instructor model is a coordinate map trained toward the identity; it has no
potential field and no electrode-distance parameter, so it does not produce the
shape derivative ``dJ/dd`` that the other benchmark methods report. This wrapper
therefore leaves ``objective`` and ``gradient`` unset and instead records map
diagnostics (identity fidelity and Jacobian conditioning) in ``extra``, which is
what determines whether the map could support a shape gradient at all.

The wrapper implements the same :class:`~benchmark.methods.base.ShapeMethod`
interface as the other methods so the driver can run it uniformly.
"""

from __future__ import annotations

import time

from benchmark.experiments.instructor_coordinate_projection.diagnostics import map_diagnostics
from benchmark.experiments.instructor_coordinate_projection.model import (
    CoordinateProjectionNet,
    TrainingConfig,
    TrapGeometry,
    train_identity_map,
)

from ..task import DesignTask, MethodResult
from .base import ShapeMethod


class InstructorCoordinateProjectionMethod(ShapeMethod):
    """Train the instructor coordinate map and report its map diagnostics."""

    name = "instructor_coordinate_projection"
    is_reference = False

    def __init__(self, *, training_steps: int = 6000, seed: int = 0) -> None:
        self.training_steps = training_steps
        self.seed = seed

    def evaluate(self, task: DesignTask) -> MethodResult:
        geometry = TrapGeometry(
            world_size_x=task.world_size_x,
            world_size_y=task.world_size_y,
            radius=task.radius,
            north_distance=task.distance,
            south_distance=task.distance,
            west_distance=task.distance,
            east_distance=task.distance,
        )
        model = CoordinateProjectionNet(width=50, depth=6, zero_last_layer=False)
        config = TrainingConfig(steps=self.training_steps)

        training = train_identity_map(model, geometry, config, seed=self.seed)
        start = time.perf_counter()
        diagnostics = map_diagnostics(model, geometry, seed=self.seed)
        eval_seconds = time.perf_counter() - start

        return MethodResult(
            name=self.name,
            available=True,
            objective=None,
            gradient=None,
            setup_seconds=training.train_seconds,
            eval_seconds=eval_seconds,
            deterministic=False,
            notes=(
                "coordinate map trained toward identity (not physics-informed); "
                "reports map diagnostics, not a shape gradient"
            ),
            extra={
                "training_steps": self.training_steps,
                "seed": self.seed,
                "freshly_trained": True,
                **diagnostics,
            },
        )
