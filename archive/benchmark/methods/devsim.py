"""Classical finite-element reference: DEVSIM + finite-difference gradient.

DEVSIM solves the electrostatics on a real triangular mesh, the same way the
rest of the project treats it as ground truth. Because DEVSIM exposes no
analytic adjoint, the trusted shape gradient is a centered five-point finite
difference of the DEVSIM objective over the electrode distance -- the honest
classical sensitivity, accurate to the mesh tolerance.

Every solve is cached to disk keyed by (distance, mesh, geometry), so the
expensive FEM work runs once and reruns are instant.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from PINN.Coordinate_Projection.validate_gradient import finite_difference_derivatives

from ..task import DesignTask, MethodResult
from .base import ShapeMethod

SPINN_ROOT = Path(__file__).resolve().parents[2]
DEVSIM_MODULE = "PINN.Coordinate_Projection.devsim_reference"


class DevsimFiniteDifferenceMethod(ShapeMethod):
    """Ground-truth FEM objective with a finite-difference shape gradient."""

    name = "devsim_fem"
    is_reference = True

    def __init__(
        self,
        cache_dir: Path,
        *,
        world_mesh_size: float = 0.04,
        electrode_mesh_size: float = 0.004,
        fit_radius: float = 0.035,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.world_mesh_size = world_mesh_size
        self.electrode_mesh_size = electrode_mesh_size
        self.fit_radius = fit_radius

    # -- public API ---------------------------------------------------------

    def evaluate(self, task: DesignTask) -> MethodResult:
        try:
            import devsim  # noqa: F401
            import gmsh  # noqa: F401
        except Exception as error:  # pragma: no cover - environment dependent
            return MethodResult(
                name=self.name,
                available=False,
                notes=f"DEVSIM/gmsh unavailable: {error}",
            )

        task.validate_stencil_in_range()
        distances = [task.distance, *task.stencil()]

        start = time.perf_counter()
        efficiencies = {d: self._efficiency(task, d) for d in distances}
        elapsed = time.perf_counter() - start

        three_point, five_point = finite_difference_derivatives(
            [efficiencies[d] for d in task.stencil()],
            task.gradient_epsilon,
        )
        stencil_spread = abs(five_point - three_point) / max(abs(five_point), 1e-12)

        return MethodResult(
            name=self.name,
            available=True,
            objective=efficiencies[task.distance],
            gradient=five_point,
            setup_seconds=0.0,
            eval_seconds=elapsed,
            deterministic=True,
            notes="finite-element ground truth; five-point finite-difference gradient",
            extra={
                "three_point_gradient": three_point,
                "stencil_relative_spread": stencil_spread,
                "world_mesh_size_cm": self.world_mesh_size,
                "electrode_mesh_size_cm": self.electrode_mesh_size,
                "n_fem_solves": len(efficiencies),
            },
        )

    # -- internals ----------------------------------------------------------

    def _efficiency(self, task: DesignTask, distance: float) -> float:
        """Quadrupole efficiency at one distance, cached across runs."""

        cached = self._read_cache(task, distance)
        if cached is not None:
            return cached
        metrics = self._run_devsim(task, distance)
        efficiency = float(metrics["quadrupole_efficiency"])
        self._write_cache(task, distance, metrics)
        return efficiency

    def _cache_path(self, task: DesignTask, distance: float) -> Path:
        signature = json.dumps(
            {
                "distance": round(distance, 8),
                "radius": task.radius,
                "rf_voltage": task.rf_voltage,
                "world_size": [task.world_size_x, task.world_size_y],
                "world_mesh": self.world_mesh_size,
                "electrode_mesh": self.electrode_mesh_size,
                "fit_radius": self.fit_radius,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(signature.encode()).hexdigest()[:16]
        return self.cache_dir / f"devsim_{digest}.json"

    def _read_cache(self, task: DesignTask, distance: float) -> float | None:
        path = self._cache_path(task, distance)
        if not path.exists():
            return None
        return float(json.loads(path.read_text())["quadrupole_efficiency"])

    def _write_cache(self, task: DesignTask, distance: float, metrics: dict) -> None:
        path = self._cache_path(task, distance)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))

    def _run_devsim(self, task: DesignTask, distance: float) -> dict:
        """Mesh and solve one geometry via the existing reference module."""

        run_dir = self.cache_dir / f"solve_{distance:.6f}"
        command = [
            sys.executable,
            "-m",
            DEVSIM_MODULE,
            "--single",
            "--distance",
            str(distance),
            "--output-dir",
            str(run_dir),
            "--radius",
            str(task.radius),
            "--rf-voltage",
            str(task.rf_voltage),
            "--world-size-x",
            str(task.world_size_x),
            "--world-size-y",
            str(task.world_size_y),
            "--world-mesh-size",
            str(self.world_mesh_size),
            "--electrode-mesh-size",
            str(self.electrode_mesh_size),
            "--fit-radius",
            str(self.fit_radius),
        ]
        subprocess.run(command, check=True, cwd=SPINN_ROOT)
        return json.loads((run_dir / "metrics.json").read_text())
