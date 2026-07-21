"""The interface every benchmarked method implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..task import DesignTask, MethodResult


class ShapeMethod(ABC):
    """A design method that can report an objective and a shape gradient.

    Implementations should be cheap to construct; all real work (training,
    meshing, solving) happens inside :meth:`evaluate` so the driver can time it
    and so an unavailable backend can report ``available=False`` without raising.
    """

    #: Short, stable identifier used as the scorecard row label.
    name: str

    #: Whether this method's shape gradient is the trusted reference that other
    #: methods are scored against. Exactly one method in a run sets this True.
    is_reference: bool = False

    @abstractmethod
    def evaluate(self, task: DesignTask) -> MethodResult:
        """Solve ``task`` and return objective, gradient, and timings."""
