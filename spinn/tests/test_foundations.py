import numpy as np

from spinn.config import Config
from spinn.foundations.audit_current_architecture import run_audit
from spinn.foundations.problems import canonical_four_rod, dale_four_rod
from spinn.geometry.four_rod import build_trap


def test_canonical_problem_matches_active_geometry():
    problem = canonical_four_rod()
    trap = build_trap(Config())
    assert problem.r0 == trap.r0
    assert problem.voltage_span == trap.V0
    for declared, active in zip(problem.electrodes, trap.electrodes()):
        assert declared.name == active.name
        assert np.allclose(declared.center, (active.cx, active.cy))
        assert declared.radius == active.radius
        assert declared.voltage == trap.electrode_voltage(active)


def test_dale_problem_preserves_original_defaults():
    problem = dale_four_rod()
    assert problem.outer_kind == "rectangle"
    assert problem.outer_size == (2.5, 2.0)
    assert problem.r0 == 0.075
    assert [e.voltage for e in problem.electrodes] == [0.0, 0.0, 300.0, 300.0]


def test_current_reference_eta_is_not_coordinate_invariant():
    audit = run_audit()
    # The deformation mask must leave the physics-measurement core fixed.
    assert audit["design_mask"]["origin"] == 0.0
    baseline, warped = (
        audit["coordinate_gauge_test"][0],
        audit["coordinate_gauge_test"][-1],
    )
    assert baseline["physical_eta"] == warped["physical_eta"] == 1.0
    assert warped["boundary_motion_at_r1"] == 0.0
    assert abs(warped["composed_reference_eta"] - 1.44) < 1e-10
