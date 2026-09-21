"""End-to-end smoke: a few steps of fixed and joint training for every backend
run without error and produce finite eta."""

import math

import pytest

from spinn.config import Config
from spinn.train.train_fixed import train_fixed
from spinn.train.train_joint import train_joint


def _tiny(config: Config) -> Config:
    config.sampling.interior_points = 256
    config.sampling.electrode_boundary_points_per_electrode = 32
    config.sampling.outer_boundary_points = 64
    config.sampling.resample_every = 5
    config.shape_network.identity_pretrain_steps = 10
    config.validation.fem_grid = 60
    config.validation.fem_checkpoint_every = 4
    return config


@pytest.mark.parametrize("backend", ["mlp", "pixel", "pig"])
def test_fixed_smoke(tmp_path, backend):
    config = Config()
    config.physics_backend.type = backend
    config = _tiny(config)
    result = train_fixed(config, steps=6, results_root=tmp_path)
    assert math.isfinite(result["eta_pinn"]["eta_fit"])
    assert math.isfinite(result["eta_fem"])


@pytest.mark.parametrize("backend", ["mlp", "pixel", "pig"])
def test_joint_smoke(tmp_path, backend):
    config = Config()
    config.physics_backend.type = backend
    config = _tiny(config)
    result = train_joint(config, steps=6, results_root=tmp_path)
    assert math.isfinite(result["final_eta_pinn"]["eta_fit"])
    assert math.isfinite(result["final_eta_fem"])
