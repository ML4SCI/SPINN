"""Configuration inheritance, overrides, and validation."""

from pathlib import Path

import pytest

from spinn.config import Config, apply_overrides, load_config


@pytest.mark.parametrize(
    "filename,backend",
    [
        ("fixed_mlp.yaml", "mlp"),
        ("fixed_pixel.yaml", "pixel"),
        ("fixed_pig.yaml", "pig"),
        ("joint_mlp.yaml", "mlp"),
        ("joint_pixel.yaml", "pixel"),
        ("joint_pig.yaml", "pig"),
    ],
)
def test_shipped_configs_load(filename: str, backend: str):
    config = load_config(filename)
    assert config.physics_backend.type == backend
    assert config.sampling.interior_points > 0


def test_overrides_are_typed_and_do_not_mutate_input():
    original = Config()
    updated = apply_overrides(
        original, ["runtime.log_every=10", "runtime.deterministic=false"]
    )
    assert updated.runtime.log_every == 10
    assert updated.runtime.deterministic is False
    assert original.runtime.log_every == 200


def test_unknown_section_is_rejected(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text("mystery:\n  value: 1\n")
    with pytest.raises(ValueError, match="unknown configuration section"):
        load_config(path)


def test_cyclic_inheritance_is_rejected(tmp_path: Path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n")
    second.write_text("extends: first.yaml\n")
    with pytest.raises(ValueError, match="cyclic configuration inheritance"):
        load_config(first)
