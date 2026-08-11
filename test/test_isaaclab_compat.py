"""Compatibility tests for supported Isaac Lab configuration APIs."""

import importlib.util
from pathlib import Path

COMPAT_PATH = Path(__file__).parents[1] / "source/booster_train/booster_train/isaaclab_compat.py"


def _load_compat_module():
    spec = importlib.util.spec_from_file_location("booster_isaaclab_compat", COMPAT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ray_caster_yaw_alignment_supports_legacy_isaac_lab():
    """Legacy RayCasterCfg receives attach_yaw_only instead of the renamed argument."""
    compat = _load_compat_module()

    class LegacyRayCasterCfg:
        def __init__(self, *, attach_yaw_only=False):
            self.attach_yaw_only = attach_yaw_only

    cfg = LegacyRayCasterCfg(**compat.ray_caster_yaw_alignment_kwargs(LegacyRayCasterCfg))

    assert cfg.attach_yaw_only is True


def test_ray_caster_yaw_alignment_supports_current_isaac_lab():
    """Current RayCasterCfg receives the explicit yaw alignment mode."""
    compat = _load_compat_module()

    class CurrentRayCasterCfg:
        def __init__(self, *, ray_alignment="base"):
            self.ray_alignment = ray_alignment

    cfg = CurrentRayCasterCfg(**compat.ray_caster_yaw_alignment_kwargs(CurrentRayCasterCfg))

    assert cfg.ray_alignment == "yaw"
