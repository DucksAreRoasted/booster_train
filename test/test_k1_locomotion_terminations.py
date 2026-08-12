"""Behavior tests for terrain-relative K1 locomotion terminations."""

import importlib.util
from pathlib import Path

import torch


TERMINATIONS_PATH = (
    Path(__file__).parents[1]
    / "source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1/locomotion/terminations.py"
)


def _load_terminations_module():
    spec = importlib.util.spec_from_file_location("k1_locomotion_terminations", TERMINATIONS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_terrain_height_rejects_isolated_obstacles():
    """A nearby obstacle must not raise the reference plane below the trunk."""
    terminations = _load_terminations_module()
    ray_heights = torch.zeros((2, 49))
    ray_heights[0, 0] = 0.5
    ray_heights[1] = torch.linspace(-0.3, 0.3, 49)

    terrain_height = terminations.local_terrain_height(ray_heights)

    assert torch.allclose(terrain_height, torch.tensor([0.0, 0.0]), atol=1.0e-6)
