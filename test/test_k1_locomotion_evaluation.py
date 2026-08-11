"""Behavior tests for deterministic K1 rough-terrain evaluation."""

import importlib.util
import torch
from pathlib import Path
from types import SimpleNamespace

EVALUATION_PATH = (
    Path(__file__).parents[1]
    / "source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1/locomotion/evaluation.py"
)


def _load_evaluation_module():
    spec = importlib.util.spec_from_file_location("k1_locomotion_evaluation", EVALUATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluation_assignment_covers_each_level_in_every_terrain_column():
    """Each five-environment terrain column contains levels zero through four."""
    evaluation = _load_evaluation_module()

    levels = evaluation.evaluation_terrain_levels(num_envs=10, num_levels=5, device="cpu")

    assert levels.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_evaluation_event_moves_environments_to_the_assigned_terrain_cells():
    """The startup event updates both level metadata and physical origins."""
    evaluation = _load_evaluation_module()
    terrain_origins = torch.arange(30, dtype=torch.float).reshape(5, 2, 3)
    terrain = SimpleNamespace(
        max_terrain_level=5,
        terrain_levels=torch.zeros(10, dtype=torch.long),
        terrain_types=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        terrain_origins=terrain_origins,
        env_origins=torch.zeros(10, 3),
    )
    env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), num_envs=10, device="cpu")

    evaluation.distribute_terrain_levels(env)

    assert terrain.terrain_levels.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert torch.equal(terrain.env_origins[0], terrain_origins[0, 0])
    assert torch.equal(terrain.env_origins[4], terrain_origins[4, 0])
    assert torch.equal(terrain.env_origins[5], terrain_origins[0, 1])
    assert torch.equal(terrain.env_origins[9], terrain_origins[4, 1])
