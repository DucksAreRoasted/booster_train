"""Behavior tests for the K1 rough-terrain curriculum."""

import importlib.util
import torch
from pathlib import Path
from types import SimpleNamespace

CURRICULUM_PATH = (
    Path(__file__).parents[1]
    / "source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1/locomotion/curriculums.py"
)


def _load_curriculum_module():
    spec = importlib.util.spec_from_file_location("k1_locomotion_curriculums", CURRICULUM_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tracking_curriculum_promotes_good_episodes_and_demotes_failures():
    """Curriculum decisions depend on task quality and survival, not net displacement."""
    curriculum = _load_curriculum_module()

    move_up, move_down = curriculum.classify_terrain_progress(
        lin_track_score=torch.tensor([0.90, 0.90, 0.40]),
        ang_track_score=torch.tensor([0.85, 0.85, 0.90]),
        failed=torch.tensor([False, True, False]),
        valid_episode=torch.tensor([True, True, True]),
    )

    assert move_up.tolist() == [True, False, False]
    assert move_down.tolist() == [False, True, True]


def test_tracking_curriculum_updates_terrain_from_normalized_episode_rewards():
    """The Isaac Lab adapter updates levels and exposes useful training diagnostics."""
    curriculum = _load_curriculum_module()

    class FakeTerrain:
        def __init__(self):
            self.terrain_levels = torch.tensor([0, 2, 1])

        def update_env_origins(self, env_ids, move_up, move_down):
            self.terrain_levels[env_ids] += move_up.to(torch.long) - move_down.to(torch.long)

    class FakeRewardManager:
        _episode_sums = {
            "track_lin_vel_xy": torch.tensor([36.0, 16.0, 0.0]),
            "track_ang_vel_z": torch.tensor([18.0, 18.0, 0.0]),
        }

        @staticmethod
        def get_term_cfg(name):
            return SimpleNamespace(weight=2.0 if name == "track_lin_vel_xy" else 1.0)

    terrain = FakeTerrain()
    env = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        reward_manager=FakeRewardManager(),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, False])),
        episode_length_buf=torch.tensor([1000, 1000, 0]),
        max_episode_length_s=20.0,
    )

    diagnostics = curriculum.terrain_levels_track(env, torch.tensor([0, 1, 2]))

    assert terrain.terrain_levels.tolist() == [1, 1, 1]
    assert diagnostics == {
        "mean_level": torch.tensor(1.0),
        "move_up_rate": torch.tensor(0.5),
        "move_down_rate": torch.tensor(0.5),
        "lin_track_score": torch.tensor(0.65),
        "ang_track_score": torch.tensor(0.9),
    }
