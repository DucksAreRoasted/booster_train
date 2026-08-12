# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab runtime checks for K1 locomotion configuration objects."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cpu")
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from booster_train.rsl_rl_compat import TienKungRslRlVecEnvWrapper  # noqa: E402
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.locomotion.env_cfg import (  # noqa: E402
    FlatEnvCfg,
    PlayFlatEnvCfg,
    PlayRoughEnvCfg,
    RoughEnvCfg,
)
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.locomotion.ppo_cfg import PPORunnerCfg  # noqa: E402


def test_locomotion_config_variants():
    """The public configs expose the documented training and play behavior."""
    flat = FlatEnvCfg()
    rough = RoughEnvCfg()
    play = PlayFlatEnvCfg()
    play_rough = PlayRoughEnvCfg()
    runner = PPORunnerCfg()

    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.scene.height_scanner is not None
    assert flat.observations.policy.height_scan is not None
    assert len(flat.actions.joint_pos.joint_names) == 6
    assert flat.commands.base_velocity.ranges.lin_vel_x == (-0.5, 0.8)

    assert rough.scene.terrain.terrain_type == "generator"
    assert rough.scene.height_scanner is not None
    assert rough.observations.policy.height_scan is not None
    assert rough.scene.terrain.terrain_generator.curriculum
    assert len(rough.scene.terrain.terrain_generator.sub_terrains) == 5
    assert rough.observations.policy.height_scan.noise.n_min == -0.02
    assert rough.observations.policy.height_scan.noise.n_max == 0.02
    assert rough.observations.critic.height_scan.noise.n_min == -0.02
    assert rough.observations.critic.height_scan.noise.n_max == 0.02
    assert rough.rewards.action_rate_l2.weight == -0.05
    assert rough.commands.base_velocity.resampling_time_range == (20.0, 20.0)
    assert rough.curriculum.terrain_levels.func.__name__ == "terrain_levels_track"

    assert flat.observations.policy.height_scan.noise.n_min == -0.1
    assert flat.rewards.action_rate_l2.weight == -0.5
    assert flat.commands.base_velocity.resampling_time_range == (5.0, 10.0)

    assert play.scene.num_envs == 50
    assert play.events.push_robot is None
    assert not play.observations.policy.enable_corruption

    assert play_rough.scene.num_envs == 50
    assert play_rough.scene.terrain.max_init_terrain_level == 0
    assert play_rough.scene.terrain.terrain_generator.seed == 42
    assert play_rough.events.distribute_terrain_levels.func.__name__ == "distribute_terrain_levels"
    assert play_rough.curriculum.terrain_levels is None
    assert play_rough.events.push_robot is None
    assert not play_rough.observations.policy.enable_corruption

    assert runner.experiment_name == "k1_locomotion"
    assert runner.max_iterations == 50000
    assert runner.clip_actions == 1.0
    assert runner.load_optimizer is False
    assert runner.to_dict()["load_optimizer"] is False
    assert runner.policy.init_noise_std == 0.8
    assert runner.algorithm.entropy_coef == 0.0


def test_rough_environment_steps_with_tracking_curriculum_and_clipped_actions():
    """The rough task runs its custom curriculum while bounding policy actions."""
    cfg = RoughEnvCfg()
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.episode_length_s = 0.04
    cfg.sim.device = "cpu"
    cfg.commands.base_velocity.debug_vis = False
    cfg.events.push_robot = None

    env = gym.make("Booster-K1-Locomotion-Rough-v0", cfg=cfg)
    wrapped_env = TienKungRslRlVecEnvWrapper(env, clip_actions=1.0)
    try:
        assert env.unwrapped.curriculum_manager.active_terms == ["terrain_levels"]
        oversized_action = torch.full(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            1000.0,
            device=env.unwrapped.device,
        )
        wrapped_env.step(oversized_action)
        assert torch.max(torch.abs(env.unwrapped.action_manager.action)).item() <= 1.0

        action = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        curriculum_log = None
        for _ in range(5):
            _, _, _, extras = wrapped_env.step(action)
            log = extras.get("log", {})
            if "Curriculum/terrain_levels/mean_level" in log:
                curriculum_log = log
                break
        assert curriculum_log is not None
        assert "Curriculum/terrain_levels/move_up_rate" in curriculum_log
        assert "Curriculum/terrain_levels/move_down_rate" in curriculum_log
        print("K1_ROUGH_LOCOMOTION_CURRICULUM_SMOKE_OK", flush=True)
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        test_locomotion_config_variants()
        test_rough_environment_steps_with_tracking_curriculum_and_clipped_actions()
    finally:
        simulation_app.close()
