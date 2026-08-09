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
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.locomotion.env_cfg import (  # noqa: E402
    FlatEnvCfg,
    PlayFlatEnvCfg,
    RoughEnvCfg,
)
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.locomotion.ppo_cfg import PPORunnerCfg  # noqa: E402


def test_locomotion_config_variants():
    """The public configs expose the documented training and play behavior."""
    flat = FlatEnvCfg()
    rough = RoughEnvCfg()
    play = PlayFlatEnvCfg()
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

    assert play.scene.num_envs == 50
    assert play.events.push_robot is None
    assert not play.observations.policy.enable_corruption

    assert runner.experiment_name == "k1_locomotion"
    assert runner.max_iterations == 50000
    assert runner.policy.init_noise_std == 0.8
    assert runner.algorithm.entropy_coef == 0.01


def test_flat_environment_steps_without_crashing():
    """A small flat environment can reset and advance for one hundred policy steps."""
    cfg = FlatEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = "cpu"
    cfg.commands.base_velocity.debug_vis = False

    env = gym.make("Booster-K1-Locomotion-Flat-v0", cfg=cfg)
    try:
        env.reset()
        action = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        for _ in range(100):
            env.step(action)
        print("K1_LOCOMOTION_SMOKE_OK envs=4 steps=100", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        test_locomotion_config_variants()
        test_flat_environment_steps_without_crashing()
    finally:
        simulation_app.close()
