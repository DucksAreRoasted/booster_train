# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from booster_train.assets.robots.booster import BOOSTER_K1_LOCOMOTION_CFG as ROBOT_CFG
from booster_train.assets.robots.booster import K1_ACTION_SCALE

from .tracking_env_cfg import TrackingEnvCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4,
            noise_range=(0.0, 0.008),
            noise_step=0.005,
            border_width=0.25,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.05, 0.25),
            platform_width=1.5,
            border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.1, 0.4),
            obstacle_height_range=(0.02, 0.05),
            num_obstacles=8,
            platform_width=1.0,
            border_width=0.25,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.01, 0.04),
            num_waves=3,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(-0.02, 0.02),
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    """K1 locomotion training on a plane with checkpoint-compatible observations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = {
            joint_pattern: K1_ACTION_SCALE[joint_pattern] for joint_pattern in self.actions.joint_pos.joint_names
        }
        self.curriculum.terrain_levels = None


@configclass
class RoughEnvCfg(FlatEnvCfg):
    """K1 locomotion on curriculum-generated terrain with 97 observations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 0
        # The terrain features are mostly 2--5 cm tall. Keep height-scan noise
        # below that signal so the policy can distinguish obstacles while still
        # learning robustness to sensor error.
        self.observations.policy.height_scan.noise = Unoise(n_min=-0.02, n_max=0.02)
        self.observations.critic.height_scan.noise = Unoise(n_min=-0.02, n_max=0.02)
        # terrain_levels_vel judges progress from net displacement over the
        # episode. Resampling direction every 5--10 seconds can punish a policy
        # that tracks commands correctly but returns toward its start position.
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        # Rough terrain needs enough freedom to adjust foot placement. Retain a
        # meaningful smoothness cost without letting it dominate tracking.
        self.rewards.action_rate_l2.weight = -0.05
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class PlayFlatEnvCfg(FlatEnvCfg):
    """Small deterministic flat scene for policy evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.push_robot = None
