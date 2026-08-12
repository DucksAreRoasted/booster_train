# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    experiment_name = "k1_locomotion"
    num_steps_per_env = 24
    save_interval = 1000
    # Joint-position actions are normalized around the default pose. Bounding
    # them prevents rare out-of-distribution observations from producing huge
    # action-rate penalties and corrupting the value targets.
    clip_actions = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.8
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.activation = "elu"
        # The environment clips sampled actions to [-1, 1]. An entropy bonus
        # can otherwise increase the Gaussian standard deviation without the
        # environment observing correspondingly larger actions.
        self.algorithm.entropy_coef = 0.0
