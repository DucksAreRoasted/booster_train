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

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.8
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.activation = "elu"
        self.algorithm.entropy_coef = 0.01
