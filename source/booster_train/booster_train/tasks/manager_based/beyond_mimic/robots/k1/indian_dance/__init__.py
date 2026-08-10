# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# 训练环境：使用 RoughEnvCfg（完整观测，含 motion_anchor_pos_b + base_lin_vel）
#
# 注意：如果换回 WoStateEstimation 配置（缺位置观测），
# 会导致 Policy 无法感知全局位置误差，机器人位置跟踪效果差。
# 详见 env_cfg.py 中 RoughEnvCfg 的注释。
# ---------------------------------------------------------------------------
gym.register(
    id="Booster-K1-indian-dance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# 推理/演示环境：使用 PlayFlatEnvCfg（完整观测，观测维度与训练一致）
# ---------------------------------------------------------------------------
gym.register(
    id="Booster-K1-indian-dance-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)