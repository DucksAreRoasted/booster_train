# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility helpers for the editable TienKung RSL-RL runner."""

from typing import Any

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .rsl_rl_observations import split_observation_result, split_observations


class TienKungRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Adapt Isaac Lab's grouped observations to TienKung's legacy runner API."""

    def reset(self) -> tuple[Any, dict]:
        return split_observation_result(super().reset())

    def get_observations(self) -> tuple[Any, dict]:
        return split_observation_result(super().get_observations())

    def step(self, actions) -> tuple[Any, Any, Any, dict]:
        observation_groups, rewards, dones, extras = super().step(actions)
        observations, extras = split_observations(observation_groups, extras)
        return observations, rewards, dones, extras
