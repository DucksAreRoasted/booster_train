# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curricula for K1 velocity tracking on rough terrain."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def classify_terrain_progress(
    lin_track_score: torch.Tensor,
    ang_track_score: torch.Tensor,
    failed: torch.Tensor,
    valid_episode: torch.Tensor,
    move_up_threshold: float = 0.75,
    move_down_threshold: float = 0.55,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify completed episodes for terrain progression.

    Tracking scores are normalized to ``[0, 1]`` by the curriculum adapter.
    A neutral band prevents terrain levels from oscillating around one cutoff.
    """
    move_up = (
        valid_episode
        & ~failed
        & (lin_track_score >= move_up_threshold)
        & (ang_track_score >= move_up_threshold)
    )
    move_down = valid_episode & (
        failed | (lin_track_score < move_down_threshold) | (ang_track_score < move_down_threshold)
    )
    move_down &= ~move_up
    return move_up, move_down


def terrain_levels_track(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    lin_reward_name: str = "track_lin_vel_xy",
    ang_reward_name: str = "track_ang_vel_z",
    move_up_threshold: float = 0.75,
    move_down_threshold: float = 0.55,
) -> dict[str, torch.Tensor]:
    """Progress terrain from normalized tracking quality and episode survival.

    Unlike Isaac Lab's distance curriculum, this remains valid for lateral and
    turning commands because it evaluates how closely the policy followed the
    command instead of measuring net displacement from the episode origin.
    """
    reward_manager = env.reward_manager
    terrain = env.scene.terrain

    def normalized_score(reward_name: str) -> torch.Tensor:
        weight = reward_manager.get_term_cfg(reward_name).weight
        if weight <= 0.0:
            raise ValueError(f"Curriculum reward '{reward_name}' must have a positive weight.")
        score = reward_manager._episode_sums[reward_name][env_ids]
        score = score / (weight * env.max_episode_length_s)
        return torch.clamp(score, 0.0, 1.0)

    lin_track_score = normalized_score(lin_reward_name)
    ang_track_score = normalized_score(ang_reward_name)
    failed = env.termination_manager.terminated[env_ids]
    valid_episode = env.episode_length_buf[env_ids] > 0

    move_up, move_down = classify_terrain_progress(
        lin_track_score,
        ang_track_score,
        failed,
        valid_episode,
        move_up_threshold,
        move_down_threshold,
    )
    terrain.update_env_origins(env_ids, move_up, move_down)

    valid_count = valid_episode.float().sum().clamp_min(1.0)
    return {
        "mean_level": torch.mean(terrain.terrain_levels.float()),
        "move_up_rate": move_up.float().sum() / valid_count,
        "move_down_rate": move_down.float().sum() / valid_count,
        "lin_track_score": (lin_track_score * valid_episode).sum() / valid_count,
        "ang_track_score": (ang_track_score * valid_episode).sum() / valid_count,
    }
