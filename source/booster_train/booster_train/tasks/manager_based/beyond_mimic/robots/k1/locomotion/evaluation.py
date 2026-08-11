# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic terrain placement for K1 locomotion evaluation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def evaluation_terrain_levels(num_envs: int, num_levels: int, device: str | torch.device) -> torch.Tensor:
    """Assign consecutive environments across every terrain difficulty level."""
    if num_levels <= 0:
        raise ValueError("Evaluation requires at least one terrain level.")
    return torch.arange(num_envs, device=device) % num_levels


def distribute_terrain_levels(env: ManagerBasedRLEnv):
    """Place each terrain-type group deterministically across all levels."""
    terrain = env.scene.terrain
    levels = evaluation_terrain_levels(env.num_envs, terrain.max_terrain_level, env.device)
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
