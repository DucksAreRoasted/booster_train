# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import RayCaster


def local_terrain_height(ray_hit_heights: torch.Tensor) -> torch.Tensor:
    """Estimate the supporting ground height without reacting to isolated obstacles.

    The median represents the terrain plane beneath the scanner on slopes and
    waves, while keeping a nearby step or obstacle from falsely raising the
    robot's minimum-clearance reference.
    """
    return torch.median(ray_hit_heights, dim=1).values


def root_height_below_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when root clearance above the locally scanned terrain is too low."""
    asset: RigidObject = env.scene[asset_cfg.name]
    height_scanner: RayCaster = env.scene[sensor_cfg.name]
    terrain_height = local_terrain_height(height_scanner.data.ray_hits_w[..., 2])
    return asset.data.root_pos_w[:, 2] - terrain_height < minimum_height
