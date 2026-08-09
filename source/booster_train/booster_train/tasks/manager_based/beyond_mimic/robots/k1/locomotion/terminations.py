# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when root clearance above the locally scanned terrain is too low."""
    asset: RigidObject = env.scene[asset_cfg.name]
    height_scanner: RayCaster = env.scene[sensor_cfg.name]
    terrain_height = torch.mean(height_scanner.data.ray_hits_w[..., 2], dim=1)
    return asset.data.root_pos_w[:, 2] - terrain_height < minimum_height
