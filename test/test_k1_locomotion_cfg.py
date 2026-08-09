# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the public K1 locomotion environment registrations."""

import ast
import gymnasium as gym
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

LOCOMOTION_TASK_IDS = {
    "Booster-K1-Locomotion-Flat-v0",
    "Booster-K1-Locomotion-Rough-v0",
    "Booster-K1-Locomotion-Flat-v0-Play",
}
LOCOMOTION_DIR = (
    Path(__file__).parents[1]
    / "source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1/locomotion"
)


def test_locomotion_tasks_are_registered():
    """All training and play variants are available through Gym."""
    runpy.run_path(str(LOCOMOTION_DIR / "__init__.py"))
    assert LOCOMOTION_TASK_IDS <= set(gym.registry)


def test_environment_variants_have_public_config_classes():
    """The registered entry points resolve to the three documented config classes."""
    module = ast.parse((LOCOMOTION_DIR / "env_cfg.py").read_text())
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    assert {"FlatEnvCfg", "RoughEnvCfg", "PlayFlatEnvCfg"} <= class_names


def test_locomotion_robot_has_exactly_twelve_movable_leg_joints():
    """The locomotion asset fixes the upper body and preserves every leg joint."""
    urdf = Path(__file__).parents[1] / "booster_assets/robots/K1/K1_locomotion.urdf"
    root = ET.parse(urdf).getroot()
    movable_joints = {joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"}
    expected_joints = {
        f"{side}_{joint}"
        for side in ("Left", "Right")
        for joint in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")
    }
    assert movable_joints == expected_joints


if __name__ == "__main__":
    test_locomotion_tasks_are_registered()
    test_environment_variants_have_public_config_classes()
    test_locomotion_robot_has_exactly_twelve_movable_leg_joints()
