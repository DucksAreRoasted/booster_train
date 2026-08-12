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
    "Booster-K1-Locomotion-Rough-v0-Play",
}
LOCOMOTION_DIR = (
    Path(__file__).parents[1]
    / "source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1/locomotion"
)


def _class_assignments(module_path: Path, class_name: str) -> dict[str, ast.expr]:
    """Return simple class-level assignments from a configuration module."""
    module = ast.parse(module_path.read_text())
    class_node = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        target.id: node.value
        for node in class_node.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def test_locomotion_tasks_are_registered():
    """All training and play variants are available through Gym."""
    runpy.run_path(str(LOCOMOTION_DIR / "__init__.py"))
    assert LOCOMOTION_TASK_IDS <= set(gym.registry)


def test_environment_variants_have_public_config_classes():
    """The registered entry points resolve to the documented config classes."""
    module = ast.parse((LOCOMOTION_DIR / "env_cfg.py").read_text())
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    assert {"FlatEnvCfg", "RoughEnvCfg", "PlayFlatEnvCfg", "PlayRoughEnvCfg"} <= class_names


def test_locomotion_runner_bounds_policy_actions():
    """Training bounds normalized policy actions before they reach the robot and rewards."""
    assignments = _class_assignments(LOCOMOTION_DIR / "ppo_cfg.py", "PPORunnerCfg")
    assert ast.literal_eval(assignments["clip_actions"]) == 1.0


def test_locomotion_training_disables_entropy_pressure_on_clipped_actions():
    """Clipped locomotion actions do not receive an incentive for unbounded policy noise."""
    module = ast.parse((LOCOMOTION_DIR / "ppo_cfg.py").read_text())
    runner_cfg = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PPORunnerCfg")
    post_init = next(
        node for node in runner_cfg.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    entropy_assignment = next(
        node
        for node in post_init.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "self.algorithm.entropy_coef"
    )

    assert ast.literal_eval(entropy_assignment.value) == 0.0


def test_locomotion_resume_retains_optimizer_unless_transfer_is_requested():
    """Ordinary resumes keep Adam state; flat-to-rough transfers opt out through the CLI."""
    assignments = _class_assignments(LOCOMOTION_DIR / "ppo_cfg.py", "PPORunnerCfg")
    assert "load_optimizer" not in assignments


def test_locomotion_com_randomization_uses_project_compatibility_term():
    """CoM randomization remains available on Isaac Lab releases before 2.2."""
    assignments = _class_assignments(LOCOMOTION_DIR / "tracking_env_cfg.py", "EventCfg")
    event_call = assignments["base_com"]
    func = next(keyword.value for keyword in event_call.keywords if keyword.arg == "func")
    assert ast.unparse(func) == "beyond_mimic_mdp.randomize_rigid_body_com"


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
    test_locomotion_runner_bounds_policy_actions()
    test_locomotion_training_disables_entropy_pressure_on_clipped_actions()
    test_locomotion_resume_retains_optimizer_unless_transfer_is_requested()
    test_locomotion_com_randomization_uses_project_compatibility_term()
    test_locomotion_robot_has_exactly_twelve_movable_leg_joints()
