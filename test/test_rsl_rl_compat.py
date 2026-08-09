# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the local TienKung RSL-RL compatibility layer."""

import torch
import runpy
from pathlib import Path

OBSERVATION_HELPERS = (
    Path(__file__).parents[1]
    / "source/booster_train/booster_train/rsl_rl_observations.py"
)
split_observations = runpy.run_path(str(OBSERVATION_HELPERS))["split_observations"]


def test_split_observations_restores_legacy_runner_contract():
    """Policy observations stay positional while critic observations move to extras."""
    policy = torch.randn(4, 97)
    critic = torch.randn(4, 101)
    source_extras = {"log": {"episode": 3}}

    legacy_policy, legacy_extras = split_observations(
        {"policy": policy, "critic": critic}, source_extras
    )

    assert legacy_policy is policy
    assert legacy_extras["observations"]["critic"] is critic
    assert legacy_extras["log"] == {"episode": 3}
    assert "observations" not in source_extras


def test_split_observations_preserves_existing_observation_extras():
    """Environment-provided extras are merged instead of overwritten."""
    policy = torch.randn(2, 97)
    timeout_obs = torch.randn(2, 1)
    source_extras = {"observations": {"timeouts": timeout_obs}}

    _, legacy_extras = split_observations({"policy": policy}, source_extras)

    assert legacy_extras["observations"]["timeouts"] is timeout_obs
