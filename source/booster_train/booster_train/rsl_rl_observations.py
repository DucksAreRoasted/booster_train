# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation conversion shared by the local RSL-RL compatibility wrapper."""

from collections.abc import Mapping
from typing import Any


def split_observations(observation_groups: Mapping[str, Any], extras: dict) -> tuple[Any, dict]:
    """Convert grouped observations to the legacy TienKung RSL-RL contract."""
    if "policy" not in observation_groups:
        raise KeyError("RSL-RL observations must contain a 'policy' group.")

    legacy_extras = dict(extras)
    extra_observations = dict(legacy_extras.get("observations", {}))
    extra_observations.update(
        (name, observations) for name, observations in observation_groups.items() if name != "policy"
    )
    legacy_extras["observations"] = extra_observations
    return observation_groups["policy"], legacy_extras
