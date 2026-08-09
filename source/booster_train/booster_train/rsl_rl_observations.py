# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation conversion shared by the local RSL-RL compatibility wrapper."""

from typing import Any


def split_observations(observation_groups: Any, extras: dict) -> tuple[Any, dict]:
    """Return legacy observations unchanged or convert grouped observations."""
    if isinstance(observation_groups, tuple) and len(observation_groups) == 2 and not extras:
        observations, legacy_extras = observation_groups
        return split_observations(observations, legacy_extras)

    if not hasattr(observation_groups, "items"):
        if "observations" not in extras:
            raise KeyError("Legacy RSL-RL observations require extras['observations'].")
        return observation_groups, extras

    if "policy" not in observation_groups:
        raise KeyError("RSL-RL observations must contain a 'policy' group.")

    legacy_extras = dict(extras)
    extra_observations = dict(legacy_extras.get("observations", {}))
    extra_observations.update(
        (name, observations) for name, observations in observation_groups.items() if name != "policy"
    )
    legacy_extras["observations"] = extra_observations
    return observation_groups["policy"], legacy_extras


def split_observation_result(result: Any) -> tuple[Any, dict]:
    """Normalize either Isaac Lab wrapper API to the legacy RSL-RL result."""
    if isinstance(result, tuple) and len(result) == 2:
        observations, extras = result
        return split_observations(observations, extras)
    return split_observations(result, {})
