# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Small compatibility adapters for supported Isaac Lab releases."""

from __future__ import annotations

import inspect


def ray_caster_yaw_alignment_kwargs(ray_caster_cfg_type: type) -> dict[str, object]:
    """Return the yaw-alignment argument supported by a RayCasterCfg type."""
    parameter_names = set(inspect.signature(ray_caster_cfg_type).parameters)
    parameter_names.update(getattr(ray_caster_cfg_type, "__dataclass_fields__", {}))

    if "ray_alignment" in parameter_names:
        return {"ray_alignment": "yaw"}
    if "attach_yaw_only" in parameter_names:
        return {"attach_yaw_only": True}
    raise TypeError("RayCasterCfg exposes neither 'ray_alignment' nor legacy 'attach_yaw_only'.")
