"""Scripted policy scaffold for the bimanual drawer task."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from control import ik_utils
from control.phase_machine import PhaseMachine
from envs.core_env import NewtonMAEnv


def waypoint_library(cfg: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Return placeholder drawer pull waypoints."""

    pull_distance = float(cfg.get("drawer_pull", 0.15))
    return {
        "handles_left": np.array([-0.1, 0.3, 0.15], dtype=np.float64),
        "handles_right": np.array([0.1, 0.3, 0.15], dtype=np.float64),
        "pull_target": np.array([0.0, 0.3 + pull_distance, 0.15], dtype=np.float64),
    }


def scripted_policy(
    env: NewtonMAEnv,
    phase_machine: PhaseMachine,
    obs: Sequence[Mapping[str, Any]],
) -> list[NDArray[np.float64]]:
    """Return placeholder coordinated actions for the drawer task."""

    _unused(env, obs)
    phase_machine.step({"phase_complete": False})
    base = np.array([0.01, 0.0, 0.0, -0.01, 0.0, 0.0, 0.01], dtype=np.float64)
    return [ik_utils.clamp_action(base, 0.04), ik_utils.clamp_action(base * -1.0, 0.04)]


def _unused(*_: Any) -> None:
    """Helper to silence unused argument warnings."""
