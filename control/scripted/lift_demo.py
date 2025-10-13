"""Scripted policy scaffold for the lift-and-place task."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from control import ik_utils
from control.phase_machine import PhaseMachine
from envs.core_env import NewtonMAEnv


def waypoint_library(cfg: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Return placeholder waypoints for the lift task."""

    height = float(cfg.get("lift_height", 0.25))
    return {
        "pre_grasp": np.array([0.4, 0.0, height], dtype=np.float64),
        "grasp": np.array([0.4, 0.0, height - 0.05], dtype=np.float64),
        "place": np.array([0.6, 0.0, height], dtype=np.float64),
    }


def scripted_policy(
    env: NewtonMAEnv,
    phase_machine: PhaseMachine,
    obs: Sequence[Mapping[str, Any]],
) -> list[NDArray[np.float64]]:
    """Return placeholder coordinated actions for the lift task."""

    _unused(env, obs)
    phase_machine.step({"phase_complete": False})
    return [_idle_action(), _idle_action(scale=-1.0)]


def _idle_action(scale: float = 1.0) -> NDArray[np.float64]:
    vector = np.full(7, 0.01 * scale, dtype=np.float64)
    return ik_utils.clamp_action(vector, max_norm=0.05)


def _unused(*_: Any) -> None:
    """Helper to silence unused argument warnings."""
