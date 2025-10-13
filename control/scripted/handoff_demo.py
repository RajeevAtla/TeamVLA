"""Scripted policy scaffold for the hand-off task."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from control import ik_utils
from control.phase_machine import PhaseMachine
from envs.core_env import NewtonMAEnv


def waypoint_library(cfg: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Return placeholder rendezvous waypoints for the hand-off task."""

    radius = float(cfg.get("handoff_radius", 0.1))
    return {
        "approach_a": np.array([radius, 0.0, 0.2], dtype=np.float64),
        "approach_b": np.array([-radius, 0.0, 0.2], dtype=np.float64),
        "handover": np.zeros(3, dtype=np.float64),
    }


def scripted_policy(
    env: NewtonMAEnv,
    phase_machine: PhaseMachine,
    obs: Sequence[Mapping[str, Any]],
) -> list[NDArray[np.float64]]:
    """Return placeholder coordinated actions for the hand-off task."""

    _unused(env, obs)
    phase_machine.step({"phase_complete": False})
    base = np.linspace(-0.02, 0.02, 7, dtype=np.float64)
    action_a = ik_utils.clamp_action(base, max_norm=0.05)
    action_b = ik_utils.clamp_action(-base, max_norm=0.05)
    return [action_a, action_b]


def _unused(*_: Any) -> None:
    """Helper to silence unused argument warnings."""
