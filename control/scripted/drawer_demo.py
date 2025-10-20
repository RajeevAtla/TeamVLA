"""Scripted policy for the bimanual drawer task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from control import ik_utils
from control.phase_machine import PhaseMachine
from envs.core_env import NewtonMAEnv

MAX_STEP = 0.05
HANDLE_APPROACH = np.array([0.0, 0.04, 0.04], dtype=np.float64)


def waypoint_library(cfg: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Return the task-specific drawer waypoints used by the scripted policy."""

    pull_distance = float(cfg.get("drawer_pull", 0.22))
    return {
        "approach_offset": HANDLE_APPROACH,
        "pull_distance": np.array([0.0, pull_distance, 0.0]),
    }


def scripted_policy(
    env: NewtonMAEnv,
    phase_machine: PhaseMachine,
    obs: Sequence[Mapping[str, Any]],
) -> list[NDArray[np.float64]]:
    """Produce complementary actions that open the drawer and release synchronously."""

    del env
    obs_a, obs_b = obs
    metadata = obs_a.get("metadata", {})
    handles = obs_a.get("objects", {})

    handle_left = np.asarray(
        handles.get("drawer_left", np.array([-0.14, -0.12, 0.1])), dtype=np.float64
    )
    handle_right = np.asarray(
        handles.get("drawer_right", np.array([0.14, -0.12, 0.1])), dtype=np.float64
    )

    phase = phase_machine.current()
    positions = [
        np.asarray(obs_a["robot_state"]["position"], dtype=np.float64),
        np.asarray(obs_b["robot_state"]["position"], dtype=np.float64),
    ]

    if phase == "reach_handles":
        targets = [handle_left + HANDLE_APPROACH, handle_right + HANDLE_APPROACH]
        grippers = [1.0, 1.0]
        distances = [metadata.get("distance_left", 1.0), metadata.get("distance_right", 1.0)]
        complete = max(distances) < 0.12
    elif phase == "grasp":
        targets = [handle_left + HANDLE_APPROACH * 0.3, handle_right + HANDLE_APPROACH * 0.3]
        grippers = [0.0, 0.0]
        complete = (
            metadata.get("distance_left", 1.0) < 0.08 and metadata.get("distance_right", 1.0) < 0.08
        )
    elif phase == "pull":
        pull_offset = np.array(
            [0.0, metadata.get("drawer_extension", 0.0) + 0.05, 0.0], dtype=np.float64
        )
        targets = [
            handle_left + HANDLE_APPROACH * 0.2 + pull_offset,
            handle_right + HANDLE_APPROACH * 0.2 + pull_offset,
        ]
        grippers = [0.0, 0.0]
        complete = metadata.get("drawer_extension", 0.0) >= 0.18
    elif phase == "hold":
        pull_offset = np.array([0.0, metadata.get("drawer_extension", 0.0), 0.0], dtype=np.float64)
        targets = [
            handle_left + HANDLE_APPROACH * 0.2 + pull_offset,
            handle_right + HANDLE_APPROACH * 0.2 + pull_offset,
        ]
        grippers = [0.0, 0.0]
        complete = metadata.get("drawer_extension", 0.0) >= 0.2
    else:  # release
        pull_offset = np.array([0.0, metadata.get("drawer_extension", 0.0), 0.0], dtype=np.float64)
        targets = [
            handle_left + HANDLE_APPROACH * 0.2 + pull_offset,
            handle_right + HANDLE_APPROACH * 0.2 + pull_offset,
        ]
        grippers = [1.0, 1.0]
        complete = metadata.get("drawer_extension", 0.0) >= 0.2

    signals = {
        "phase_complete": bool(complete),
        "collision": bool(metadata.get("collision")),
        "task_success": bool(metadata.get("task_success")),
    }
    actions = [
        _action_towards(positions[0], targets[0], grippers[0]),
        _action_towards(positions[1], targets[1], grippers[1]),
    ]
    phase_machine.step(signals)
    return actions


def _action_towards(
    current: NDArray[np.float64], target: NDArray[np.float64], grip: float
) -> NDArray[np.float64]:
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = target
    solved = ik_utils.solve_ik(current, pose)
    delta = solved - current
    action = np.concatenate([delta[:3], [ik_utils.gripper_command(grip)]])
    return ik_utils.clamp_action(action, MAX_STEP)
