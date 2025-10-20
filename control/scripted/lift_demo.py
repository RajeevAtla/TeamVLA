"""Scripted policy for the lift-and-place task using the tabletop simulator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from control import ik_utils
from control.phase_machine import PhaseMachine
from envs.core_env import NewtonMAEnv

MAX_STEP = 0.06
LIFT_OFFSET = np.array([0.0, 0.0, 0.04], dtype=np.float64)
SIDE_OFFSET = np.array([0.1, 0.0, 0.02], dtype=np.float64)


def waypoint_library(cfg: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Return the static waypoints used by the scripted lifter."""

    height = float(cfg.get("lift_height", 0.32))
    return {
        "lift_height": np.array([0.0, 0.0, height], dtype=np.float64),
        "side_offset_a": -SIDE_OFFSET,
        "side_offset_b": SIDE_OFFSET,
    }


def scripted_policy(
    env: NewtonMAEnv,
    phase_machine: PhaseMachine,
    obs: Sequence[Mapping[str, Any]],
) -> list[NDArray[np.float64]]:
    """Compute coordinated lift-and-place actions for both agents."""

    del env  # The policy relies exclusively on observations.
    obs_a, obs_b = obs
    metadata = obs_a.get("metadata", {})
    payload = np.asarray(obs_a.get("objects", {}).get("payload", np.zeros(3)), dtype=np.float64)
    target_xy = np.asarray(metadata.get("target_xy", payload[:2]), dtype=np.float64)
    target_height = float(metadata.get("target_height", payload[2] + 0.15))

    phase = phase_machine.current()
    positions = [
        np.asarray(obs_a["robot_state"]["position"], dtype=np.float64),
        np.asarray(obs_b["robot_state"]["position"], dtype=np.float64),
    ]

    if phase == "reach":
        target_positions = [payload + SIDE_OFFSET * -1, payload + SIDE_OFFSET]
        grippers = [1.0, 1.0]
        complete = _both_within(positions, target_positions, thresh=0.05)
    elif phase == "grasp":
        target_positions = [payload + LIFT_OFFSET * 0.5, payload + LIFT_OFFSET * 0.5]
        grippers = [0.0, 0.0]
        complete = metadata.get("distance_a", 1.0) < 0.1 and metadata.get("distance_b", 1.0) < 0.1
    elif phase == "lift":
        lift_goal = payload.copy()
        lift_goal[2] = target_height
        target_positions = [lift_goal, lift_goal]
        grippers = [0.0, 0.0]
        complete = metadata.get("object_height", 0.0) >= target_height - 0.02
    elif phase == "place":
        place_goal = np.array([target_xy[0], target_xy[1], target_height], dtype=np.float64)
        target_positions = [place_goal, place_goal]
        grippers = [0.0, 0.0]
        complete = metadata.get("distance_to_target", 1.0) <= 0.05
    else:  # release
        place_goal = np.array([target_xy[0], target_xy[1], target_height], dtype=np.float64)
        target_positions = [place_goal, place_goal]
        grippers = [1.0, 1.0]
        complete = metadata.get("distance_to_target", 1.0) <= 0.05

    signals = {
        "phase_complete": bool(complete),
        "collision": bool(metadata.get("collision")),
        "task_success": bool(metadata.get("task_success")),
    }

    actions = [
        _action_towards(positions[0], target_positions[0], grippers[0]),
        _action_towards(positions[1], target_positions[1], grippers[1]),
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


def _both_within(
    positions: Sequence[NDArray[np.float64]],
    targets: Sequence[NDArray[np.float64]],
    *,
    thresh: float,
) -> bool:
    return all(
        np.linalg.norm(pos - tgt) <= thresh for pos, tgt in zip(positions, targets, strict=True)
    )
