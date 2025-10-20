"""Scripted policy for the hand-off task using deterministic heuristics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from control import ik_utils
from control.phase_machine import PhaseMachine
from envs.core_env import NewtonMAEnv

MAX_STEP = 0.06
APPROACH_OFFSET = np.array([0.0, 0.0, 0.06], dtype=np.float64)
HANDOFF_OFFSET = np.array([0.0, 0.02, 0.06], dtype=np.float64)


def waypoint_library(cfg: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Return the canonical waypoints leveraged by the hand-off controller."""

    radius = float(cfg.get("handoff_radius", 0.1))
    height = float(cfg.get("handoff_height", 0.22))
    return {
        "source_offset": np.array([0.0, 0.0, height - 0.1]),
        "handoff_offset": np.array([0.0, 0.0, height]),
        "radius": np.array([radius, 0.0, 0.0]),
    }


def scripted_policy(
    env: NewtonMAEnv,
    phase_machine: PhaseMachine,
    obs: Sequence[Mapping[str, Any]],
) -> list[NDArray[np.float64]]:
    """Compute cooperative hand-off actions for both agents."""

    del env
    obs_a, obs_b = obs
    metadata = obs_a.get("metadata", {})
    baton = np.asarray(obs_a.get("objects", {}).get("baton", np.zeros(3)), dtype=np.float64)
    source = np.asarray(metadata.get("source", baton), dtype=np.float64)
    handoff = np.asarray(metadata.get("handoff", baton), dtype=np.float64)
    target = np.asarray(metadata.get("target", baton), dtype=np.float64)

    phase = phase_machine.current()
    positions = [
        np.asarray(obs_a["robot_state"]["position"], dtype=np.float64),
        np.asarray(obs_b["robot_state"]["position"], dtype=np.float64),
    ]

    if phase == "reach_source":
        targets = [source + APPROACH_OFFSET, handoff + APPROACH_OFFSET]
        grippers = [1.0, 1.0]
        complete = (
            metadata.get("distance_source", 1.0) < 0.12
            and np.linalg.norm(positions[1] - targets[1]) < 0.12
        )
    elif phase == "grasp_source":
        targets = [source + APPROACH_OFFSET * 0.5, handoff + APPROACH_OFFSET]
        grippers = [0.0, 1.0]
        complete = (
            metadata.get("distance_source", 1.0) < 0.08
            and metadata.get("distance_handoff", 1.0) < 0.15
        )
    elif phase == "handover":
        pose_a, pose_b = ik_utils.plan_rendezvous(positions[0], positions[1], handoff)
        targets = [pose_a[:3] + HANDOFF_OFFSET * 0.0, pose_b[:3] + HANDOFF_OFFSET * 0.0]
        grippers = [0.0, 0.0]
        complete = metadata.get("distance_handoff", 1.0) < 0.08
    elif phase == "grasp_target":
        targets = [target + APPROACH_OFFSET, target + APPROACH_OFFSET * 0.5]
        grippers = [1.0, 0.0]
        complete = metadata.get("distance_target", 1.0) < 0.1
    else:  # release
        targets = [target + APPROACH_OFFSET, target + APPROACH_OFFSET]
        grippers = [1.0, 1.0]
        complete = metadata.get("distance_target", 1.0) < 0.05

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
