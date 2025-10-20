"""Inverse kinematics and motion-planning utilities for TeamVLA."""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any
from typing import Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - only for static analysis
    from envs.sim_state import SimulationState

StateLike = Mapping[str, Any] | SimulationState

POSITION_DIMS = 3
POSE_DIMS = 7
DEFAULT_STEP = 0.05


def solve_ik(
    q_init: NDArray[np.float64],
    target_pose: NDArray[np.float64],
    *,
    limits: Mapping[str, NDArray[np.float64]] | None = None,
    max_iters: int = 100,
    atol: float = 1e-3,
) -> NDArray[np.float64]:
    """Iteratively nudge a configuration towards the target pose.

    This helper deliberately keeps the implementation lightweight: it treats ``q_init`` as a
    position vector (typically 3-DoF) and performs damped gradient steps toward the translation
    component of ``target_pose``. Joint limits are respected on every iteration and the solver
    terminates once the Euclidean error falls below ``atol``.
    """

    current = _ensure_vector(q_init)
    target = np.asarray(target_pose, dtype=np.float64).reshape(-1)
    goal = target[: current.shape[0]]
    for _ in range(max_iters):
        error = goal - current
        if np.linalg.norm(error) <= atol:
            break
        step = np.clip(error, -DEFAULT_STEP, DEFAULT_STEP)
        current = current + step
        if limits:
            current = _apply_limits(current, limits)
    return current


def ee_pose_from_state(state: StateLike, agent_id: int) -> NDArray[np.float64]:
    """Extract an end-effector pose for ``agent_id`` from a simulation state or observation."""

    position = _extract_position(state, agent_id)
    pose = np.zeros(POSE_DIMS, dtype=np.float64)
    pose[:POSITION_DIMS] = position
    pose[3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return pose


def plan_rendezvous(
    ee_a: NDArray[np.float64],
    ee_b: NDArray[np.float64],
    obj_pose: NDArray[np.float64],
    *,
    offset: float = 0.08,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return mirrored grasp targets around an object using the agents' approach vector."""

    ee_a = np.asarray(ee_a, dtype=np.float64).reshape(-1)
    ee_b = np.asarray(ee_b, dtype=np.float64).reshape(-1)
    obj = np.asarray(obj_pose, dtype=np.float64).reshape(-1)

    midpoint = (ee_a[:POSITION_DIMS] + ee_b[:POSITION_DIMS]) * 0.5
    approach = obj[:POSITION_DIMS] - midpoint
    if np.linalg.norm(approach) < 1e-6:
        approach = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    approach /= np.linalg.norm(approach)
    lateral = np.array([-approach[1], approach[0], 0.0], dtype=np.float64)
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    lateral /= np.linalg.norm(lateral)

    target_a = obj[:POSITION_DIMS] - lateral * offset
    target_b = obj[:POSITION_DIMS] + lateral * offset
    pose_a = np.zeros(POSE_DIMS, dtype=np.float64)
    pose_b = np.zeros(POSE_DIMS, dtype=np.float64)
    pose_a[:POSITION_DIMS] = target_a
    pose_b[:POSITION_DIMS] = target_b
    pose_a[3:] = pose_b[3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return pose_a, pose_b


def gripper_command(open_ratio: float) -> float:
    """Map a normalized gripper ratio to actuator command space."""

    return float(np.clip(open_ratio, 0.0, 1.0))


def clamp_action(delta_q: NDArray[np.float64], max_norm: float) -> NDArray[np.float64]:
    """Clamp the action vector to a maximum Euclidean norm."""

    vector = _ensure_vector(delta_q)
    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)


# ---------------------------------------------------------------------------#
# Internal helpers                                                           #
# ---------------------------------------------------------------------------#


def _extract_position(state: Mapping[str, Any], agent_id: int) -> NDArray[np.float64]:
    """Support extracting positions from SimulationState, observations, or dicts."""

    key = f"agent_{agent_id}"
    if hasattr(state, "arms") and key in state.arms:  # type: ignore[attr-defined]
        position = state.arms[key].position  # type: ignore[index]
        return np.asarray(position, dtype=np.float64)
    if "arms" in state and key in state["arms"]:
        position = state["arms"][key].position  # type: ignore[index]
        return np.asarray(position, dtype=np.float64)
    robot_state = None
    if isinstance(state, MutableMapping):
        robot_state = state.get("robot_state")
    if robot_state and "position" in robot_state:
        return np.asarray(robot_state["position"], dtype=np.float64)
    raise KeyError(f"Unable to locate pose information for agent {agent_id}.")


def _ensure_vector(array: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _apply_limits(
    joints: NDArray[np.float64], limits: Mapping[str, NDArray[np.float64]]
) -> NDArray[np.float64]:
    lower = limits.get("lower")
    upper = limits.get("upper")
    if lower is not None:
        joints = np.maximum(joints, np.asarray(lower, dtype=np.float64))
    if upper is not None:
        joints = np.minimum(joints, np.asarray(upper, dtype=np.float64))
    return joints
