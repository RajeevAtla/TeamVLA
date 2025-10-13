"""Inverse kinematics and motion-planning utilities for TeamVLA."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def solve_ik(
    q_init: NDArray[np.float64],
    target_pose: NDArray[np.float64],
    *,
    limits: Mapping[str, NDArray[np.float64]] | None = None,
    max_iters: int = 100,
    atol: float = 1e-3,
) -> NDArray[np.float64]:
    """Compute a damped-least-squares IK solution (placeholder implementation).

    The returned configuration currently mirrors ``q_init`` while applying joint limits.
    A fully-fledged solver will replace the placeholder in a future phase.
    """

    q_current = _ensure_vector(q_init)
    if limits:
        q_current = _apply_limits(q_current, limits)
    _unused(target_pose, max_iters, atol)
    return q_current


def ee_pose_from_state(state: Mapping[str, Any], agent_id: int) -> NDArray[np.float64]:
    """Extract an end-effector pose from the environment state (placeholder)."""

    _unused(state)
    return np.zeros(7, dtype=np.float64) + float(agent_id)


def plan_rendezvous(
    ee_a: NDArray[np.float64],
    ee_b: NDArray[np.float64],
    obj_pose: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute mirrored grasp poses around an object (placeholder)."""

    center = (ee_a + ee_b + obj_pose) / 3.0
    return center.copy(), center.copy()


def gripper_command(open_ratio: float) -> float:
    """Map a normalized gripper ratio to actuator command space."""

    return float(np.clip(open_ratio, 0.0, 1.0))


def clamp_action(delta_q: NDArray[np.float64], max_norm: float) -> NDArray[np.float64]:
    """Clamp the joint delta vector to a maximum Euclidean norm."""

    vector = _ensure_vector(delta_q)
    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm == 0.0:
        return vector
    scale = max_norm / norm
    return vector * scale


def _ensure_vector(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a contiguous float64 vector copy of the input."""

    vector = np.asarray(array, dtype=np.float64).reshape(-1)
    return np.ascontiguousarray(vector)


def _apply_limits(
    joints: NDArray[np.float64], limits: Mapping[str, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Apply lower/upper bounds to the joint vector."""

    lower = limits.get("lower")
    upper = limits.get("upper")
    if lower is not None:
        joints = np.maximum(joints, np.asarray(lower, dtype=np.float64))
    if upper is not None:
        joints = np.minimum(joints, np.asarray(upper, dtype=np.float64))
    return joints


def _unused(*args: Any, **kwargs: Any) -> None:
    """Helper to silence unused parameter warnings in placeholders."""
