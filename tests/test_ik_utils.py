"""Tests for control.ik_utils placeholders."""

from __future__ import annotations

import numpy as np

from control import ik_utils


def test_solve_ik_respects_limits() -> None:
    q = np.array([0.5, -0.5])
    limits = {"lower": np.array([0.0, -1.0]), "upper": np.array([1.0, 0.0])}
    solved = ik_utils.solve_ik(q, target_pose=np.zeros(6), limits=limits)
    assert solved[0] == 0.5
    assert solved[1] == -0.5


def test_clamp_action_limits_norm() -> None:
    delta = np.array([1.0, 0.0, 0.0])
    clamped = ik_utils.clamp_action(delta, max_norm=0.1)
    assert np.isclose(np.linalg.norm(clamped), 0.1)


def test_plan_rendezvous_returns_center() -> None:
    ee_a = np.ones(3)
    ee_b = np.zeros(3)
    obj = np.array([0.0, 0.0, 1.0])
    pose_a, pose_b = ik_utils.plan_rendezvous(ee_a, ee_b, obj)
    expected = (ee_a + ee_b + obj) / 3.0
    assert np.allclose(pose_a, expected)
    assert np.allclose(pose_b, expected)


def test_gripper_command_clamps_range() -> None:
    assert ik_utils.gripper_command(-0.1) == 0.0
    assert ik_utils.gripper_command(0.5) == 0.5
    assert ik_utils.gripper_command(1.5) == 1.0


def test_ee_pose_from_state_depends_on_agent_id() -> None:
    pose_a = ik_utils.ee_pose_from_state({}, agent_id=0)
    pose_b = ik_utils.ee_pose_from_state({}, agent_id=1)
    assert pose_a[-1] == 0.0
    assert pose_b[-1] == 1.0
