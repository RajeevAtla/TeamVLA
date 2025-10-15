"""Tests for the TeamVLA control IK utilities."""

from __future__ import annotations

import numpy as np

from control import ik_utils
from envs.sim_state import ArmState, SimulationModel, SimulationObject, SimulationState


def _make_sim_state() -> SimulationState:
    model = SimulationModel(
        workspace_min=np.array([-1.0, -1.0, 0.0]),
        workspace_max=np.array([1.0, 1.0, 1.0]),
        action_limit=0.1,
        dt=0.05,
    )
    del model  # Model unused directly in tests but mimics real usage.
    arms = {
        "agent_0": ArmState(
            name="agent_0",
            position=np.array([-0.2, 0.0, 0.2]),
            velocity=np.zeros(3),
        ),
        "agent_1": ArmState(
            name="agent_1",
            position=np.array([0.2, 0.0, 0.2]),
            velocity=np.zeros(3),
        ),
    }
    state = SimulationState(
        arms=arms,
        objects={"payload": SimulationObject("payload", np.zeros(3), np.zeros(3), radius=0.05)},
        rng=np.random.default_rng(0),
    )
    return state


def test_solve_ik_converges_towards_target() -> None:
    q_init = np.array([0.0, 0.0, 0.0])
    target = np.array([0.3, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
    solved = ik_utils.solve_ik(q_init, target, max_iters=20, atol=1e-4)
    assert np.linalg.norm(solved - target[:3]) < 0.05


def test_solve_ik_respects_limits() -> None:
    q_init = np.array([0.5, -0.5])
    limits = {"lower": np.array([0.0, -1.0]), "upper": np.array([0.6, 0.0])}
    solved = ik_utils.solve_ik(q_init, target_pose=np.array([1.0, 0.0, 0.0]), limits=limits)
    assert solved[0] <= 0.6
    assert solved[1] >= -1.0


def test_clamp_action_limits_norm() -> None:
    delta = np.array([1.0, 0.0, 0.0])
    clamped = ik_utils.clamp_action(delta, max_norm=0.1)
    assert np.isclose(np.linalg.norm(clamped), 0.1)


def test_plan_rendezvous_returns_lateral_targets() -> None:
    ee_a = np.array([-0.2, 0.0, 0.25])
    ee_b = np.array([0.2, 0.0, 0.25])
    obj = np.array([0.0, 0.0, 0.2])
    pose_a, pose_b = ik_utils.plan_rendezvous(ee_a, ee_b, obj)
    # Ensure they straddle the object on opposite sides.
    assert np.sign(pose_a[0]) == -np.sign(pose_b[0]) or np.isclose(pose_a[0], pose_b[0])
    assert np.allclose(pose_a[2], obj[2])
    assert np.allclose(pose_b[2], obj[2])


def test_gripper_command_clamps_range() -> None:
    assert ik_utils.gripper_command(-0.1) == 0.0
    assert ik_utils.gripper_command(0.5) == 0.5
    assert ik_utils.gripper_command(1.5) == 1.0


def test_ee_pose_from_simulation_state() -> None:
    state = _make_sim_state()
    pose = ik_utils.ee_pose_from_state(state, agent_id=0)
    assert pose.shape[0] == 7
    assert np.allclose(pose[:3], state.arms["agent_0"].position)
