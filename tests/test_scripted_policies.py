"""Tests for scripted policy heuristics."""

from __future__ import annotations

import numpy as np

from control.phase_machine import PhaseMachine
from control.scripted import drawer_demo, handoff_demo, lift_demo
from envs.core_env import NewtonMAEnv


def _make_env(task_name: str) -> tuple[NewtonMAEnv, list[dict[str, object]]]:
    env = NewtonMAEnv({"task_name": task_name, "max_steps": 5})
    obs = env.reset("placeholder instruction")
    return env, obs


def test_lift_scripted_policy_returns_delta_with_gripper_channel() -> None:
    env, obs = _make_env("lift")
    machine = PhaseMachine(phases=("reach", "grasp", "lift"))
    actions = lift_demo.scripted_policy(env, machine, obs)
    assert len(actions) == 2
    for action in actions:
        assert action.shape == (4,)
        assert np.linalg.norm(action[:3]) <= 0.06 + 1e-6
        assert 0.0 <= action[3] <= 1.0


def test_handoff_waypoint_library_contains_expected_entries() -> None:
    waypoints = handoff_demo.waypoint_library({})
    assert {"source_offset", "handoff_offset", "radius"}.issubset(waypoints)


def test_drawer_policy_outputs_two_valid_actions() -> None:
    env, obs = _make_env("drawer")
    machine = PhaseMachine(phases=("reach_handles", "grasp", "pull"))
    actions = drawer_demo.scripted_policy(env, machine, obs)
    assert len(actions) == 2
    for action in actions:
        assert action.shape == (4,)
        assert np.linalg.norm(action[:3]) <= 0.05 + 1e-6
        assert 0.0 <= action[3] <= 1.0
