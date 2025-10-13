"""Tests for scripted policy scaffolds."""

from __future__ import annotations

import numpy as np

from control.phase_machine import PhaseMachine
from control.scripted import drawer_demo, handoff_demo, lift_demo
from envs.core_env import NewtonMAEnv


def _make_env(task_name: str) -> NewtonMAEnv:
    env = NewtonMAEnv({"task_name": task_name, "max_steps": 2})
    env.reset("placeholder instruction")
    return env


def test_lift_scripted_policy_returns_two_actions() -> None:
    env = _make_env("lift")
    machine = PhaseMachine(phases=("reach", "grasp"))
    actions = lift_demo.scripted_policy(env, machine, obs=[{}, {}])
    assert len(actions) == 2
    assert all(action.shape == (7,) for action in actions)


def test_handoff_waypoint_library_contains_keys() -> None:
    waypoints = handoff_demo.waypoint_library({})
    assert {"approach_a", "approach_b", "handover"}.issubset(waypoints)


def test_drawer_actions_are_clamped() -> None:
    env = _make_env("drawer")
    machine = PhaseMachine(phases=("reach", "pull"))
    actions = drawer_demo.scripted_policy(env, machine, obs=[{}, {}])
    norms = [np.linalg.norm(act) for act in actions]
    assert max(norms) <= 0.04 + 1e-6
