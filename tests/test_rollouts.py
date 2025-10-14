"""Tests for rollout utilities."""

from __future__ import annotations

from typing import Any

from eval.rollouts import run_episode, run_suite


class _DummyEnv:
    def __init__(self) -> None:
        self._config = type("Cfg", (), {"max_steps": 5})()
        self._task = "lift"

    def reset(self, instruction: str) -> list[dict[str, Any]]:
        self._instruction = instruction
        return [{"rgb": None}, {"rgb": None}]

    def step(self, actions):
        info = {"task_success": False, "collision": 0.0, "coordination": 1.0}
        return [{}, {}], [0.0, 0.0], False, info

    def set_task(self, name: str) -> None:
        self._task = name


def _policy(_obs):
    return [[0.0], [0.0]]


def test_run_episode_returns_trajectory() -> None:
    env = _DummyEnv()
    traj = run_episode(env, _policy, instruction="test", max_steps=3)
    assert "steps" in traj


def test_run_suite_handles_multiple_tasks() -> None:
    env = _DummyEnv()
    results = run_suite(env, _policy, tasks=["lift", "handoff"], n_eps=1, unseen=False)
    assert len(results) == 2
