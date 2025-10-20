"""Unit tests for the NewtonMAEnv scaffolding."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from envs.core_env import EnvironmentConfig, NewtonMAEnv
from envs.tasks.base import TaskMetadata, deregister_task, register_task


@dataclass
class _DummyTask:
    name: str = "dummy_phase1"
    _phases: tuple[str, ...] = ("phase_a", "phase_b")

    def build_scene(self, model: Any) -> None:  # pragma: no cover - not used in tests
        _unused(model)

    def phases(self) -> Sequence[str]:
        return self._phases

    def reset(self, state: Mapping[str, Any], rng: np.random.Generator) -> TaskMetadata:
        _unused(state, rng)
        return TaskMetadata(phases=self._phases, extras={"task": self.name})

    def reward(self, state: Mapping[str, Any], phase: str) -> tuple[float, float]:
        _unused(state, phase)
        return (0.0, 0.0)

    def success(self, state: Mapping[str, Any]) -> bool:
        _unused(state)
        return False

    def scripted_action(self, obs: Mapping[str, Any], phase: str, agent_id: int) -> Sequence[float]:
        _unused(obs, phase, agent_id)
        return (0.0, 0.0)

    def info(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        _unused(state)
        return {"task": self.name}


@pytest.fixture(name="dummy_task_name")
def fixture_dummy_task_name() -> Iterator[str]:
    name = _DummyTask.name
    register_task(name, _DummyTask)
    yield name
    deregister_task(name)


def test_environment_reset_returns_two_observations(dummy_task_name: str) -> None:
    env = NewtonMAEnv({"task_name": dummy_task_name, "max_steps": 5})
    observations = env.reset("lift the box")
    assert len(observations) == 2
    assert observations[0]["instruction"] == "lift the box"
    assert observations[0]["task"] == dummy_task_name


def test_step_returns_rewards_and_info(dummy_task_name: str) -> None:
    env = NewtonMAEnv({"task_name": dummy_task_name, "max_steps": 3})
    env.reset("handover the object")

    obs, rewards, done, info = env.step([[0.0], [0.0]])

    assert len(obs) == 2
    assert rewards == [0.0, 0.0]
    assert not done
    assert info["task"] == dummy_task_name
    assert info["step"] == 1


def test_set_task_switches_to_registered_task(dummy_task_name: str) -> None:
    env = NewtonMAEnv({"task_name": dummy_task_name})
    env.reset("prepare")
    env.set_task("lift")
    assert env.get_state_dict()["task"] == "lift"


def test_environment_config_parses_defaults() -> None:
    cfg = EnvironmentConfig.from_mapping({})
    assert cfg.task_name == "lift"
    assert cfg.seed == 0
    assert cfg.max_steps == 200


def _unused(*_: Any) -> None:
    """Helper to silence unused argument warnings."""
