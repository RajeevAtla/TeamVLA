"""Tests for the task registry helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from envs.tasks.base import TaskMetadata, TaskSpec, deregister_task, get_task, register_task


@dataclass
class _TempTask:
    name: str = "temp_task"
    _phases: tuple[str, ...] = ("phase",)

    def build_scene(self, model: Any) -> None:  # pragma: no cover - placeholder
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


def test_register_and_get_task_roundtrip() -> None:
    register_task(_TempTask.name, _TempTask)
    try:
        task = get_task(_TempTask.name)
        assert isinstance(task, TaskSpec)
        assert task.name == _TempTask.name
    finally:
        deregister_task(_TempTask.name)


def test_registering_duplicate_task_raises() -> None:
    register_task(_TempTask.name, _TempTask)
    try:
        with pytest.raises(ValueError):
            register_task(_TempTask.name, _TempTask)
    finally:
        deregister_task(_TempTask.name)


def test_get_unknown_task_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_task("unknown_task")


def _unused(*_: Any) -> None:
    """Helper to silence unused argument warnings."""
