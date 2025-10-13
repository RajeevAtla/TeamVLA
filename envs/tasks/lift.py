"""Lift-and-place cooperative task skeleton."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import TaskMetadata, register_task_class


@register_task_class
class LiftTask:
    """Skeleton implementation of the lift-and-place cooperative task."""

    name = "lift"

    def __init__(self) -> None:
        self._phases: Sequence[str] = ("reach", "grasp", "lift", "place", "release")

    def build_scene(self, model: Any) -> None:
        raise NotImplementedError("Phase 2 will populate the Newton scene for the lift task.")

    def phases(self) -> Sequence[str]:
        return self._phases

    def reset(self, state: Any, rng: Any) -> TaskMetadata:
        meta = {"task": self.name, "seed": getattr(rng, "seed", None)}
        return TaskMetadata(phases=self._phases, extras=meta)

    def reward(self, state: Any, phase: str) -> tuple[float, float]:
        _unused(state, phase)
        return (0.0, 0.0)

    def success(self, state: Any) -> bool:
        _unused(state)
        return False

    def scripted_action(self, obs: Mapping[str, Any], phase: str, agent_id: int) -> Sequence[float]:
        _unused(obs, phase, agent_id)
        raise NotImplementedError("Phase 2 will provide scripted policies.")

    def info(self, state: Any) -> Mapping[str, Any]:
        _unused(state)
        return {"task": self.name}


def _unused(*args: Any, **kwargs: Any) -> None:
    """Helper to silence unused-variable warnings in skeletal implementations."""
