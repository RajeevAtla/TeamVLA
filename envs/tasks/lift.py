"""Lift-and-place cooperative task implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.sim_state import SimulationObject, SimulationState

from .base import TaskMetadata, register_task_class


@dataclass(slots=True)
class _LiftRuntime:
    distances: tuple[float, float] = (np.inf, np.inf)
    height: float = 0.0
    xy_distance: float = np.inf
    holders: tuple[bool, bool] = (False, False)
    phase_flags: dict[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.phase_flags is None:
            self.phase_flags = {}


@register_task_class
class LiftTask:
    """Cooperative lift-and-place task that rewards synchronized lifting."""

    name = "lift"

    def __init__(self) -> None:
        self._phases: Sequence[str] = ("reach", "grasp", "lift", "place", "release")

    def build_scene(self, model: Any) -> None:  # pragma: no cover - model not used yet.
        _unused(model)

    def phases(self) -> Sequence[str]:
        return self._phases

    def reset(self, state: SimulationState, rng: np.random.Generator) -> TaskMetadata:
        state.objects.clear()
        left = state.arms["agent_0"]
        right = state.arms["agent_1"]
        left.position[:] = np.array([-0.28, -0.05, 0.24])
        right.position[:] = np.array([0.28, -0.05, 0.24])
        left.velocity[:] = 0.0
        right.velocity[:] = 0.0
        left.gripper = right.gripper = 1.0

        rng_offsets = rng.uniform(-0.05, 0.05, size=2)
        cube_position = np.array([0.0, -0.05, 0.08], dtype=np.float64)
        cube_position[:2] += rng_offsets
        place_xy = np.asarray(rng.uniform([-0.2, 0.15], [0.2, 0.3]), dtype=np.float64)
        lift_height = 0.32
        surface_height = 0.05

        cube = SimulationObject(
            name="payload",
            position=cube_position,
            target=np.array([place_xy[0], place_xy[1], surface_height]),
            radius=0.055,
        )
        state.objects["payload"] = cube
        state.task_state = {
            "task": self.name,
            "runtime": _LiftRuntime(),
            "target_xy": cube.target[:2].copy(),
            "target_height": lift_height,
            "surface_height": surface_height,
            "grasp_threshold": 0.07,
            "release_threshold": 0.85,
        }
        extras = {
            "task": self.name,
            "target_xy": cube.target[:2].tolist(),
            "target_height": lift_height,
        }
        return TaskMetadata(phases=self._phases, extras=extras)

    def reward(self, state: SimulationState, phase: str) -> tuple[float, float]:
        runtime = self._runtime(state)
        cube = state.objects["payload"]
        left = state.arms["agent_0"]
        right = state.arms["agent_1"]
        runtime.phase_flags.clear()

        distances = (
            float(np.linalg.norm(left.position - cube.position)),
            float(np.linalg.norm(right.position - cube.position)),
        )
        runtime.distances = distances
        runtime.holders = ("agent_0" in cube.holders, "agent_1" in cube.holders)
        runtime.height = float(cube.position[2])
        runtime.xy_distance = float(
            np.linalg.norm(cube.position[:2] - state.task_state["target_xy"])
        )

        rewards = np.zeros(2, dtype=np.float64)
        if phase == "reach":
            rewards = 1.0 - np.clip(np.array(distances) / 0.35, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(distances[0] < 0.12 and distances[1] < 0.12)
        elif phase == "grasp":
            rewards = np.array(runtime.holders, dtype=np.float64)
            closeness = distances[0] < 0.1 and distances[1] < 0.1
            runtime.phase_flags[phase] = bool(all(runtime.holders) and closeness)
        elif phase == "lift":
            lift_height = state.task_state["target_height"]
            rewards.fill(np.clip(runtime.height / lift_height, 0.0, 1.1))
            runtime.phase_flags[phase] = bool(runtime.height >= lift_height - 0.01)
        elif phase == "place":
            horizontal = np.clip(1.0 - runtime.xy_distance / 0.35, 0.0, 1.2)
            rewards.fill(horizontal)
            runtime.phase_flags[phase] = bool(runtime.xy_distance <= 0.05)
        elif phase == "release":
            openers = np.array([left.gripper, right.gripper])
            rewards = np.clip(openers, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(
                openers.min() >= 0.85
                and not cube.holders
                and runtime.xy_distance <= 0.05
                and runtime.height <= state.task_state["surface_height"] + 0.02
            )
        else:
            rewards.fill(0.0)

        return float(rewards[0]), float(rewards[1])

    def success(self, state: SimulationState) -> bool:
        runtime = self._runtime(state)
        surface = state.task_state["surface_height"]
        return bool(
            runtime.phase_flags.get("release")
            and runtime.xy_distance <= 0.05
            and runtime.height <= surface + 0.02
            and not state.objects["payload"].holders
        )

    def scripted_action(self, obs: Mapping[str, Any], phase: str, agent_id: int) -> Sequence[float]:
        _unused(obs, phase, agent_id)
        raise NotImplementedError("Scripted policies are provided in Phase 2.")

    def info(self, state: SimulationState) -> Mapping[str, Any]:
        runtime = self._runtime(state)
        return {
            "task": self.name,
            "distance_a": runtime.distances[0],
            "distance_b": runtime.distances[1],
            "object_height": runtime.height,
            "distance_to_target": runtime.xy_distance,
            "phase_complete": bool(runtime.phase_flags.get(state.task_state.get("phase", ""))),
        }

    def _runtime(self, state: SimulationState) -> _LiftRuntime:
        runtime = state.task_state.setdefault("runtime", _LiftRuntime())
        state.task_state["phase"] = state.task_state.get("phase", self._phases[0])
        return runtime


def _unused(*args: Any, **kwargs: Any) -> None:
    """Helper to silence unused-variable warnings."""
