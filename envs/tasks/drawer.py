"""Bimanual drawer task implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.sim_state import SimulationObject, SimulationState

from .base import TaskMetadata, register_task_class


@dataclass(slots=True)
class _DrawerRuntime:
    distances: tuple[float, float] = (np.inf, np.inf)
    extension: float = 0.0
    hold_counter: int = 0
    phase_flags: dict[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.phase_flags is None:
            self.phase_flags = {}


@register_task_class
class DrawerTask:
    """Bimanual drawer task requiring synchronized pulling and release."""

    name = "drawer"

    def __init__(self) -> None:
        self._phases: Sequence[str] = ("reach_handles", "grasp", "pull", "hold", "release")

    def build_scene(self, model: Any) -> None:  # pragma: no cover
        _unused(model)

    def phases(self) -> Sequence[str]:
        return self._phases

    def reset(self, state: SimulationState, rng: np.random.Generator) -> TaskMetadata:
        base_left = np.array([-0.14, -0.12, 0.1], dtype=np.float64)
        base_right = np.array([0.14, -0.12, 0.1], dtype=np.float64)
        jitter = rng.uniform(-0.01, 0.01, size=2)
        base_left[:2] += jitter
        base_right[:2] -= jitter

        state.objects.clear()

        handle_left = SimulationObject(
            name="drawer_left",
            position=base_left.copy(),
            target=base_left.copy(),
            radius=0.035,
        )
        handle_right = SimulationObject(
            name="drawer_right",
            position=base_right.copy(),
            target=base_right.copy(),
            radius=0.035,
        )
        state.objects["drawer_left"] = handle_left
        state.objects["drawer_right"] = handle_right

        state.arms["agent_0"].position[:] = base_left + np.array([0.0, 0.04, 0.16])
        state.arms["agent_1"].position[:] = base_right + np.array([0.0, 0.04, 0.16])
        for arm in state.arms.values():
            arm.velocity[:] = 0.0
            arm.gripper = 1.0

        state.task_state = {
            "task": self.name,
            "runtime": _DrawerRuntime(),
            "handle_initial": {"drawer_left": base_left, "drawer_right": base_right},
            "drawer_extension": 0.0,
            "max_extension": 0.25,
            "surface_height": 0.05,
            "grasp_threshold": 0.06,
            "release_threshold": 0.9,
            "hold_steps_required": 5,
            "assigned_handles": {
                "agent_0": "drawer_left",
                "agent_1": "drawer_right",
            },
        }
        extras = {
            "task": self.name,
            "handles": {
                "left": base_left.tolist(),
                "right": base_right.tolist(),
            },
        }
        return TaskMetadata(phases=self._phases, extras=extras)

    def reward(self, state: SimulationState, phase: str) -> tuple[float, float]:
        runtime = self._runtime(state)
        runtime.phase_flags.clear()
        if state.task_state.get("phase") != "hold":
            runtime.hold_counter = 0

        handles = {name: state.objects[name] for name in ("drawer_left", "drawer_right")}
        assignments = state.task_state["assigned_handles"]
        initial = state.task_state["handle_initial"]

        distances = []
        for agent_name in ("agent_0", "agent_1"):
            arm = state.arms[agent_name]
            handle_name = assignments[agent_name]
            handle = handles[handle_name]
            distances.append(float(np.linalg.norm(arm.position - handle.position)))
        runtime.distances = (distances[0], distances[1])

        # Update extension based on handle displacement
        ext = 0.0
        for name, base in initial.items():
            current = state.objects[name].position
            ext += float(current[1] - base[1])
        ext = np.clip(ext / 2.0, 0.0, state.task_state["max_extension"])
        state.task_state["drawer_extension"] = ext
        runtime.extension = ext

        # Clamp handles to remain aligned with drawer face.
        for name, base in initial.items():
            handle = state.objects[name]
            handle.position[0] = base[0]
            handle.position[2] = base[2]
            handle.position[1] = base[1] + ext

        reward = np.zeros(2, dtype=np.float64)
        holdings = (
            "agent_0" in state.objects[assignments["agent_0"]].holders,
            "agent_1" in state.objects[assignments["agent_1"]].holders,
        )

        if phase == "reach_handles":
            reward[:] = 1.0 - np.clip(np.array(runtime.distances) / 0.35, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(
                runtime.distances[0] < 0.12 and runtime.distances[1] < 0.12
            )
        elif phase == "grasp":
            reward[:] = np.array(holdings, dtype=np.float64)
            runtime.phase_flags[phase] = bool(all(holdings))
        elif phase == "pull":
            progress = np.clip(runtime.extension / state.task_state["max_extension"], 0.0, 1.2)
            reward[:] = progress
            runtime.phase_flags[phase] = bool(all(holdings) and runtime.extension >= 0.18)
        elif phase == "hold":
            if runtime.extension >= 0.2 and all(holdings):
                runtime.hold_counter += 1
            else:
                runtime.hold_counter = 0
            reward.fill(runtime.extension)
            runtime.phase_flags[phase] = bool(
                runtime.hold_counter >= state.task_state["hold_steps_required"]
            )
        elif phase == "release":
            openers = np.array([state.arms["agent_0"].gripper, state.arms["agent_1"].gripper])
            reward[:] = np.clip(openers, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(
                openers.min() >= 0.9
                and not any(
                    agent in state.objects[assignments[agent]].holders for agent in assignments
                )
                and runtime.extension >= 0.2
            )
        else:
            reward[:] = 0.0

        return float(reward[0]), float(reward[1])

    def success(self, state: SimulationState) -> bool:
        runtime = self._runtime(state)
        return bool(runtime.phase_flags.get("release") and runtime.extension >= 0.2)

    def scripted_action(self, obs: Mapping[str, Any], phase: str, agent_id: int) -> Sequence[float]:
        _unused(obs, phase, agent_id)
        raise NotImplementedError("Scripted controllers live in control.scripted drawer module.")

    def info(self, state: SimulationState) -> Mapping[str, Any]:
        runtime = self._runtime(state)
        phase_name = state.task_state.get("phase", self._phases[0])
        return {
            "task": self.name,
            "distance_left": runtime.distances[0],
            "distance_right": runtime.distances[1],
            "drawer_extension": runtime.extension,
            "phase_complete": bool(runtime.phase_flags.get(phase_name)),
        }

    def _runtime(self, state: SimulationState) -> _DrawerRuntime:
        runtime = state.task_state.setdefault("runtime", _DrawerRuntime())
        return runtime


def _unused(*args: Any, **kwargs: Any) -> None:
    """Helper to silence unused-variable warnings."""
