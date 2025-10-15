"""Hand-off cooperative task implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from envs.sim_state import SimulationObject, SimulationState

from .base import TaskMetadata, register_task_class


@dataclass(slots=True)
class _HandOffRuntime:
    distance_source: float = np.inf
    distance_handoff: float = np.inf
    distance_target: float = np.inf
    holders: tuple[bool, bool] = (False, False)
    phase_flags: dict[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.phase_flags is None:
            self.phase_flags = {}


@register_task_class
class HandOffTask:
    """Two-arm hand-off that transfers a baton from agent A to agent B."""

    name = "handoff"

    def __init__(self) -> None:
        self._phases: Sequence[str] = (
            "reach_source",
            "grasp_source",
            "handover",
            "grasp_target",
            "release",
        )

    def build_scene(self, model: Any) -> None:  # pragma: no cover
        _unused(model)

    def phases(self) -> Sequence[str]:
        return self._phases

    def reset(self, state: "SimulationState", rng: np.random.Generator) -> TaskMetadata:
        src = np.array([-0.38, -0.1, 0.08], dtype=np.float64)
        handoff = np.array([0.0, 0.05, 0.20], dtype=np.float64)
        tgt = np.array([0.35, 0.12, 0.08], dtype=np.float64)

        src[:2] += rng.uniform(-0.02, 0.02, size=2)

        state.objects.clear()

        baton = SimulationObject(
            name="baton",
            position=src.copy(),
            target=tgt.copy(),
            radius=0.04,
        )
        state.objects["baton"] = baton

        agent_a = state.arms["agent_0"]
        agent_b = state.arms["agent_1"]
        agent_a.position[:] = src + np.array([0.05, 0.0, 0.17])
        agent_b.position[:] = handoff + np.array([0.05, 0.0, 0.08])
        for arm in (agent_a, agent_b):
            arm.velocity[:] = 0.0
            arm.gripper = 1.0

        state.task_state = {
            "task": self.name,
            "runtime": _HandOffRuntime(),
            "source_pos": src,
            "handoff_pos": handoff,
            "target_pos": tgt,
            "surface_height": 0.05,
            "grasp_threshold": 0.06,
            "release_threshold": 0.88,
        }
        extras = {
            "task": self.name,
            "source": src.tolist(),
            "handoff": handoff.tolist(),
            "target": tgt.tolist(),
        }
        return TaskMetadata(phases=self._phases, extras=extras)

    def reward(self, state: "SimulationState", phase: str) -> tuple[float, float]:
        runtime = self._runtime(state)
        baton = state.objects["baton"]
        agent_a = state.arms["agent_0"]
        agent_b = state.arms["agent_1"]

        runtime.phase_flags.clear()
        source = state.task_state["source_pos"]
        handoff = state.task_state["handoff_pos"]
        target = state.task_state["target_pos"]

        runtime.distance_source = float(np.linalg.norm(agent_a.position - source))
        runtime.distance_handoff = float(np.linalg.norm(baton.position - handoff))
        runtime.distance_target = float(np.linalg.norm(baton.position - target))
        runtime.holders = ("agent_0" in baton.holders, "agent_1" in baton.holders)

        reward = np.zeros(2, dtype=np.float64)
        if phase == "reach_source":
            reward[0] = 1.0 - np.clip(runtime.distance_source / 0.35, 0.0, 1.0)
            reward[1] = 1.0 - np.clip(np.linalg.norm(agent_b.position - handoff) / 0.35, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(
                runtime.distance_source < 0.12 and np.linalg.norm(agent_b.position - handoff) < 0.12
            )
        elif phase == "grasp_source":
            reward[0] = 1.0 if runtime.holders[0] else 0.0
            reward[1] = 1.0 - np.clip(np.linalg.norm(agent_b.position - handoff) / 0.25, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(runtime.holders[0] and runtime.distance_source < 0.1)
        elif phase == "handover":
            reward[:] = 1.0 if all(runtime.holders) else 0.0
            runtime.phase_flags[phase] = bool(
                all(runtime.holders) and runtime.distance_handoff < 0.08
            )
        elif phase == "grasp_target":
            reward[1] = 1.0 if runtime.holders[1] else 0.0
            reward[0] = np.clip(1.0 - runtime.distance_handoff / 0.3, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(
                runtime.holders[1] and runtime.distance_target < 0.08 and not runtime.holders[0]
            )
        elif phase == "release":
            openers = np.array([agent_a.gripper, agent_b.gripper])
            reward[:] = np.clip(openers, 0.0, 1.0)
            runtime.phase_flags[phase] = bool(
                openers.min() >= 0.85
                and not baton.holders
                and runtime.distance_target < 0.05
                and baton.position[2] <= state.task_state["surface_height"] + 0.02
            )
        else:
            reward[:] = 0.0

        return float(reward[0]), float(reward[1])

    def success(self, state: "SimulationState") -> bool:
        runtime = self._runtime(state)
        return bool(
            runtime.phase_flags.get("release")
            and runtime.distance_target < 0.05
            and not state.objects["baton"].holders
        )

    def scripted_action(self, obs: Mapping[str, Any], phase: str, agent_id: int) -> Sequence[float]:
        _unused(obs, phase, agent_id)
        raise NotImplementedError("Scripted policies are provided in Phase 2.")

    def info(self, state: "SimulationState") -> Mapping[str, Any]:
        runtime = self._runtime(state)
        phase_name = state.task_state.get("phase", self._phases[0])
        return {
            "task": self.name,
            "distance_source": runtime.distance_source,
            "distance_handoff": runtime.distance_handoff,
            "distance_target": runtime.distance_target,
            "phase_complete": bool(runtime.phase_flags.get(phase_name)),
        }

    def _runtime(self, state: "SimulationState") -> _HandOffRuntime:
        runtime = state.task_state.setdefault("runtime", _HandOffRuntime())
        return runtime


def _unused(*args: Any, **kwargs: Any) -> None:
    """Helper to silence unused-variable warnings."""
