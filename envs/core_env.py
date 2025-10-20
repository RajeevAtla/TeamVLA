"""Deterministic two-arm simulation environment for TeamVLA."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from envs.sim_state import ArmState, SimulationModel, SimulationState
from envs.tasks.base import TaskSpec, get_task, iter_registered_tasks

LOGGER = logging.getLogger(__name__)

ACTION_SIZE = 4  # Δx, Δy, Δz, gripper
NUM_AGENTS = 2


@dataclass(slots=True)
class EnvironmentConfig:
    """Parsed environment configuration."""

    task_name: str
    seed: int = 0
    max_steps: int = 200

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> EnvironmentConfig:
        """Parse a configuration mapping into an EnvironmentConfig instance."""

        task_name = str(cfg.get("task_name", "lift"))
        seed = int(cfg.get("seed", 0))
        max_steps = int(cfg.get("max_steps", 200))
        return cls(task_name=task_name, seed=seed, max_steps=max_steps)


class NewtonMAEnv:
    """Deterministic tabletop simulator that mimics a Newton multi-arm API."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._config = EnvironmentConfig.from_mapping(cfg)
        self._rng = np.random.default_rng(self._config.seed)
        self._model, self._state = build_multi_robot_world(cfg, seed=self._config.seed)
        self._state.rng = self._rng
        self._task: TaskSpec = cfg.get("task") or get_task(self._config.task_name)
        self._task.build_scene(self._model)
        self._instruction: str = ""
        self._step_count: int = 0
        self._phases: Sequence[str] = tuple(self._task.phases())
        self._phase_index: int = 0
        self._current_phase: str = self._phases[0] if self._phases else "default"

    def reset(self, instruction: str) -> list[dict[str, Any]]:
        """Reset the environment and return fresh observations."""

        self._instruction = instruction
        self._step_count = 0
        self._phase_index = 0
        self._current_phase = self._phases[0] if self._phases else "default"
        self._state.reset_step()

        meta = self._task.reset(self._state, self._rng)
        self._phases = tuple(meta.phases)
        self._current_phase = self._phases[0] if self._phases else "default"
        self._phase_index = 0
        self._state.task_state["phase"] = self._current_phase
        extras = dict(meta.extras)
        extras.update({"task": self._task.name, "phase": self._current_phase})

        return [self._build_observation(agent_id=i, extras=extras) for i in range(NUM_AGENTS)]

    def step(
        self, actions: Sequence[Sequence[float]]
    ) -> tuple[list[dict[str, Any]], list[float], bool, dict[str, Any]]:
        """Apply per-agent actions and advance the simulated world by one step."""

        self._validate_actions(actions)
        self._apply_actions(actions)
        self._state.step_index += 1
        self._step_count += 1
        self._state.task_state["phase"] = self._current_phase
        self._handle_object_attachments()

        rewards = self._sanitize_rewards(self._task.reward(self._state, self._current_phase))
        done = bool(self._task.success(self._state) or self._step_count >= self._config.max_steps)
        info = self._build_info(done)
        observations = [self._build_observation(agent_id=i, extras=info) for i in range(NUM_AGENTS)]
        return observations, rewards, done, info

    def render(self, mode: str = "rgb_array") -> dict[str, Any]:
        """Render RGB observations. Returns placeholders until rendering is implemented."""

        _unused(mode)
        return {
            "agent_a": self._render_topdown("agent_0"),
            "agent_b": self._render_topdown("agent_1"),
        }

    def close(self) -> None:
        """Clean up resources."""

        self._state.objects.clear()

    def set_task(self, name: str) -> None:
        """Switch to a new task by name."""

        self._task = get_task(name)
        self._task.build_scene(self._model)
        self._phases = tuple(self._task.phases())
        self._phase_index = 0
        self._current_phase = self._phases[0] if self._phases else "default"
        self._state.task_state["phase"] = self._current_phase

    def get_state_dict(self) -> dict[str, Any]:
        """Return a lightweight snapshot of the environment state."""

        return {
            "task": self._task.name,
            "seed": self._config.seed,
            "step": self._step_count,
            "phase": self._current_phase,
            "phases": list(self._phases),
        }

    def available_tasks(self) -> list[str]:
        """Expose registered tasks for downstream tooling."""

        return sorted(iter_registered_tasks())

    # ---------------------------------------------------------------------#
    # Internal helpers                                                     #
    # ---------------------------------------------------------------------#

    def _build_observation(self, agent_id: int, extras: Mapping[str, Any]) -> dict[str, Any]:
        return obs_i(
            state=self._state,
            model=self._model,
            agent_id=agent_id,
            instruction=self._instruction,
            extras=extras,
            phase=self._current_phase,
        )

    def _render_topdown(self, focus: str) -> NDArray[np.uint8]:
        return render_topdown(self._state, self._model, focus)

    def _validate_actions(self, actions: Sequence[Sequence[float]]) -> None:
        if len(actions) != NUM_AGENTS:
            raise ValueError(f"Expected actions for exactly {NUM_AGENTS} agents.")
        for idx, action in enumerate(actions):
            if not isinstance(action, Sequence):
                raise TypeError(f"Action for agent {idx} must be a sequence.")
            if len(action) < ACTION_SIZE:
                raise ValueError(
                    f"Action for agent {idx} must contain at least {ACTION_SIZE} elements."
                )

    def _apply_actions(self, actions: Sequence[Sequence[float]]) -> None:
        for idx, action in enumerate(actions):
            agent = f"agent_{idx}"
            arm = self._state.arms[agent]
            vector = np.asarray(action, dtype=np.float64).reshape(-1)
            delta = np.clip(vector[:3], -self._model.action_limit, self._model.action_limit)
            new_position = np.clip(
                arm.position + delta,
                self._model.workspace_min,
                self._model.workspace_max,
            )
            arm.velocity = (new_position - arm.position) / self._model.dt
            arm.position = new_position
            arm.gripper = float(np.clip(vector[3], 0.0, 1.0))

    def _handle_object_attachments(self) -> None:
        threshold = float(self._state.task_state.get("grasp_threshold", 0.075))
        release_threshold = float(self._state.task_state.get("release_threshold", 0.8))
        for obj in self._state.objects.values():
            holders_to_remove: set[str] = set()
            for name, arm in self._state.arms.items():
                distance = np.linalg.norm(arm.position - obj.position)
                if arm.gripper < 0.25 and distance <= max(threshold, obj.radius * 1.5):
                    obj.attach(name)
                elif arm.gripper > release_threshold:
                    holders_to_remove.add(name)
            for name in holders_to_remove:
                obj.detach(name)

            if obj.holders:
                holder_positions = np.array(
                    [self._state.arms[name].position for name in obj.holders],
                    dtype=np.float64,
                )
                obj.position = holder_positions.mean(axis=0)
            else:
                surface_height = float(self._state.task_state.get("surface_height", 0.05))
                obj.position[2] = max(surface_height, obj.position[2] - 0.015)

    def _advance_phase(self) -> None:
        if self._phase_index < len(self._phases) - 1:
            self._phase_index += 1
            self._current_phase = self._phases[self._phase_index]
            self._state.task_state["phase"] = self._current_phase

    def _sanitize_rewards(self, rewards: Sequence[float]) -> list[float]:
        rewards = list(rewards)
        if len(rewards) >= NUM_AGENTS:
            return rewards[:NUM_AGENTS]
        rewards.extend([0.0] * (NUM_AGENTS - len(rewards)))
        return rewards

    def _build_info(self, done: bool) -> dict[str, Any]:
        previous_phase = self._current_phase
        info = dict(self._task.info(self._state))
        should_advance = bool(info.get("phase_complete")) and not done
        if should_advance:
            self._advance_phase()
        info.update(
            {
                "task": self._task.name,
                "phase": self._current_phase,
                "previous_phase": previous_phase,
                "step": self._step_count,
                "done": done,
                "task_success": bool(self._task.success(self._state)),
            }
        )
        return info


def build_multi_robot_world(
    cfg: Mapping[str, Any], *, seed: int | None = None
) -> tuple[SimulationModel, SimulationState]:
    """Construct the simplified dual-arm tabletop simulation."""

    del cfg  # Unused placeholder until richer configuration is required.
    rng = np.random.default_rng(seed)
    workspace_min = np.array([-0.6, -0.5, 0.0], dtype=np.float64)
    workspace_max = np.array([0.6, 0.5, 0.55], dtype=np.float64)
    model = SimulationModel(
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        action_limit=0.08,
        dt=0.05,
    )
    arms = {
        "agent_0": ArmState(
            name="agent_0",
            position=np.array([-0.35, 0.0, 0.28], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
        ),
        "agent_1": ArmState(
            name="agent_1",
            position=np.array([0.35, 0.0, 0.28], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
        ),
    }
    state = SimulationState(arms=arms, objects={}, rng=rng)
    return model, state


def setup_cameras(model: SimulationModel, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Placeholder for configuring per-agent cameras."""

    _unused(model, cfg)
    return {"agent_a": None, "agent_b": None}


def make_solver(cfg: Mapping[str, Any]) -> Any:
    """Placeholder for creating the Newton physics solver."""

    _unused(cfg)
    return None


def obs_i(
    state: SimulationState,
    model: SimulationModel,
    agent_id: int,
    instruction: str,
    extras: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    """Construct a structured observation for the specified agent."""

    agent_name = f"agent_{agent_id}"
    peer_name = f"agent_{1 - agent_id}"
    arm = state.arms[agent_name]
    peer = state.arms[peer_name]
    object_summary = {name: obj.position.copy() for name, obj in state.objects.items()}

    observation = {
        "agent_id": agent_id,
        "task": extras.get("task"),
        "phase": phase,
        "instruction": instruction,
        "robot_state": {
            "position": arm.position.copy(),
            "velocity": arm.velocity.copy(),
            "gripper": float(arm.gripper),
        },
        "peer_state": {
            "position": peer.position.copy(),
            "velocity": peer.velocity.copy(),
            "gripper": float(peer.gripper),
        },
        "objects": object_summary,
        "metadata": dict(extras),
        "rgb": render_topdown(state, model, agent_name),
    }
    return observation


def render_topdown(
    state: SimulationState, model: SimulationModel, focus: str | None = None
) -> NDArray[np.uint8]:
    """Render a top-down synthetic image showing arm/object locations."""

    del focus
    size = 48
    image = np.zeros((size, size, 3), dtype=np.uint8)

    def _project(point: NDArray[np.float64]) -> tuple[int, int]:
        norm = (point[:2] - model.workspace_min[:2]) / (
            model.workspace_max[:2] - model.workspace_min[:2] + 1e-8
        )
        x = int(np.clip(norm[0] * (size - 1), 0, size - 1))
        y = int(np.clip(norm[1] * (size - 1), 0, size - 1))
        return x, y

    for idx, arm in state.arms.items():
        x, y = _project(arm.position)
        color = (
            np.array([0, 180, 255], dtype=np.uint8)
            if idx == "agent_0"
            else np.array([255, 120, 0], dtype=np.uint8)
        )
        image[max(0, y - 1) : min(size, y + 2), max(0, x - 1) : min(size, x + 2)] = color

    for obj in state.objects.values():
        x, y = _project(obj.position)
        image[max(0, y - 1) : min(size, y + 2), max(0, x - 1) : min(size, x + 2)] = np.array(
            [120, 255, 120], dtype=np.uint8
        )
    return image


def _unused(*args: Any, **kwargs: Any) -> None:
    """Helper for silencing unused argument warnings in placeholders."""
