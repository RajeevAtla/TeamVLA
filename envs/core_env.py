"""Core multi-agent environment scaffolding for TeamVLA."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from envs.tasks.base import TaskSpec, get_task, iter_registered_tasks

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EnvironmentConfig:
    """Parsed environment configuration."""

    task_name: str
    seed: int = 0
    max_steps: int = 200

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "EnvironmentConfig":
        """Parse a configuration mapping into an EnvironmentConfig instance."""

        task_name = str(cfg.get("task_name", "lift"))
        seed = int(cfg.get("seed", 0))
        max_steps = int(cfg.get("max_steps", 200))
        return cls(task_name=task_name, seed=seed, max_steps=max_steps)


class NewtonMAEnv:
    """Minimal Newton environment skeleton supporting task swapping."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._config = EnvironmentConfig.from_mapping(cfg)
        self._rng = np.random.default_rng(self._config.seed)
        self._model, self._state = build_multi_robot_world(cfg)
        self._task: TaskSpec = cfg.get("task") or get_task(self._config.task_name)
        self._instruction: str = ""
        self._step_count: int = 0
        self._phases: Sequence[str] = tuple(self._task.phases())
        self._current_phase: str = self._phases[0] if self._phases else "default"

    def reset(self, instruction: str) -> list[dict[str, Any]]:
        """Reset the environment and return fresh observations."""

        self._instruction = instruction
        self._step_count = 0
        meta = self._task.reset(self._state, self._rng)
        self._phases = tuple(meta.phases)
        self._current_phase = self._phases[0] if self._phases else "default"
        return [obs_i(agent_id=i, instruction=instruction, extras=meta.extras) for i in range(2)]

    def step(self, actions: Sequence[Sequence[float]]) -> tuple[list[dict[str, Any]], list[float], bool, dict[str, Any]]:
        """Apply per-agent actions and advance the simulated world by one step."""

        self._validate_actions(actions)
        self._step_count += 1
        rewards = self._sanitize_rewards(self._task.reward(self._state, self._current_phase))
        done = self._task.success(self._state) or self._step_count >= self._config.max_steps
        info = self._build_info(done)
        observations = [
            obs_i(agent_id=i, instruction=self._instruction, extras=info)
            for i in range(2)
        ]
        return observations, rewards, done, info

    def render(self, mode: str = "rgb_array") -> dict[str, Any]:
        """Render RGB observations. Returns placeholders until rendering is implemented."""

        _unused(mode)
        return {"agent_a": None, "agent_b": None}

    def close(self) -> None:
        """Clean up resources. Placeholder for future Newton handles."""

        self._model = None
        self._state = None

    def set_task(self, name: str) -> None:
        """Switch to a new task by name."""

        self._task = get_task(name)
        self._phases = tuple(self._task.phases())
        self._current_phase = self._phases[0] if self._phases else "default"

    def get_state_dict(self) -> dict[str, Any]:
        """Return a lightweight snapshot of the environment state."""

        return {
            "task": self._task.name,
            "seed": self._config.seed,
            "step": self._step_count,
            "phases": list(self._phases),
        }

    def available_tasks(self) -> list[str]:
        """Expose registered tasks for downstream tooling."""

        return sorted(iter_registered_tasks())

    def _validate_actions(self, actions: Sequence[Sequence[float]]) -> None:
        if len(actions) != 2:
            raise ValueError("Expected actions for exactly two agents.")
        for idx, action in enumerate(actions):
            if not isinstance(action, Sequence):
                raise TypeError(f"Action for agent {idx} must be a sequence.")
            if len(action) == 0:
                LOGGER.warning("Agent %s provided an empty action sequence.", idx)

    def _sanitize_rewards(self, rewards: Sequence[float]) -> list[float]:
        return list(rewards[:2]) if len(rewards) >= 2 else list(rewards) + [0.0] * (2 - len(rewards))

    def _build_info(self, done: bool) -> dict[str, Any]:
        info = dict(self._task.info(self._state))
        info.update({"task": self._task.name, "step": self._step_count, "done": done})
        return info


def build_multi_robot_world(cfg: Mapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
    """Placeholder for constructing the Newton model and state."""

    _unused(cfg)
    return None, {"joints": [], "objects": []}


def setup_cameras(model: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Placeholder for configuring per-agent cameras."""

    _unused(model, cfg)
    return {"agent_a": None, "agent_b": None}


def make_solver(cfg: Mapping[str, Any]) -> Any:
    """Placeholder for creating the Newton physics solver."""

    _unused(cfg)
    return None


def obs_i(agent_id: int, instruction: str, extras: Mapping[str, Any]) -> dict[str, Any]:
    """Construct a placeholder observation for the specified agent."""

    return {
        "agent_id": agent_id,
        "instruction": instruction,
        "task": extras.get("task"),
        "metadata": dict(extras),
    }


def _unused(*args: Any, **kwargs: Any) -> None:
    """Helper for silencing unused argument warnings in placeholders."""
