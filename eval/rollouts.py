"""Rollout utilities for evaluating TeamVLA policies."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


PolicyFn = Callable[[list[Mapping[str, Any]]], list[Any]]


@dataclass(slots=True)
class RolloutConfig:
    max_steps: int
    instruction: str
    seed: int | None = None
    record_infos: bool = False


@dataclass(slots=True)
class RolloutResult:
    task: str
    instruction: str
    steps: int
    success: bool
    collisions: list[float] = field(default_factory=list)
    coordination: list[float] = field(default_factory=list)
    rewards: list[list[float]] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)
    step_success: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task": self.task,
            "instruction": self.instruction,
            "steps": self.steps,
            "success": self.success,
            "collisions": self.collisions,
            "coordination": self.coordination,
            "rewards": self.rewards,
        }
        if self.step_success is not None:
            payload["step_success"] = self.step_success
        if self.info:
            payload["info"] = self.info
        return payload


def run_episode(env: Any, policy: PolicyFn, cfg: RolloutConfig) -> RolloutResult:
    """Execute a single policy rollout in the environment."""

    if cfg.seed is not None:
        _seed_environment(env, cfg.seed)

    observations = env.reset(cfg.instruction)
    info_proto: MutableMapping[str, Any] = getattr(
        env, "get_state_dict", lambda: {"task": cfg.instruction}
    )()
    task_name = info_proto.get("task", getattr(env, "_task", cfg.instruction))

    trajectory = RolloutResult(task=task_name, instruction=cfg.instruction, steps=0, success=False)

    for step in range(cfg.max_steps):
        actions = policy(observations)
        observations, rewards, done, info = env.step(actions)
        trajectory.steps = step + 1
        trajectory.rewards.append([float(r) for r in rewards])
        trajectory.collisions.append(float(info.get("collision", 0.0)))
        trajectory.coordination.append(float(info.get("coordination", 1.0)))
        if cfg.record_infos:
            trajectory.info.append(dict(info))
        success = bool(info.get("task_success", False))
        if success and not trajectory.success:
            trajectory.step_success = step + 1
        trajectory.success = success or trajectory.success
        if success or done:
            break
    return trajectory


def run_suite(
    env: Any,
    policy: PolicyFn,
    tasks: Iterable[str],
    n_eps: int,
    *,
    max_steps: int,
    seed: int | None = None,
    record_infos: bool = False,
) -> list[dict[str, Any]]:
    """Run multiple episodes across a suite of tasks."""

    results: list[dict[str, Any]] = []
    for task in tasks:
        env.set_task(task)
        for episode_idx in range(n_eps):
            cfg = RolloutConfig(
                max_steps=max_steps,
                instruction=f"Execute {task}",
                seed=None if seed is None else seed + episode_idx,
                record_infos=record_infos,
            )
            traj = run_episode(env, policy, cfg)
            results.append(traj.to_dict())
    LOGGER.info("Completed %s rollouts", len(results))
    return results


def _seed_environment(env: Any, seed: int) -> None:
    setter = getattr(env, "set_seed", None)
    if callable(setter):
        setter(seed)
        return
    if hasattr(env, "_rng"):
        try:
            import numpy as np

            env._rng = np.random.default_rng(seed)  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover
            pass
