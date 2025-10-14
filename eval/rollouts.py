"""Rollout utilities for evaluating TeamVLA policies."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping

LOGGER = logging.getLogger(__name__)


def run_episode(
    env: Any,
    policy: Callable[[list[Mapping[str, Any]]], list[Any]],
    instruction: str,
    max_steps: int,
) -> dict[str, Any]:
    """Execute a single policy rollout in the environment."""

    observations = env.reset(instruction)
    trajectory: dict[str, Any] = {
        "steps": 0,
        "success": False,
        "collisions": [],
        "coordination": [],
    }
    for step in range(max_steps):
        actions = policy(observations)
        observations, rewards, done, info = env.step(actions)
        trajectory["steps"] = step + 1
        trajectory["success"] = bool(done and info.get("task_success", False))
        trajectory["collisions"].append(info.get("collision", 0.0))
        trajectory["coordination"].append(info.get("coordination", 1.0))
        if trajectory["success"]:
            trajectory["step_success"] = step + 1
            break
        if done:
            break
    return trajectory


def run_suite(
    env: Any,
    policy: Callable[[list[Mapping[str, Any]]], list[Any]],
    tasks: Iterable[str],
    n_eps: int,
    unseen: bool,
) -> list[dict[str, Any]]:
    """Run multiple episodes across a suite of tasks."""

    del unseen  # Placeholder; reserved for future domain split handling
    results: list[dict[str, Any]] = []
    cfg = getattr(env, "_config", None)
    default_steps = getattr(cfg, "max_steps", 200)
    for task in tasks:
        env.set_task(task)
        for _ in range(n_eps):
            traj = run_episode(env, policy, instruction=f"Execute {task}", max_steps=default_steps)
            results.append(traj)
    LOGGER.info("Completed %s rollouts", len(results))
    return results
