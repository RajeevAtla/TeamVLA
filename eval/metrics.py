"""Evaluation metrics for TeamVLA trajectories."""

from __future__ import annotations

from typing import Iterable, Mapping


def success_at_T(trajs: Iterable[Mapping[str, object]], horizon: int) -> float:
    """Compute success rate within a horizon."""

    trajs = list(trajs)
    if not trajs:
        return 0.0
    successes = sum(1 for traj in trajs if bool(traj.get("success", False)) and traj.get("steps", 0) <= horizon)
    return successes / len(trajs)


def time_to_success(traj: Mapping[str, object]) -> float | None:
    """Return the first timestep at which success occurred, if any."""

    steps = traj.get("step_success")
    if steps is None:
        return None
    return float(steps)


def coordination_score(traj: Mapping[str, object], epsilon: float) -> float:
    """Compute fraction of steps in which both agents satisfied coordination criteria."""

    contacts = traj.get("coordination", [])
    if not contacts:
        return 0.0
    satisfying = sum(1 for value in contacts if value >= 1.0 - epsilon)
    return satisfying / len(contacts)


def collision_cost(traj: Mapping[str, object]) -> float:
    """Return cumulative collision cost."""

    penalties = traj.get("collisions", [])
    return float(sum(abs(value) for value in penalties))


def aggregate_results(results: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Aggregate per-episode metrics into averages."""

    results = list(results)
    if not results:
        return {"success_rate": 0.0, "collision": 0.0}
    success = sum(1 for item in results if item.get("success")) / len(results)
    collision = sum(float(item.get("collision", 0.0)) for item in results) / len(results)
    return {"success_rate": success, "collision": collision}
