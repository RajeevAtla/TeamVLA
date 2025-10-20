"""Evaluation metrics for TeamVLA trajectories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrajectoryStats:
    """Summary statistics computed from rollout trajectories."""

    success_rate: float
    success_at_T: float
    average_time_to_success: float | None
    coordination: float
    collision: float


def success_at_T(trajs: Sequence[Mapping[str, object]], horizon: int) -> float:
    """Compute the fraction of episodes that succeeded within ``horizon`` steps."""

    if not trajs:
        return 0.0
    successes = sum(
        1 for traj in trajs if bool(traj.get("success")) and int(traj.get("steps", 0)) <= horizon
    )
    return successes / len(trajs)


def time_to_success(traj: Mapping[str, object]) -> float | None:
    """Return the first timestep at which success occurred, if any."""

    step = traj.get("step_success")
    return float(step) if step is not None else None


def coordination_score(traj: Mapping[str, object], epsilon: float = 0.0) -> float:
    """Compute the fraction of steps satisfying a coordination threshold."""

    contacts_obj: Any = traj.get("coordination", [])
    if not isinstance(contacts_obj, Iterable):
        return 0.0
    contacts = list(contacts_obj)
    if not contacts:
        return 0.0
    satisfying = sum(1 for value in contacts if float(value) >= 1.0 - epsilon)
    return satisfying / len(contacts)


def collision_cost(traj: Mapping[str, object]) -> float:
    """Return average collision magnitude for a trajectory."""

    penalties_obj: Any = traj.get("collisions", [])
    if not isinstance(penalties_obj, Iterable):
        return 0.0
    penalties = list(penalties_obj)
    if not penalties:
        return 0.0
    return float(sum(abs(float(value)) for value in penalties) / len(penalties))


def aggregate_results(
    trajs: Iterable[Mapping[str, object]], horizon: int | None = None
) -> TrajectoryStats:
    """Aggregate per-episode metrics into averaged statistics."""

    trajs = list(trajs)
    if not trajs:
        return TrajectoryStats(0.0, 0.0, None, 0.0, 0.0)

    successes = [bool(traj.get("success")) for traj in trajs]
    success_rate = sum(successes) / len(trajs)

    horizon_value = (
        horizon if horizon is not None else max(int(traj.get("steps", 0)) for traj in trajs)
    )
    sat = success_at_T(trajs, horizon=horizon_value)

    times = [time_to_success(traj) for traj in trajs]
    successful_times = [time for time in times if time is not None]
    avg_time = sum(successful_times) / len(successful_times) if successful_times else None

    coordination = sum(coordination_score(traj, epsilon=0.05) for traj in trajs) / len(trajs)
    collision = sum(collision_cost(traj) for traj in trajs) / len(trajs)

    return TrajectoryStats(
        success_rate=success_rate,
        success_at_T=sat,
        average_time_to_success=avg_time,
        coordination=coordination,
        collision=collision,
    )
