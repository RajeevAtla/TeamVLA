"""Tests for evaluation metrics."""

from __future__ import annotations

from eval import metrics
from tests.utils import rollout_summary


def test_success_at_T_counts_success() -> None:
    trajs = [
        rollout_summary(success=True, steps=5),
        rollout_summary(success=False, steps=8),
    ]
    assert metrics.success_at_T(trajs, horizon=6) == 0.5


def test_time_to_success_returns_step() -> None:
    assert metrics.time_to_success({"step_success": 4}) == 4.0
    assert metrics.time_to_success({}) is None


def test_coordination_score_averages_values() -> None:
    traj = rollout_summary(coordination=[1.0, 0.8, 0.5])
    score = metrics.coordination_score(traj, epsilon=0.3)
    assert 0 <= score <= 1


def test_aggregate_results_produces_stats() -> None:
    trajs = [
        rollout_summary(success=True, step_success=5),
        rollout_summary(success=False),
    ]
    summary = metrics.aggregate_results(trajs, horizon=12)
    assert summary.success_rate == 0.5
    assert summary.success_at_T == 0.5
    assert summary.average_time_to_success == 5.0
    assert 0 <= summary.coordination <= 1
    assert summary.collision >= 0.0
