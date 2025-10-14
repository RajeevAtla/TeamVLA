"""Tests for evaluation metrics."""

from __future__ import annotations

from eval import metrics


def test_success_at_T_counts_success() -> None:
    trajs = [
        {"success": True, "steps": 5},
        {"success": False, "steps": 10},
    ]
    assert metrics.success_at_T(trajs, horizon=6) == 0.5


def test_time_to_success_returns_step() -> None:
    assert metrics.time_to_success({"step_success": 4}) == 4.0
    assert metrics.time_to_success({}) is None


def test_coordination_score_averages_values() -> None:
    traj = {"coordination": [1.0, 0.8, 0.5]}
    score = metrics.coordination_score(traj, epsilon=0.3)
    assert 0 <= score <= 1


def test_aggregate_results_produces_means() -> None:
    results = [{"success": True, "collision": 1.0}, {"success": False, "collision": 3.0}]
    agg = metrics.aggregate_results(results)
    assert agg["success_rate"] == 0.5
    assert agg["collision"] == 2.0
