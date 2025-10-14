"""Tests for evaluation benchmark CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import bench


def test_parse_args_defaults() -> None:
    args = bench.parse_args([])
    assert args.episodes == 1
    assert "lift" in args.tasks


def test_main_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_run_suite(env, policy, tasks, n_eps, unseen):  # noqa: ANN001
        called["tasks"] = tasks
        return [{"success": True, "collision": 0.0, "steps": 1}]

    monkeypatch.setattr(bench, "run_suite", fake_run_suite)
    monkeypatch.setattr(bench, "NewtonMAEnv", lambda cfg: type("Env", (), {"close": lambda self: None})())
    output = tmp_path / "summary.json"
    bench.main(["--tasks", "lift", "--episodes", "1", "--output", str(output)])
    assert output.exists()
    assert "lift" in called["tasks"]
