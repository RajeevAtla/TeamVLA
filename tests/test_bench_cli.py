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

    def fake_benchmark(tasks, policy, **kwargs):  # noqa: ANN001
        called["tasks"] = list(tasks)
        called["kwargs"] = kwargs
        _unused(policy)
        return [{"success": True, "steps": 1, "collisions": [0.0], "coordination": [1.0]}]

    monkeypatch.setattr(bench, "benchmark", fake_benchmark)
    output = tmp_path / "summary.json"
    bench.main(["--tasks", "lift", "--episodes", "1", "--output", str(output)])
    assert output.exists()
    assert "lift" in called["tasks"]


def _unused(*_: object) -> None:
    pass

