"""Tests for CLI scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import collect_demos, render_videos


def test_collect_parse_args_defaults() -> None:
    args = collect_demos.parse_args([])
    assert args.episodes == 1
    assert args.task == "lift"


def test_collect_invokes_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"start": 0, "add": 0, "end": 0}

    class FakeWriter:
        def __init__(self, *_, **__):
            pass

        def start_episode(self, *_, **__):
            calls["start"] += 1

        def add_step(self, *_, **__):
            calls["add"] += 1

        def end_episode(self, *_, **__):
            calls["end"] += 1

        def close(self):
            pass

    class FakeEnv:
        def __init__(self, *_):
            self._config = type("Cfg", (), {"max_steps": 2})()
            self._phases = ("reach", "act")

        def reset(self, *_):
            return [{"rgb": np.zeros((48, 48, 3), dtype=np.uint8)}, {"rgb": np.zeros((48, 48, 3), dtype=np.uint8)}]

        def step(self, *_):
            return [{}, {}], [0.0, 0.0], True, {"task_success": True, "step": 1, "collision": 0.0, "coordination": 1.0}

        def close(self):
            pass

    monkeypatch.setattr(collect_demos, "EpisodeWriter", FakeWriter)
    monkeypatch.setattr(collect_demos, "NewtonMAEnv", FakeEnv)
    monkeypatch.setitem(collect_demos.SCRIPTED_POLICIES, "lift", lambda env, pm, obs: [np.zeros(4), np.zeros(4)])
    args = collect_demos.parse_args(["--episodes", "1", "--out", str(tmp_path)])
    collect_demos.collect(args)
    assert calls["start"] == 1
    assert calls["end"] == 1


def test_render_parse_args_defaults() -> None:
    args = render_videos.parse_args([])
    assert args.fps == 20


def test_render_episode_writes_file(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode.npz"
    np.savez(episode_path, steps=np.array([{ "rgb_a": np.zeros((48,48,3), dtype=np.uint8)}], dtype=object))
    render_videos.render_episode(episode_path, tmp_path / "video")
    assert (tmp_path / "video.npy").exists() or (tmp_path / "video.mp4").exists()

