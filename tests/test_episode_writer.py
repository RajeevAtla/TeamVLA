"""Tests for EpisodeWriter functionality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.writer import EpisodeWriter


def _meta() -> dict[str, object]:
    return {"task": "lift", "episode_id": "demo", "success": False}


def _step(task: str = "lift") -> dict[str, object]:
    return {
        "rgb_a": np.zeros((2, 2, 3), dtype=np.uint8),
        "rgb_b": np.zeros((2, 2, 3), dtype=np.uint8),
        "q_a": np.zeros(7),
        "q_b": np.zeros(7),
        "action_a": np.zeros(7),
        "action_b": np.zeros(7),
        "grip_a": 0.0,
        "grip_b": 0.0,
        "instruction": "lift",
        "task": task,
        "success": False,
    }


def test_episode_writer_writes_npz(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path)
    writer.start_episode(_meta())
    writer.add_step(_step())
    path = writer.end_episode(success=True)

    with np.load(path, allow_pickle=True) as data:
        meta = data["meta"].item()
        assert meta["success"] is True
        assert meta["num_steps"] == 1
        assert meta["version"] >= 1
        steps = data["steps"].tolist()
        assert len(steps) == 1


def test_writer_auto_generates_episode_id(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, episode_prefix="auto")
    episode_meta = writer.start_episode({"task": "lift"})
    assert episode_meta.episode_id.startswith("auto_")
    writer.add_step(_step())
    writer.end_episode()


def test_ending_without_start_raises(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path)
    with pytest.raises(RuntimeError):
        writer.end_episode()
