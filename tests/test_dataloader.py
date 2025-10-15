"""Tests for MultiTaskDataset indexing and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data.dataloader import MultiTaskDataset
from data.writer import EpisodeWriter


def _meta(task: str, episode_id: str | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {"task": task, "success": False}
    if episode_id is not None:
        meta["episode_id"] = episode_id
    return meta


def _step(task: str) -> dict[str, Any]:
    return {
        "rgb_a": np.zeros((1, 1, 3), dtype=np.uint8),
        "rgb_b": np.zeros((1, 1, 3), dtype=np.uint8),
        "q_a": np.zeros(2),
        "q_b": np.zeros(2),
        "action_a": np.zeros(2),
        "action_b": np.zeros(2),
        "grip_a": 0.0,
        "grip_b": 0.0,
        "instruction": "demo",
        "task": task,
        "success": False,
    }


def _write_episode(tmp_path: Path, task: str, episode_id: str, steps: int) -> None:
    writer = EpisodeWriter(tmp_path, auto_episode_id=False)
    writer.start_episode(_meta(task, episode_id))
    for _ in range(steps):
        writer.add_step(_step(task))
    writer.end_episode()


def test_dataset_indexes_steps(tmp_path: Path) -> None:
    _write_episode(tmp_path, "lift", "0001", steps=2)
    dataset = MultiTaskDataset([tmp_path])
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["task"] == "lift"


def test_dataset_filters_by_task(tmp_path: Path) -> None:
    _write_episode(tmp_path, "lift", "0001", steps=1)
    _write_episode(tmp_path, "drawer", "0002", steps=1)
    dataset = MultiTaskDataset([tmp_path], tasks=["drawer"])
    assert len(dataset) == 1
    assert dataset[0]["task"] == "drawer"


def test_dataset_applies_transforms(tmp_path: Path) -> None:
    _write_episode(tmp_path, "lift", "0001", steps=1)

    def mark(step: dict[str, Any]) -> dict[str, Any]:
        step = dict(step)
        step["mark"] = True
        return step

    dataset = MultiTaskDataset([tmp_path], transforms=[mark])
    assert dataset[0]["mark"] is True


def test_dataset_limit_per_task(tmp_path: Path) -> None:
    _write_episode(tmp_path, "lift", "0001", steps=1)
    _write_episode(tmp_path, "lift", "0002", steps=1)
    dataset = MultiTaskDataset([tmp_path], limit_per_task=1)
    assert len(dataset) == 1

