"""Tests for dataset schema validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from data import schema


def _valid_step() -> dict[str, object]:
    zeros = np.zeros((2, 2, 3), dtype=np.uint8)
    return {
        "rgb_a": zeros,
        "rgb_b": zeros,
        "q_a": np.zeros(7),
        "q_b": np.zeros(7),
        "action_a": np.zeros(7),
        "action_b": np.zeros(7),
        "grip_a": 0.0,
        "grip_b": 0.0,
        "instruction": "lift the block",
        "task": "lift",
        "success": False,
    }


def test_validate_step_accepts_valid_data() -> None:
    schema.validate_step(_valid_step())


def test_validate_step_missing_field_raises() -> None:
    step = _valid_step()
    step.pop("rgb_a")
    with pytest.raises(schema.SchemaError):
        schema.validate_step(step)


def test_validate_episode_meta_returns_dataclass() -> None:
    meta = schema.validate_episode_meta({"task": "lift", "episode_id": "0001", "success": True})
    assert meta.task == "lift"
    assert meta.success
