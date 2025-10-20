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


def test_validate_step_rejects_wrong_dtype() -> None:
    step = _valid_step()
    step["rgb_a"] = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(schema.SchemaError):
        schema.validate_step(step)


def test_validate_episode_meta_returns_dataclass() -> None:
    meta = schema.validate_episode_meta(
        {"task": "lift", "episode_id": "0001", "success": True, "num_steps": 3}
    )
    assert meta.task == "lift"
    assert meta.success
    assert meta.num_steps == 3


def test_validate_episode_reports_negative_steps() -> None:
    with pytest.raises(schema.SchemaError):
        schema.validate_episode_meta({"task": "lift", "episode_id": "bad", "num_steps": -1})
