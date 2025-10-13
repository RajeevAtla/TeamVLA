"""Data schema definitions and validation helpers for TeamVLA episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class SchemaError(ValueError):
    """Raised when dataset entries violate the expected schema."""


FieldType = tuple[type, ...] | type


STEP_FIELDS: Mapping[str, FieldType] = {
    "rgb_a": np.ndarray,
    "rgb_b": np.ndarray,
    "q_a": np.ndarray,
    "q_b": np.ndarray,
    "action_a": np.ndarray,
    "action_b": np.ndarray,
    "grip_a": (float, np.floating),
    "grip_b": (float, np.floating),
    "instruction": str,
    "task": str,
    "success": bool,
}


@dataclass(slots=True)
class EpisodeMeta:
    """Metadata describing an episode."""

    task: str
    episode_id: str
    success: bool


def validate_step(step: Mapping[str, Any]) -> None:
    """Validate a single step dictionary against the schema."""

    for field, expected in STEP_FIELDS.items():
        if field not in step:
            raise SchemaError(f"Missing field '{field}' in step data.")
        if not isinstance(step[field], expected):
            raise SchemaError(f"Field '{field}' must be of type {expected}.")


def validate_episode_meta(meta: Mapping[str, Any]) -> EpisodeMeta:
    """Validate and coerce episode metadata."""

    task = str(meta.get("task"))
    if not task:
        raise SchemaError("Episode metadata must include a non-empty 'task'.")
    episode_id = str(meta.get("episode_id"))
    if not episode_id:
        raise SchemaError("Episode metadata must include 'episode_id'.")
    success = bool(meta.get("success", False))
    return EpisodeMeta(task=task, episode_id=episode_id, success=success)


def ensure_np_dtype(array: Any, dtype: np.dtype) -> NDArray[Any]:
    """Return a numpy array with the requested dtype."""

    return np.asarray(array, dtype=dtype)
