"""Data schema definitions and validation helpers for TeamVLA episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

EPISODE_FILE_VERSION = 1


class SchemaError(ValueError):
    """Raised when dataset entries violate the expected schema."""


@dataclass(slots=True)
class FieldSpec:
    """Specification describing how a single step field should look."""

    name: str
    types: tuple[type, ...]
    required: bool = True
    ndim: int | None = None
    last_dim: int | None = None
    dtype: tuple[type, ...] | type | None = None
    description: str = ""

    def validate(self, value: Any) -> None:
        if not isinstance(value, self.types):
            raise SchemaError(f"Field '{self.name}' must be of type {self.types}.")
        if isinstance(value, np.ndarray):
            self._validate_array(value)

    def _validate_array(self, array: NDArray[Any]) -> None:
        if self.ndim is not None and array.ndim != self.ndim:
            raise SchemaError(
                f"Field '{self.name}' must have {self.ndim} dimensions; received {array.ndim}."
            )
        if self.last_dim is not None and array.shape[-1] != self.last_dim:
            raise SchemaError(
                f"Field '{self.name}' must have a final dimension of {self.last_dim}; "
                f"received {array.shape[-1]}."
            )
        if self.dtype is not None:
            expected = self.dtype if isinstance(self.dtype, tuple) else (self.dtype,)
            expected_types = {np.dtype(dt).type for dt in expected}
            if array.dtype.type not in expected_types:
                raise SchemaError(
                    f"Field '{self.name}' must have dtype in {expected}; received {array.dtype}."
                )


STEP_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("rgb_a", (np.ndarray,), ndim=3, last_dim=3, dtype=(np.uint8,), description="Agent A RGB frame"),
    FieldSpec("rgb_b", (np.ndarray,), ndim=3, last_dim=3, dtype=(np.uint8,), description="Agent B RGB frame"),
    FieldSpec("q_a", (np.ndarray,), description="Agent A joint configuration"),
    FieldSpec("q_b", (np.ndarray,), description="Agent B joint configuration"),
    FieldSpec("action_a", (np.ndarray,), description="Previous action for agent A"),
    FieldSpec("action_b", (np.ndarray,), description="Previous action for agent B"),
    FieldSpec("grip_a", (float, np.floating), description="Normalized gripper command"),
    FieldSpec("grip_b", (float, np.floating), description="Normalized gripper command"),
    FieldSpec("instruction", (str,), description="Natural language instruction"),
    FieldSpec("task", (str,), description="Task identifier"),
    FieldSpec("success", (bool,), description="Step-level success indicator"),
)

OPTIONAL_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("depth_a", (np.ndarray,), required=False, ndim=3, last_dim=1, dtype=(np.float32, np.float64)),
    FieldSpec("depth_b", (np.ndarray,), required=False, ndim=3, last_dim=1, dtype=(np.float32, np.float64)),
    FieldSpec("phase", (str,), required=False),
    FieldSpec("timestamp", (float, int, np.floating, np.integer), required=False),
)

REQUIRED_FIELD_NAMES = {spec.name for spec in STEP_SPECS if spec.required}
ALL_FIELD_SPECS = {spec.name: spec for spec in (*STEP_SPECS, *OPTIONAL_FIELDS)}


@dataclass(slots=True)
class EpisodeMeta:
    """Metadata describing an episode."""

    task: str
    episode_id: str
    success: bool
    num_steps: int
    version: int = EPISODE_FILE_VERSION


def validate_step(step: Mapping[str, Any]) -> None:
    """Validate a single step dictionary against the schema."""

    missing = REQUIRED_FIELD_NAMES - set(step.keys())
    if missing:
        raise SchemaError(f"Step is missing required fields: {sorted(missing)}")

    for name, value in step.items():
        spec = ALL_FIELD_SPECS.get(name)
        if spec is None:
            # Unrecognised fields are allowed but should not be numpy scalars.
            continue
        spec.validate(value)


def validate_episode_meta(meta: Mapping[str, Any], *, num_steps: int | None = None) -> EpisodeMeta:
    """Validate and coerce episode metadata."""

    task = str(meta.get("task", "")).strip()
    if not task:
        raise SchemaError("Episode metadata must include a non-empty 'task'.")

    episode_id = str(meta.get("episode_id", "")).strip()
    if not episode_id:
        raise SchemaError("Episode metadata must include 'episode_id'.")

    success = bool(meta.get("success", False))
    steps = int(meta.get("num_steps", num_steps or 0))
    if steps < 0:
        raise SchemaError("Episode metadata cannot report a negative 'num_steps'.")
    version = int(meta.get("version", EPISODE_FILE_VERSION))
    return EpisodeMeta(task=task, episode_id=episode_id, success=success, num_steps=steps, version=version)


def validate_episode(steps: Iterable[Mapping[str, Any]]) -> None:
    """Validate every step in an iterable."""

    for index, step in enumerate(steps):
        try:
            validate_step(step)
        except SchemaError as exc:  # pragma: no cover - defensive
            raise SchemaError(f"Invalid step at index {index}: {exc}") from exc


def ensure_np_dtype(array: Any, dtype: np.dtype) -> NDArray[Any]:
    """Return a numpy array with the requested dtype."""

    return np.asarray(array, dtype=dtype)
