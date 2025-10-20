"""Utility helpers shared across the TeamVLA test suite."""

from __future__ import annotations

from typing import Any

import numpy as np


def random_rgb(width: int = 48, height: int = 48, *, seed: int | None = None) -> np.ndarray:
    """Generate a random uint8 RGB image for vision fixtures."""

    rng = np.random.default_rng(seed)
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


def rollout_summary(
    *,
    success: bool = False,
    steps: int = 10,
    collisions: list[float] | None = None,
    coordination: list[float] | None = None,
    step_success: int | None = None,
) -> dict[str, Any]:
    """Create a lightweight rollout summary mapping for evaluation tests."""

    payload: dict[str, Any] = {
        "success": success,
        "steps": steps,
        "collisions": collisions or [0.0],
        "coordination": coordination or [1.0],
    }
    if step_success is not None:
        payload["step_success"] = step_success
    return payload


def assert_close(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-6) -> None:
    """Assert that two numpy arrays are approximately equal."""

    if not np.allclose(actual, expected, atol=atol):
        diff = np.abs(actual - expected).max()
        raise AssertionError(f"Arrays differ by up to {diff}, tolerance {atol}.")
