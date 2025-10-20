"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml

from tests.utils import random_rgb, rollout_summary


@pytest.fixture(autouse=True, scope="session")
def _set_thread_environment() -> None:
    """Limit torch thread usage during tests to keep runtime predictable."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch optional in some environments
        pass


if TYPE_CHECKING:
    from envs import NewtonMAEnv


@pytest.fixture
def lift_env() -> Iterator["NewtonMAEnv"]:
    """Provide a lightweight NewtonMAEnv instance for tests that need environment context."""

    from envs import NewtonMAEnv

    env = NewtonMAEnv({"task_name": "lift", "max_steps": 5, "seed": 123})
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def synthetic_rollout() -> dict[str, object]:
    """Return a representative rollout summary used by evaluation tests."""

    return rollout_summary(
        success=True,
        steps=6,
        collisions=[0.0, 0.05],
        coordination=[1.0, 0.9, 0.8],
        step_success=4,
    )


@pytest.fixture
def synthetic_trajectory() -> list[dict[str, object]]:
    """Produce a short trajectory with observation-like payloads for dataset tests."""

    trajectory: list[dict[str, object]] = []
    for idx in range(3):
        trajectory.append(
            {
                "rgb_a": random_rgb(seed=idx),
                "rgb_b": random_rgb(seed=idx + 17),
                "action_a": np.zeros(4, dtype=np.float32),
                "action_b": np.zeros(4, dtype=np.float32),
                "reward_a": float(idx),
                "reward_b": float(idx),
                "instruction": "Demo instruction",
                "task": "lift",
            }
        )
    return trajectory


@pytest.fixture
def config_loader() -> Callable[[str], dict[str, object]]:
    """Return a helper that loads YAML configuration files from configs/."""

    def _load(name: str) -> dict[str, object]:
        path = Path("configs") / name
        if not path.exists():
            raise FileNotFoundError(f"Config {name} not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    return _load
