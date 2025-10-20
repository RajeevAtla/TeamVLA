"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest


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
def lift_env() -> Iterator[NewtonMAEnv]:
    """Provide a lightweight NewtonMAEnv instance for tests that need environment context."""

    from envs import NewtonMAEnv

    env = NewtonMAEnv({"task_name": "lift", "max_steps": 5, "seed": 123})
    try:
        yield env
    finally:
        env.close()
