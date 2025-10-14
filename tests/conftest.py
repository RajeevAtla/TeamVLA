"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os

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
