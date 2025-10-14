"""Tests for optimizer and scheduler helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train import schedulers


def test_make_optimizer_returns_adamw() -> None:
    model = torch.nn.Linear(4, 4)
    optimizer = schedulers.make_optimizer(model, {"name": "adamw", "lr": 0.01})
    assert isinstance(optimizer, torch.optim.AdamW)


def test_cosine_with_warmup_decreases_after_warmup() -> None:
    model = torch.nn.Linear(4, 4)
    optimizer = schedulers.make_optimizer(model, {"name": "adamw", "lr": 0.01})
    scheduler = schedulers.cosine_with_warmup(optimizer, warmup_steps=1, total_steps=10)
    lrs = []
    for step in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[1] <= lrs[0]
