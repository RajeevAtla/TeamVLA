"""Tests for bc_trainer utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from train import bc_trainer, schedulers


class _DummyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        return {"target": torch.tensor([float(idx)], dtype=torch.float32)}


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def loss(self, batch):
        pred = self.weight.expand_as(batch["target"])
        total = torch.mean((pred - batch["target"]) ** 2)
        return {"total": total}


def test_build_model_returns_single_brain() -> None:
    model = bc_trainer.build_model({"type": "single_brain"})
    assert model.__class__.__name__ == "SingleBrainVLA"


def test_train_one_epoch_runs() -> None:
    model = _DummyModel()
    loader = torch.utils.data.DataLoader(_DummyDataset(), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = schedulers.cosine_with_warmup(optimizer, warmup_steps=1, total_steps=10)
    metrics = bc_trainer.train_one_epoch(model, loader, optimizer, scheduler)
    assert metrics["batches"] == 2


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = schedulers.cosine_with_warmup(optimizer, warmup_steps=1, total_steps=10)
    checkpoint_path = tmp_path / "ckpt.pt"
    bc_trainer.save_checkpoint(checkpoint_path, model, optimizer, scheduler, step=5, cfg={"seed": 1})
    step = bc_trainer.load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    assert step == 5
