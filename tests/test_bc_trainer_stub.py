"""Torch-free checkpoint coverage for bc_trainer."""

from __future__ import annotations

from pathlib import Path

from train import bc_trainer


class _Dummy:
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0

    def state_dict(self) -> dict[str, float]:
        return {self._name: self._value}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._value = state[self._name]


def test_checkpoint_roundtrip_without_torch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bc_trainer, "torch", None)
    model = _Dummy("model")
    optimizer = _Dummy("optimizer")
    scheduler = _Dummy("scheduler")
    state = bc_trainer.TrainingState(epoch=2, global_step=12, best_metric=0.75)
    ckpt = tmp_path / "stub.ckpt"

    bc_trainer.save_checkpoint(ckpt, model, optimizer, scheduler, state=state, cfg={"seed": 1})
    restored = bc_trainer.load_checkpoint(ckpt, model, optimizer, scheduler)

    assert restored.epoch == 2
    assert restored.global_step == 12
