"""Tests for training loss helpers."""

from __future__ import annotations

import pytest

from train import losses

torch = pytest.importorskip("torch")


def test_huber_action_loss_returns_scalar() -> None:
    pred = torch.zeros((2, 4))
    target = torch.ones((2, 4))
    value = losses.huber_action_loss(pred, target)
    assert value.shape == ()


def test_grip_bce_loss() -> None:
    pred = torch.zeros((2, 1))
    target = torch.ones((2, 1))
    value = losses.grip_bce_loss(pred, target)
    assert value.shape == ()


def test_sync_loss_handles_phase_tensor() -> None:
    a = torch.zeros((2, 3))
    b = torch.ones((2, 3))
    phase = torch.ones((2,))
    value = losses.sync_loss(a, b, mode="cooperative", phase=phase)
    assert value > 0


def test_collision_penalty_scales_values() -> None:
    collisions = torch.tensor([1.0, -2.0, 3.0])
    penalty = losses.collision_penalty(collisions, alpha=0.5)
    assert penalty == pytest.approx(1.0)


def test_compute_behavior_cloning_losses_combines_terms() -> None:
    outputs = {
        "pred_a": torch.zeros((2, 4)),
        "pred_b": torch.zeros((2, 4)),
        "grip_logits_a": torch.zeros((2, 1)),
        "grip_logits_b": torch.zeros((2, 1)),
    }
    batch = {
        "action_a": torch.ones((2, 4)),
        "action_b": torch.ones((2, 4)),
        "grip_a": torch.ones((2, 1)),
        "grip_b": torch.ones((2, 1)),
    }
    result = losses.compute_behavior_cloning_losses(outputs, batch, {"actions": 1.0, "grip": 0.5})
    assert "total" in result
    assert result["total"] > 0
