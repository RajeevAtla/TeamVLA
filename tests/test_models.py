"""Tests for VLA model scaffolds."""

from __future__ import annotations

import numpy as np
import pytest

from models.vla_singlebrain import MsgPassingVLA, SingleBrainVLA

torch = pytest.importorskip("torch")


def _dummy_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    images = torch.zeros((batch_size, 3, 64, 64))
    tokens = {"input_ids": torch.zeros((batch_size, 32), dtype=torch.long), "attention_mask": torch.ones((batch_size, 32))}
    return {
        "rgb_a": images,
        "rgb_b": images,
        "text_tokens": tokens,
        "action_a": torch.zeros((batch_size, 8)),
        "action_b": torch.zeros((batch_size, 8)),
        "grip_a": torch.zeros((batch_size, 1)),
        "grip_b": torch.zeros((batch_size, 1)),
    }


def test_single_brain_forward_outputs_actions() -> None:
    model = SingleBrainVLA({"vision_dim": 16, "text_dim": 16, "fusion_dim": 32, "action_dim": 8})
    outputs = model(_dummy_batch())
    assert outputs["pred_a"].shape == (2, 8)
    assert outputs["pred_b"].shape == (2, 8)
    assert outputs["grip_logits_a"].shape == (2, 1)
    assert outputs["features"].shape[1] == 32


def test_single_brain_act_returns_numpy_actions() -> None:
    model = SingleBrainVLA({"vision_dim": 16, "text_dim": 16, "fusion_dim": 32, "action_dim": 8})
    obs = [
        {"rgb": np.zeros((3, 64, 64), dtype=np.float32), "instruction": "test"},
        {"rgb": np.zeros((3, 64, 64), dtype=np.float32), "instruction": "test"},
    ]
    actions = model.act(obs)
    assert len(actions) == 2
    assert actions[0].shape[0] == 8


def test_msg_passing_inherits_single_brain_behavior() -> None:
    model = MsgPassingVLA({"vision_dim": 16, "text_dim": 16, "fusion_dim": 32, "action_dim": 8})
    outputs = model(_dummy_batch())
    assert "pred_a" in outputs
