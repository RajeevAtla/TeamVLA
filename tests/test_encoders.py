"""Tests for vision and language encoders."""

from __future__ import annotations

import pytest

from models.encoders import language, vision

torch = pytest.importorskip("torch")


def test_build_vision_encoder_forward_shape() -> None:
    encoder = vision.build_vision_encoder(out_dim=32)
    images = torch.zeros((2, 3, 64, 64))
    features = vision.forward_vision(encoder, images)
    assert features.shape == (2, 32)


def test_tokenize_and_forward_text() -> None:
    encoder = language.build_text_encoder(d_model=16)
    tokens = language.tokenize(["hello", "teamvla"], max_length=8)
    output = language.forward_text(encoder, tokens)
    assert output.shape == (2, 16)
