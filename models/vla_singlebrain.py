"""Single-brain VLA baseline model combining vision, language, and action heads."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from models.encoders.language import build_text_encoder, forward_text, tokenize
from models.encoders.vision import build_vision_encoder


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("PyTorch is required for VLA models.")


class SingleBrainVLA(nn.Module if nn is not None else object):
    """Lightweight single-brain policy approximator."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        _require_torch()
        super().__init__()
        self._cfg = cfg
        vision_dim = int(cfg.get("vision_dim", 128))
        text_dim = int(cfg.get("text_dim", 128))
        fusion_dim = int(cfg.get("fusion_dim", 256))
        action_dim = int(cfg.get("action_dim", 8))

        self.vision_encoder = build_vision_encoder(out_dim=vision_dim)
        self.text_encoder = build_text_encoder(d_model=text_dim)
        self._fusion_dim = fusion_dim
        self._fusion = nn.Sequential(
            nn.Linear(vision_dim * 2 + text_dim, fusion_dim),
            nn.ReLU(),
        )
        self._head_a = nn.Linear(fusion_dim, action_dim)
        self._head_b = nn.Linear(fusion_dim, action_dim)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        feats = self._encode_modalities(batch)
        fused = self._fuse(*feats)
        return {
            "pred_a": self._head_a(fused),
            "pred_b": self._head_b(fused),
        }

    def act(self, obs: Sequence[Mapping[str, Any]]) -> list[Any]:
        _require_torch()
        self.eval()
        with torch.no_grad():
            batch = self._batch_from_observations(obs)
            outputs = self.forward(batch)
        return [outputs["pred_a"].squeeze(0).cpu().numpy(), outputs["pred_b"].squeeze(0).cpu().numpy()]

    def loss(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        _require_torch()
        outputs = self.forward(batch)
        from train import losses  # Local import to avoid circular dependency

        loss_a = losses.huber_action_loss(outputs["pred_a"], batch["action_a"])
        loss_b = losses.huber_action_loss(outputs["pred_b"], batch["action_b"])
        total = loss_a + loss_b
        return {"total": total, "loss_a": loss_a, "loss_b": loss_b}

    def _encode_modalities(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = batch.get("text_tokens")
        if tokens is None:
            tokens = tokenize(batch["text"])
        feat_text = forward_text(self.text_encoder, tokens)
        feat_a = self.vision_encoder(batch["rgb_a"])
        feat_b = self.vision_encoder(batch["rgb_b"])
        return feat_a, feat_b, feat_text

    def _fuse(self, feat_a: torch.Tensor, feat_b: torch.Tensor, feat_text: torch.Tensor) -> torch.Tensor:
        fused_input = torch.cat([feat_a, feat_b, feat_text], dim=-1)
        return self._fusion(fused_input)

    def _batch_from_observations(self, obs: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        obs_a, obs_b = obs
        images_a = torch.tensor(obs_a.get("rgb", obs_a.get("rgb_a")), dtype=torch.float32).unsqueeze(0)
        images_b = torch.tensor(obs_b.get("rgb", obs_b.get("rgb_b")), dtype=torch.float32).unsqueeze(0)
        tokens = tokenize([obs_a.get("instruction", "" )])
        return {
            "rgb_a": images_a,
            "rgb_b": images_b,
            "text_tokens": tokens,
            "action_a": torch.zeros(images_a.size(0), self._head_a.out_features),
            "action_b": torch.zeros(images_b.size(0), self._head_b.out_features),
        }
