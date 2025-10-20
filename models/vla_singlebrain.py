"""Single-brain and message-passing VLA architectures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore[assignment]
    import torch.nn as nn  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    torch = cast("Any", None)
    nn = cast("Any", None)

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
        dropout = float(cfg.get("dropout", 0.0))

        self.vision_encoder = build_vision_encoder(out_dim=vision_dim)
        self.text_encoder = build_text_encoder(d_model=text_dim)

        fusion_layers: list[nn.Module] = [
            nn.Linear(vision_dim * 2 + text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            fusion_layers.append(nn.Dropout(dropout))
        fusion_layers.extend(
            [
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(inplace=True),
            ]
        )
        self._fusion = nn.Sequential(*fusion_layers)
        self._fusion_dim = fusion_dim

        self._head_a = nn.Linear(fusion_dim, action_dim)
        self._head_b = nn.Linear(fusion_dim, action_dim)
        self._grip_head_a = nn.Linear(fusion_dim, 1)
        self._grip_head_b = nn.Linear(fusion_dim, 1)

    # ------------------------------------------------------------------#
    # Forward utilities                                                  #
    # ------------------------------------------------------------------#

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        feats = self._encode_modalities(batch)
        fused = self._fuse(*feats)
        actions_a = self._head_a(fused)
        actions_b = self._head_b(fused)
        grip_a = self._grip_head_a(fused)
        grip_b = self._grip_head_b(fused)
        return {
            "pred_a": actions_a,
            "pred_b": actions_b,
            "grip_logits_a": grip_a,
            "grip_logits_b": grip_b,
            "features": fused,
        }

    def act(self, obs: Sequence[Mapping[str, Any]]) -> list[Any]:
        _require_torch()
        self.eval()
        device = (
            next(self.parameters()).device if isinstance(self, nn.Module) else torch.device("cpu")
        )
        with torch.no_grad():
            batch = self._batch_from_observations(obs, device=device)
            outputs = self.forward(batch)
        return [
            outputs["pred_a"].squeeze(0).cpu().numpy(),
            outputs["pred_b"].squeeze(0).cpu().numpy(),
        ]

    def loss(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        _require_torch()
        outputs = self.forward(batch)
        from train import losses  # Local import to avoid circular dependency

        weights = self._cfg.get("loss_weights", {})
        return losses.compute_behavior_cloning_losses(outputs, batch, weights)

    # ------------------------------------------------------------------#
    # Helpers                                                            #
    # ------------------------------------------------------------------#

    def _encode_modalities(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = batch.get("text_tokens")
        if tokens is None:
            text = batch.get("text")
            if text is None:
                raise KeyError("Batch must supply 'text_tokens' or 'text'.")
            if isinstance(text, str):
                text = [text]
            tokens = tokenize(text)
        feat_text = forward_text(self.text_encoder, tokens)
        feat_a = self.vision_encoder(batch["rgb_a"])
        feat_b = self.vision_encoder(batch["rgb_b"])
        return feat_a, feat_b, feat_text

    def _fuse(
        self, feat_a: torch.Tensor, feat_b: torch.Tensor, feat_text: torch.Tensor
    ) -> torch.Tensor:
        fused_input = torch.cat([feat_a, feat_b, feat_text], dim=-1)
        return self._fusion(fused_input)

    def _batch_from_observations(
        self, obs: Sequence[Mapping[str, Any]], *, device: torch.device
    ) -> dict[str, Any]:
        obs_a, obs_b = obs
        rgb_a = torch.tensor(
            obs_a.get("rgb", obs_a.get("rgb_a")), dtype=torch.float32, device=device
        ).unsqueeze(0)
        rgb_b = torch.tensor(
            obs_b.get("rgb", obs_b.get("rgb_b")), dtype=torch.float32, device=device
        ).unsqueeze(0)
        tokens = tokenize([obs_a.get("instruction", "")])
        tokens = {key: value.to(device) for key, value in tokens.items()}
        return {
            "rgb_a": rgb_a,
            "rgb_b": rgb_b,
            "text_tokens": tokens,
            "action_a": torch.zeros(rgb_a.size(0), self._head_a.out_features, device=device),
            "action_b": torch.zeros(rgb_b.size(0), self._head_b.out_features, device=device),
            "grip_a": torch.zeros(rgb_a.size(0), 1, device=device),
            "grip_b": torch.zeros(rgb_b.size(0), 1, device=device),
        }


class MsgPassingVLA(SingleBrainVLA):
    """Extends SingleBrainVLA with lightweight message passing."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        _require_torch()
        dim = self._fusion_dim
        self._message_a = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self._message_b = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self._gate = nn.Sigmoid()

    def _fuse(
        self, feat_a: torch.Tensor, feat_b: torch.Tensor, feat_text: torch.Tensor
    ) -> torch.Tensor:
        base = super()._fuse(feat_a, feat_b, feat_text)
        msg_a = self._message_a(base)
        msg_b = self._message_b(base)
        gate = self._gate(msg_a - msg_b)
        return base + gate * (msg_a + msg_b) * 0.5
