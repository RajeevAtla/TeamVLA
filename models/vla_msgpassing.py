"""Message-passing VLA model variant."""

from __future__ import annotations

from typing import Any, Mapping

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from models.vla_singlebrain import SingleBrainVLA, _require_torch


class MsgPassingVLA(SingleBrainVLA):
    """Extends SingleBrainVLA with lightweight message passing."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        _require_torch()
        dim = self._fusion_dim
        self._msg_proj_a = nn.Linear(dim, dim)
        self._msg_proj_b = nn.Linear(dim, dim)

    def _fuse(self, feat_a: torch.Tensor, feat_b: torch.Tensor, feat_text: torch.Tensor) -> torch.Tensor:
        fused = super()._fuse(feat_a, feat_b, feat_text)
        message_a = torch.tanh(self._msg_proj_a(fused))
        message_b = torch.tanh(self._msg_proj_b(fused))
        combined = fused + 0.5 * (message_a + message_b)
        return combined
