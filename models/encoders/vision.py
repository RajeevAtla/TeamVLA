"""Vision encoder factories for TeamVLA models."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on optional torch installation
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - support environments without torch
    torch = None
    nn = None


def build_vision_encoder(name: str = "linear", pretrained: bool = False, out_dim: int = 128) -> Any:
    """Build a lightweight vision encoder."""

    _unused(pretrained)
    if name != "linear":  # pragma: no cover - additional variants come later
        raise ValueError(f"Unknown vision encoder variant '{name}'.")
    _require_torch()
    return _LinearVisionEncoder(out_dim)


def forward_vision(encoder: Any, images: Any) -> Any:
    """Forward images through the provided encoder."""

    _require_torch()
    return encoder(images)


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - guard branch
        raise ImportError("PyTorch is required to use the vision encoders.")


class _LinearVisionEncoder(nn.Module if nn is not None else object):
    """Simple MLP encoder over flattened RGB frames."""

    def __init__(self, out_dim: int) -> None:
        _require_torch()
        super().__init__()
        self._flatten = nn.Flatten()
        self._mlp = nn.Sequential(
            nn.Linear(3 * 64 * 64, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        images = images.float() / 255.0
        flattened = self._flatten(images)
        return self._mlp(flattened)


def _unused(*_: object) -> None:
    """Placeholder helper for unused arguments."""
