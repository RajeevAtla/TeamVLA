"""Vision encoder factories for TeamVLA models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(slots=True)
class VisionEncoderConfig:
    """Configuration for the default convolutional vision encoder."""

    in_channels: int = 3
    channels: Sequence[int] = (32, 64, 128)
    out_dim: int = 128
    use_batch_norm: bool = True
    dropout: float = 0.0


def build_vision_encoder(
    name: str = "conv",
    *,
    pretrained: bool = False,
    out_dim: int = 128,
    config: VisionEncoderConfig | None = None,
) -> Any:
    """Build a lightweight vision encoder.

    Parameters
    ----------
    name:
        Encoder variant. Currently only ``"conv"`` is implemented.
    pretrained:
        Reserved for future use; currently ignored to keep the project self-contained.
    out_dim:
        Output embedding dimensionality.
    config:
        Optional configuration overriding defaults for the convolutional encoder.
    """

    _unused(pretrained)
    if name != "conv":  # pragma: no cover - other variants can be added later
        raise ValueError(f"Unknown vision encoder variant '{name}'.")
    cfg = config or VisionEncoderConfig(out_dim=out_dim)
    cfg = VisionEncoderConfig(
        in_channels=cfg.in_channels,
        channels=tuple(cfg.channels),
        out_dim=out_dim,
        use_batch_norm=cfg.use_batch_norm,
        dropout=cfg.dropout,
    )
    _require_torch()
    return _ConvVisionEncoder(cfg)


def forward_vision(encoder: Any, images: Any) -> Any:
    """Forward images through the provided encoder."""

    _require_torch()
    return encoder(images)


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - guard branch
        raise ImportError("PyTorch is required to use the vision encoders.")


class _ConvVisionEncoder(nn.Module if nn is not None else object):
    """Simple convolutional encoder with adaptive pooling and linear projection."""

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        _require_torch()
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = cfg.in_channels
        for idx, out_channels in enumerate(cfg.channels):
            layers.extend(
                _conv_block(
                    in_channels,
                    out_channels,
                    use_bn=cfg.use_batch_norm,
                    name=f"stage{idx}",
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        head_layers: list[nn.Module] = [nn.Linear(in_channels, cfg.out_dim)]
        if cfg.dropout > 0.0:
            head_layers.insert(0, nn.Dropout(cfg.dropout))
        self.head = nn.Sequential(*head_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        images = images.float() / 255.0
        encoded = self.features(images)
        pooled = self.pool(encoded).flatten(1)
        return self.head(pooled)


def _conv_block(in_channels: int, out_channels: int, *, use_bn: bool, name: str) -> Iterable[nn.Module]:
    del name  # Reserved for future debugging hooks
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.extend([nn.ReLU(inplace=True)])
    return layers


def _unused(*_: object) -> None:
    """Placeholder helper for unused arguments."""

