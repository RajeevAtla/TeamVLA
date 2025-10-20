"""Optimizers and learning-rate schedulers for TeamVLA training."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, pi
from typing import Any, cast

try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    torch = cast("Any", None)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for optimizer and scheduler helpers.")


def make_optimizer(model: Any, cfg: Mapping[str, Any]) -> Any:
    """Instantiate an optimizer based on configuration."""

    _require_torch()
    name = cfg.get("name", "adamw").lower()
    params = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(cfg.get("lr", 1e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )
    if name == "sgd":  # pragma: no cover - alternative optimizer
        return torch.optim.SGD(
            params,
            lr=float(cfg.get("lr", 1e-3)),
            momentum=float(cfg.get("momentum", 0.9)),
        )
    raise ValueError(f"Unsupported optimizer '{name}'.")


def cosine_with_warmup(optimizer: Any, warmup_steps: int, total_steps: int) -> Any:
    """Create a cosine annealing scheduler with warmup."""

    _require_torch()
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be less than total_steps.")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + cos(pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
