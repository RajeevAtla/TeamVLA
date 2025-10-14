"""Common loss utilities for TeamVLA training."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


def _require_torch() -> None:
    if torch is None or F is None:  # pragma: no cover
        raise ImportError("PyTorch is required for loss computation.")


def huber_action_loss(pred: Any, target: Any, delta: float = 1.0) -> Any:
    """Compute smooth L1 loss between predicted and target joint deltas."""

    _require_torch()
    return F.smooth_l1_loss(pred, target, beta=delta)


def grip_bce_loss(pred: Any, target: Any) -> Any:
    """Binary cross entropy loss for gripper logits."""

    _require_torch()
    return F.binary_cross_entropy_with_logits(pred, target)


def sync_loss(ee_a: Any, ee_b: Any, mode: str, phase: Any) -> Any:
    """Penalize disagreement between end-effector poses during synchronized phases."""

    _require_torch()
    diff = ee_a - ee_b
    weight = torch.ones_like(phase, dtype=ee_a.dtype) if torch.is_tensor(phase) else 1.0
    _unused(mode)
    return (diff.pow(2).mean(dim=-1) * weight).mean()


def collision_penalty(collisions: Any, alpha: float) -> Any:
    """Scale collision impulses by multiplier alpha."""

    _require_torch()
    return alpha * collisions.abs().mean()


def _unused(*_: object) -> None:
    """Helper for unused placeholder arguments."""
