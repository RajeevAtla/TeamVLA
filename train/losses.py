"""Common loss utilities for TeamVLA training."""

from __future__ import annotations

from collections.abc import Mapping
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


def huber_action_loss(pred: Any, target: Any, delta: float = 1.0, reduction: str = "mean") -> Any:
    """Compute smooth L1 loss between predicted and target joint deltas."""

    _require_torch()
    return F.smooth_l1_loss(pred, target, beta=delta, reduction=reduction)


def grip_bce_loss(pred: Any, target: Any, reduction: str = "mean") -> Any:
    """Binary cross entropy loss for gripper logits."""

    _require_torch()
    return F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)


def sync_loss(ee_a: Any, ee_b: Any, phase_mask: Any | None = None) -> Any:
    """Penalize disagreement between end-effector poses during synchronized phases."""

    _require_torch()
    diff = ee_a - ee_b
    sq = diff.pow(2).sum(dim=-1)
    if phase_mask is not None:
        weight = phase_mask.float()
        sq = sq * weight
        return sq.sum() / weight.sum().clamp_min(1.0)
    return sq.mean()


def collision_penalty(collisions: Any, alpha: float) -> Any:
    """Scale collision impulses by multiplier alpha."""

    _require_torch()
    return alpha * collisions.abs().mean()


def compute_behavior_cloning_losses(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate the standard BC losses with configurable weights."""

    _require_torch()
    weights = {"actions": 1.0, "grip": 1.0, "sync": 0.0, "collision": 0.0, **(weights or {})}
    loss_dict: dict[str, Any] = {}
    total = torch.zeros((), device=_tensor_device(outputs))

    if "pred_a" in outputs and "action_a" in batch:
        act_loss_a = huber_action_loss(outputs["pred_a"], batch["action_a"], delta=1.0)
        act_loss_b = huber_action_loss(outputs["pred_b"], batch["action_b"], delta=1.0)
        action_loss = 0.5 * (act_loss_a + act_loss_b)
        loss_dict["action"] = action_loss
        total = total + weights.get("actions", 1.0) * action_loss

    if "grip_logits_a" in outputs and "grip_a" in batch:
        grip_loss_a = grip_bce_loss(outputs["grip_logits_a"], batch["grip_a"].float())
        grip_loss_b = grip_bce_loss(outputs["grip_logits_b"], batch["grip_b"].float())
        grip_loss = 0.5 * (grip_loss_a + grip_loss_b)
        loss_dict["grip"] = grip_loss
        total = total + weights.get("grip", 1.0) * grip_loss

    if weights.get("sync", 0.0) > 0 and "ee_pose_a" in batch and "ee_pose_b" in batch:
        phase_mask = batch.get("sync_mask")
        sync = sync_loss(batch["ee_pose_a"], batch["ee_pose_b"], phase_mask)
        loss_dict["sync"] = sync
        total = total + weights["sync"] * sync

    if weights.get("collision", 0.0) > 0 and "collision" in batch:
        coll = collision_penalty(batch["collision"], alpha=1.0)
        loss_dict["collision"] = coll
        total = total + weights["collision"] * coll

    loss_dict["total"] = total
    return loss_dict


def _tensor_device(outputs: Mapping[str, Any]) -> torch.device:
    for value in outputs.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")

