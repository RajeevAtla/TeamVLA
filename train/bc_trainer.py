"""Behavior Cloning trainer scaffolding for TeamVLA."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

try:  # pragma: no cover - optional torch dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None

from data.dataloader import make_dataloader
from models import MsgPassingVLA, SingleBrainVLA
from train import losses, schedulers

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingState:
    epoch: int
    global_step: int
    best_metric: float


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def build_model(cfg: Mapping[str, Any]) -> Any:
    """Construct the configured VLA model."""

    model_type = cfg.get("type", "single_brain")
    if model_type == "single_brain":
        return SingleBrainVLA(cfg)
    if model_type == "msg_passing":
        return MsgPassingVLA(cfg)
    raise ValueError(f"Unknown model type '{model_type}'.")


def build_data(cfg: Mapping[str, Any]) -> Any:
    """Construct the training dataloader."""

    return make_dataloader(cfg)


def train_one_epoch(model: Any, loader: Any, optimizer: Any, scheduler: Any | None = None) -> dict[str, float]:
    """Run a single epoch of behavior cloning training."""

    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for training.")
    model.train()
    metrics = {"loss": 0.0, "batches": 0}
    for batch in loader:
        optimizer.zero_grad()
        outputs = model.loss(batch)
        total = outputs["total"]
        total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        metrics["loss"] += float(total.detach().cpu())
        metrics["batches"] += 1
    if metrics["batches"]:
        metrics["loss"] /= metrics["batches"]
    return metrics


def evaluate(model: Any, env: Any, tasks: list[str], cfg: Mapping[str, Any]) -> dict[str, float]:
    """Placeholder evaluation loop returning empty metrics."""

    _unused(model, env, tasks, cfg)
    return {"success_rate": 0.0}


def save_checkpoint(path: str | Path, model: Any, optimizer: Any, scheduler: Any | None, step: int, cfg: Mapping[str, Any]) -> None:
    """Persist model and optimizer state to disk."""

    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for checkpointing.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": cfg,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: Any, optimizer: Any | None = None, scheduler: Any | None = None) -> int:
    """Load checkpoint and optionally restore optimizer/scheduler state."""

    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for checkpointing.")
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return int(payload.get("step", 0))


def main(cfg_path: str = "configs/train_bc.yaml") -> None:
    """Entry point for launching the training loop."""

    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    logging.basicConfig(level=getattr(logging, cfg.get("log_level", "INFO")))
    set_seed(int(cfg.get("seed", 0)))

    model = build_model(cfg.get("model", {}))
    dataloader = build_data(cfg.get("dataset", {}))
    optimizer = schedulers.make_optimizer(model, cfg.get("optimizer", {}))
    scheduler = schedulers.cosine_with_warmup(
        optimizer,
        warmup_steps=int(cfg.get("scheduler", {}).get("warmup_steps", 100)),
        total_steps=int(cfg.get("scheduler", {}).get("total_steps", 1000)),
    )
    state = TrainingState(epoch=0, global_step=0, best_metric=float("inf"))

    for epoch in range(int(cfg.get("epochs", 1))):
        metrics = train_one_epoch(model, dataloader, optimizer, scheduler)
        LOGGER.info("Epoch %s | loss=%.4f", epoch, metrics["loss"])
        state.epoch = epoch + 1
        state.global_step += metrics["batches"]

    LOGGER.info("Training complete after %s epochs", state.epoch)


def _unused(*_: Any) -> None:
    """Helper for unused placeholder arguments."""
