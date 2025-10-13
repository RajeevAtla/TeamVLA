"""Dataset loader utilities for TeamVLA multi-task training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from data.schema import EpisodeMeta

Transform = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class _IndexEntry:
    path: Path
    step_index: int
    task: str


class MultiTaskDataset:
    """PyTorch-style dataset that iterates over multi-task episodes."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        transforms: Sequence[Transform] | Transform | None = None,
        tasks: Sequence[str] | None = None,
    ) -> None:
        self._roots = [Path(root) for root in roots]
        self._transforms = _normalize_transforms(transforms)
        self._task_filter = set(tasks) if tasks else None
        self._index = self._build_index()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._index[idx]
        with np.load(entry.path, allow_pickle=True) as episode:
            steps = episode["steps"].tolist()
            step = dict(steps[entry.step_index])
        return self._apply_transforms(step)

    def _build_index(self) -> list[_IndexEntry]:
        entries: list[_IndexEntry] = []
        for root in self._roots:
            for path in sorted(root.glob("*.npz")):
                with np.load(path, allow_pickle=True) as episode:
                    meta = EpisodeMeta(**episode["meta"].item())
                    if self._task_filter and meta.task not in self._task_filter:
                        continue
                    steps = episode["steps"].tolist()
                    for step_idx in range(len(steps)):
                        entries.append(_IndexEntry(path=path, step_index=step_idx, task=meta.task))
        return entries

    def _apply_transforms(self, step: dict[str, Any]) -> dict[str, Any]:
        result = step
        for transform in self._transforms:
            result = transform(result)
        return result


def make_dataloader(cfg: Mapping[str, Any]) -> Any:
    """Factory for creating a torch DataLoader from configuration."""

    try:  # pragma: no cover - torch import is optional during scaffolding
        import torch
    except ImportError as exc:  # noqa: F841
        raise ImportError("PyTorch is required to build the training dataloader.") from exc

    dataset = MultiTaskDataset(
        roots=cfg.get("roots", []),
        transforms=cfg.get("transforms"),
        tasks=cfg.get("tasks"),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=cfg.get("shuffle", False),
        num_workers=cfg.get("num_workers", 0),
    )


def _normalize_transforms(transforms: Sequence[Transform] | Transform | None) -> list[Transform]:
    if transforms is None:
        return []
    if callable(transforms):
        return [transforms]
    return list(transforms)
