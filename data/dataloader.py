"""Dataset loader utilities for TeamVLA multi-task training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from data.schema import EpisodeMeta, validate_episode_meta

Transform = Callable[[dict[str, Any]], dict[str, Any]]

if TYPE_CHECKING:  # pragma: no cover - only for static analysis
    from torch.utils.data import Dataset as _TorchDataset

    _DatasetBase = _TorchDataset[dict[str, Any]]
else:  # pragma: no cover - torch is optional at runtime
    _DatasetBase = object


@dataclass(slots=True)
class _EpisodeRecord:
    path: Path
    meta: EpisodeMeta
    step_count: int


@dataclass(slots=True)
class _StepRef:
    episode: _EpisodeRecord
    index: int


class MultiTaskDataset(_DatasetBase):
    """PyTorch-style dataset that iterates over multi-task episodes."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        transforms: Sequence[Transform] | Transform | None = None,
        tasks: Sequence[str] | None = None,
        limit_per_task: int | None = None,
    ) -> None:
        self._roots = [Path(root) for root in roots]
        self._transforms = _normalize_transforms(transforms)
        self._task_filter = set(tasks) if tasks else None
        self._limit_per_task = limit_per_task
        self._episodes = self._build_manifest()
        self._index = self._build_step_index(self._episodes)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        step_ref = self._index[idx]
        with np.load(step_ref.episode.path, allow_pickle=True) as episode:
            steps = episode["steps"].tolist()
            step = dict(steps[step_ref.index])
        return self._apply_transforms(step)

    # ------------------------------------------------------------------#
    # Internal helpers                                                   #
    # ------------------------------------------------------------------#

    def _build_manifest(self) -> list[_EpisodeRecord]:
        records: list[_EpisodeRecord] = []
        for root in self._roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.npz")):
                with np.load(path, allow_pickle=True) as episode:
                    raw_meta = episode["meta"].item()
                    steps = episode["steps"].tolist()
                meta = validate_episode_meta(raw_meta, num_steps=len(steps))
                if self._task_filter and meta.task not in self._task_filter:
                    continue
                records.append(_EpisodeRecord(path=path, meta=meta, step_count=len(steps)))

        if self._limit_per_task is not None and self._limit_per_task >= 0:
            records = self._apply_task_limit(records, self._limit_per_task)
        return records

    def _build_step_index(self, episodes: Iterable[_EpisodeRecord]) -> list[_StepRef]:
        index: list[_StepRef] = []
        for record in episodes:
            for step_idx in range(record.step_count):
                index.append(_StepRef(episode=record, index=step_idx))
        return index

    def _apply_task_limit(self, records: list[_EpisodeRecord], limit: int) -> list[_EpisodeRecord]:
        per_task: dict[str, list[_EpisodeRecord]] = {}
        for record in records:
            per_task.setdefault(record.meta.task, []).append(record)
        limited: list[_EpisodeRecord] = []
        for _task, episodes in per_task.items():
            limited.extend(episodes[:limit]) if limit else limited.extend(episodes)
        return sorted(limited, key=lambda rec: (rec.meta.task, rec.meta.episode_id))

    def _apply_transforms(self, step: dict[str, Any]) -> dict[str, Any]:
        for transform in self._transforms:
            step = transform(step)
        return step


def make_dataloader(cfg: Mapping[str, Any]) -> Any:
    """Factory for creating a torch DataLoader from configuration."""

    try:  # pragma: no cover - torch is optional during scaffolding
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required to build the training dataloader.") from exc

    dataset = MultiTaskDataset(
        roots=cfg.get("roots", []),
        transforms=cfg.get("transforms"),
        tasks=cfg.get("tasks"),
        limit_per_task=cfg.get("limit_per_task"),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=cfg.get("shuffle", False),
        num_workers=cfg.get("num_workers", 0),
        drop_last=cfg.get("drop_last", False),
    )


def _normalize_transforms(transforms: Sequence[Transform] | Transform | None) -> list[Transform]:
    if transforms is None:
        return []
    if callable(transforms):
        return [transforms]
    return list(transforms)
