"""Episode writer utilities for persisting TeamVLA datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from data.schema import EpisodeMeta, validate_episode_meta, validate_step


class EpisodeWriter:
    """Utility for streaming TeamVLA episodes to disk."""

    def __init__(self, out_dir: str | Path, fmt: str = "npz", compress: bool = True) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._fmt = fmt
        self._compress = compress
        self._current_meta: EpisodeMeta | None = None
        self._steps: list[dict[str, Any]] = []

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start_episode(self, meta: Mapping[str, Any]) -> None:
        """Begin a new episode with the supplied metadata."""

        if self._current_meta is not None:
            raise RuntimeError("An episode is already in progress.")
        self._current_meta = validate_episode_meta(meta)
        self._steps.clear()

    def add_step(self, step: Mapping[str, Any]) -> None:
        """Append a validated step to the current episode."""

        if self._current_meta is None:
            raise RuntimeError("start_episode must be called before add_step.")
        validate_step(step)
        self._steps.append(dict(step))

    def end_episode(self, success: bool | None = None) -> str:
        """Finalize the current episode and write it to disk."""

        if self._current_meta is None:
            raise RuntimeError("No episode in progress to end.")
        meta = self._override_success(self._current_meta, success)
        path = self._write_episode(meta, self._steps)
        self._current_meta = None
        self._steps = []
        return str(path)

    def close(self) -> None:
        """Abort any in-progress episode without writing."""

        self._current_meta = None
        self._steps = []

    def _override_success(self, meta: EpisodeMeta, success: bool | None) -> EpisodeMeta:
        if success is None:
            return meta
        return EpisodeMeta(task=meta.task, episode_id=meta.episode_id, success=bool(success))

    def _write_episode(self, meta: EpisodeMeta, steps: Iterable[Mapping[str, Any]]) -> Path:
        if self._fmt != "npz":
            raise ValueError(f"Unsupported episode format '{self._fmt}'.")
        payload = {
            "meta": asdict(meta),
            "steps": np.array(list(steps), dtype=object),
        }
        path = self._episode_path(meta)
        if self._compress:
            np.savez_compressed(path, **payload)
        else:
            np.savez(path, **payload)
        return path

    def _episode_path(self, meta: EpisodeMeta) -> Path:
        filename = f"{meta.task}_{meta.episode_id}.{self._fmt}"
        return self._out_dir / filename
