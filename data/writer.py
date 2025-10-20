"""Episode writer utilities for persisting TeamVLA datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import EPISODE_FILE_VERSION, EpisodeMeta, validate_episode_meta, validate_step


class EpisodeWriter:
    """Utility for streaming TeamVLA episodes to disk."""

    def __init__(
        self,
        out_dir: str | Path,
        fmt: str = "npz",
        *,
        compress: bool = True,
        episode_prefix: str | None = None,
        auto_episode_id: bool = True,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._fmt = fmt
        self._compress = compress
        self._episode_prefix = episode_prefix or "episode"
        self._auto_episode_id = auto_episode_id
        self._current_meta: EpisodeMeta | None = None
        self._steps: list[dict[str, Any]] = []
        self._episode_counter = 0

    # ------------------------------------------------------------------#
    # Context manager helpers                                           #
    # ------------------------------------------------------------------#

    def __enter__(self) -> EpisodeWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------#
    # Public API                                                        #
    # ------------------------------------------------------------------#

    def start_episode(self, meta: Mapping[str, Any]) -> EpisodeMeta:
        """Begin a new episode with the supplied metadata."""

        if self._current_meta is not None:
            raise RuntimeError("An episode is already in progress.")

        meta_dict = dict(meta)
        if self._auto_episode_id and not meta_dict.get("episode_id"):
            meta_dict["episode_id"] = self._generate_episode_id()
        episode_meta = validate_episode_meta(meta_dict)
        self._current_meta = episode_meta
        self._steps.clear()
        return episode_meta

    def add_step(self, step: Mapping[str, Any]) -> None:
        """Append a validated step to the current episode."""

        if self._current_meta is None:
            raise RuntimeError("start_episode must be called before add_step.")
        validate_step(step)
        self._steps.append(dict(step))

    def end_episode(self, success: bool | None = None) -> Path:
        """Finalize the current episode and write it to disk."""

        if self._current_meta is None:
            raise RuntimeError("No episode in progress to end.")
        meta = self._finalize_meta(self._current_meta, success)
        path = self._write_episode(meta, self._steps)
        self._current_meta = None
        self._steps = []
        return path

    def close(self) -> None:
        """Abort any in-progress episode without writing."""

        self._current_meta = None
        self._steps = []

    @property
    def active(self) -> bool:
        """Return True if an episode is currently being recorded."""

        return self._current_meta is not None

    # ------------------------------------------------------------------#
    # Internal helpers                                                   #
    # ------------------------------------------------------------------#

    def _generate_episode_id(self) -> str:
        self._episode_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{self._episode_prefix}_{timestamp}_{self._episode_counter:04d}"

    def _finalize_meta(self, meta: EpisodeMeta, success: bool | None) -> EpisodeMeta:
        payload = {
            "task": meta.task,
            "episode_id": meta.episode_id,
            "success": bool(success if success is not None else meta.success),
            "num_steps": len(self._steps),
            "version": EPISODE_FILE_VERSION,
        }
        return validate_episode_meta(payload, num_steps=len(self._steps))

    def _write_episode(self, meta: EpisodeMeta, steps: Iterable[Mapping[str, Any]]) -> Path:
        if self._fmt != "npz":
            raise ValueError(f"Unsupported episode format '{self._fmt}'.")
        steps_list = list(steps)
        meta_payload = asdict(meta)
        steps_array = np.array(steps_list, dtype=object)
        path = self._episode_path(meta)
        if self._compress:
            np.savez_compressed(path, meta=cast(Any, meta_payload), steps=steps_array)
        else:
            np.savez(path, meta=cast(Any, meta_payload), steps=steps_array)
        return path

    def _episode_path(self, meta: EpisodeMeta) -> Path:
        filename = f"{meta.task}_{meta.episode_id}.{self._fmt}"
        return self._out_dir / filename
