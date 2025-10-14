"""Script to convert recorded episodes into MP4 videos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render TeamVLA episodes to video.")
    parser.add_argument("--episodes", type=Path, default=Path("data/episodes"))
    parser.add_argument("--out", type=Path, default=Path("videos"))
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args(argv)


def render_episode(ep_path: Path, out_path: Path, fps: int = 20) -> None:
    """Placeholder rendering that saves numpy arrays instead of actual video."""

    with np.load(ep_path, allow_pickle=True) as data:
        frames = data["steps"].tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path.with_suffix(".npy"), frames)


def render_all(episodes_dir: Path, out_dir: Path, fps: int) -> None:
    for episode_path in episodes_dir.rglob("*.npz"):
        target = out_dir / episode_path.stem
        render_episode(episode_path, target, fps=fps)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    render_all(args.episodes, args.out, args.fps)


if __name__ == "__main__":  # pragma: no cover
    main()
