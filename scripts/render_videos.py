"""Script to convert recorded episodes into MP4 videos."""

import argparse
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

_imageio_module: Any | None
try:  # pragma: no cover - optional dependency
    _imageio_module = importlib.import_module("imageio.v2")
except ImportError:  # pragma: no cover
    _imageio_module = None

imageio = cast(Any, _imageio_module)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render TeamVLA episodes to video.")
    parser.add_argument("--episodes", type=Path, default=Path("data/episodes"))
    parser.add_argument("--out", type=Path, default=Path("videos"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--agent", choices=["a", "b", "both"], default="both")
    return parser.parse_args(argv)


def render_episode(ep_path: Path, out_path: Path, *, fps: int = 20, agent: str = "both") -> None:
    """Render a single episode to an MP4 (or numpy fallback)."""

    with np.load(ep_path, allow_pickle=True) as data:
        steps = data["steps"].tolist()
    frames = [_compose_frame(step, agent=agent) for step in steps]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(out_path.with_suffix(".mp4"), frames, fps=fps)
    else:
        np.save(out_path.with_suffix(".npy"), frames)


def render_all(episodes_dir: Path, out_dir: Path, *, fps: int, agent: str) -> None:
    for episode_path in episodes_dir.rglob("*.npz"):
        target = out_dir / episode_path.stem
        render_episode(episode_path, target, fps=fps, agent=agent)


def _compose_frame(step: Mapping[str, object], *, agent: str) -> np.ndarray:
    rgb_a = step.get("rgb_a")
    rgb_b = step.get("rgb_b")
    if rgb_a is None and rgb_b is None:
        return np.zeros((48, 48, 3), dtype=np.uint8)
    if agent == "a":
        return _ensure_uint8(rgb_a)
    if agent == "b":
        return _ensure_uint8(rgb_b)
    return np.hstack([_ensure_uint8(rgb_a), _ensure_uint8(rgb_b)])


def _ensure_uint8(frame: object) -> np.ndarray:
    if frame is None:
        return np.zeros((48, 48, 3), dtype=np.uint8)
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)
    return array


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    render_all(args.episodes, args.out, fps=args.fps, agent=args.agent)


if __name__ == "__main__":  # pragma: no cover
    main()
