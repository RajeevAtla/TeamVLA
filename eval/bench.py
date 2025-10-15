"""Benchmark CLI for evaluating TeamVLA policies."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

try:  # pragma: no cover - optional torch dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None

from eval.metrics import aggregate_results
from eval.rollouts import run_suite
from envs import NewtonMAEnv

try:  # pragma: no cover - optional torch dependency
    from models.vla_singlebrain import SingleBrainVLA
except ImportError:  # pragma: no cover
    SingleBrainVLA = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TeamVLA evaluation benchmark.")
    parser.add_argument("--tasks", nargs="*", default=["lift", "handoff", "drawer"], help="Tasks to evaluate.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--record-infos", action="store_true", help="Store per-step info in results.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load.")
    return parser.parse_args(argv)


def load_policy(checkpoint: Path | None = None) -> Callable[[list[dict[str, Any]]], list[list[float]]]:
    """Load a trained policy if Torch/checkpoint are available; otherwise zero actions."""

    if checkpoint is None or SingleBrainVLA is None or torch is None:
        return _zero_policy

    if not checkpoint.exists():
        LOGGER.warning("Checkpoint %s not found. Using zero-action policy.", checkpoint)
        return _zero_policy

    payload = torch.load(checkpoint, map_location="cpu")
    model_cfg = payload.get("config", {}).get("model", {})
    model = SingleBrainVLA(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()

    def policy(observations: list[dict[str, Any]]) -> list[list[float]]:
        actions = model.act(observations)
        return [action.tolist() for action in actions]

    return policy


def build_env(task: str, max_steps: int) -> NewtonMAEnv:
    return NewtonMAEnv({"task_name": task, "max_steps": max_steps})


def benchmark(
    tasks: Iterable[str],
    policy: Callable[[list[dict[str, Any]]], list[Any]],
    *,
    max_steps: int,
    episodes: int,
    seed: int | None,
    record_infos: bool,
) -> list[dict[str, Any]]:
    tasks = list(tasks)
    if not tasks:
        raise ValueError("At least one task must be specified.")
    env = build_env(tasks[0], max_steps)
    try:
        results = run_suite(
            env,
            policy,
            tasks=tasks,
            n_eps=episodes,
            max_steps=max_steps,
            seed=seed,
            record_infos=record_infos,
        )
    finally:
        env.close()
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    policy = load_policy(args.checkpoint)
    results = benchmark(
        args.tasks,
        policy,
        max_steps=args.max_steps,
        episodes=args.episodes,
        seed=args.seed,
        record_infos=args.record_infos,
    )
    summary = aggregate_results(results, horizon=args.max_steps)
    logging.info("Benchmark summary: %s", summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")


def _zero_policy(observations: list[dict[str, Any]]) -> list[list[float]]:
    _unused(observations)
    return [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def _unused(*_: Any) -> None:
    pass


if __name__ == "__main__":  # pragma: no cover
    main()

