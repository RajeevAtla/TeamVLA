"""Benchmark CLI for evaluating TeamVLA policies."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

from eval.metrics import aggregate_results
from eval.rollouts import run_suite
from envs import NewtonMAEnv

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TeamVLA evaluation benchmark.")
    parser.add_argument("--tasks", nargs="*", default=["lift", "handoff", "drawer"], help="Tasks to evaluate.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--record-infos", action="store_true", help="Store per-step info in results.")
    return parser.parse_args(argv)


def load_policy(checkpoint: str | None = None) -> Callable[[list[dict[str, Any]]], list[list[float]]]:
    """Placeholder policy loader that outputs zero actions."""

    del checkpoint  # Placeholder until trained checkpoints are produced.

    def policy(observations: list[dict[str, Any]]) -> list[list[float]]:
        _unused(observations)
        return [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

    return policy


def build_env(task: str, max_steps: int) -> NewtonMAEnv:
    return NewtonMAEnv({"task_name": task, "max_steps": max_steps})


def benchmark(tasks: Iterable[str], policy: Callable[[list[dict[str, Any]]], list[Any]], *, max_steps: int, episodes: int, seed: int | None, record_infos: bool) -> list[dict[str, Any]]:
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
    policy = load_policy()
    results = benchmark(args.tasks, policy, max_steps=args.max_steps, episodes=args.episodes, seed=args.seed, record_infos=args.record_infos)
    summary = aggregate_results(results, horizon=args.max_steps)
    logging.info("Benchmark summary: %s", summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")


def _unused(*_: Any) -> None:
    pass


if __name__ == "__main__":  # pragma: no cover
    main()
