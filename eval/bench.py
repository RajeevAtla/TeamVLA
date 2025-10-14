"""Benchmark CLI for evaluating TeamVLA policies."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from eval.metrics import aggregate_results, success_at_T
from eval.rollouts import run_suite
from envs import NewtonMAEnv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TeamVLA evaluation benchmark.")
    parser.add_argument("--config", default="configs/train_bc.yaml", help="Path to config file.")
    parser.add_argument("--tasks", nargs="*", default=["lift", "handoff", "drawer"], help="Tasks to evaluate.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    env = NewtonMAEnv({"task_name": args.tasks[0], "max_steps": args.max_steps})

    def policy(observations: list[dict[str, Any]]) -> list[list[float]]:
        return [[0.0] * 7, [0.0] * 7]

    results = run_suite(env, policy, tasks=args.tasks, n_eps=args.episodes, unseen=False)
    summary = aggregate_results(results)
    summary["success_at_T"] = success_at_T(results, horizon=args.max_steps)
    logging.info("Benchmark summary: %s", summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(str(summary), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
