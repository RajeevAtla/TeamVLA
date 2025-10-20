"""Script for generating scripted demonstration episodes."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from control.phase_machine import PhaseMachine, phase_signals_from_state
from control.scripted import (
    scripted_drawer_policy,
    scripted_handoff_policy,
    scripted_lift_policy,
)
from data.writer import EpisodeWriter
from envs import NewtonMAEnv

SCRIPTED_POLICIES: dict[str, Callable] = {
    "lift": scripted_lift_policy,
    "handoff": scripted_handoff_policy,
    "drawer": scripted_drawer_policy,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted TeamVLA demonstrations.")
    parser.add_argument("--out", type=Path, default=Path("data/episodes"))
    parser.add_argument("--task", choices=list(SCRIPTED_POLICIES), default="lift", help="Single task to collect.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional list of tasks to collect.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility.")
    parser.add_argument("--record-infos", action="store_true", help="Store environment info in steps.")
    return parser.parse_args(argv)


def collect(args: argparse.Namespace) -> None:
    tasks = args.tasks or [args.task]
    for task in tasks:
        _collect_task(task, args)


def _collect_task(task: str, args: argparse.Namespace) -> None:
    env = NewtonMAEnv({"task_name": task, "max_steps": args.max_steps, "seed": args.seed or 0})
    policy = SCRIPTED_POLICIES[task]
    writer = EpisodeWriter(args.out / task)
    try:
        for _episode_idx in range(args.episodes):
            observations = env.reset(f"Perform {task}")
            phase_machine = PhaseMachine(tuple(env._phases))  # type: ignore[attr-defined]
            writer.start_episode({"task": task})
            info: dict[str, object] = {}
            for _step in range(args.max_steps):
                actions = policy(env, phase_machine, observations)
                observations, rewards, done, info = env.step(actions)
                step_payload = _step_payload(observations, actions, rewards, task, info)
                if args.record_infos:
                    step_payload["info"] = dict(info)
                writer.add_step(step_payload)
                signals = phase_signals_from_state({}, info)
                phase_machine.step(signals)
                if done:
                    break
            writer.end_episode(success=bool(info.get("task_success", False)))
    finally:
        writer.close()
        env.close()


def _step_payload(observations, actions, rewards, task: str, info: dict) -> dict[str, object]:
    rgb_a = observations[0].get("rgb")
    rgb_b = observations[1].get("rgb")
    if rgb_a is None:
        rgb_a = np.zeros((48, 48, 3), dtype=np.uint8)
    if rgb_b is None:
        rgb_b = np.zeros((48, 48, 3), dtype=np.uint8)
    action_a = np.asarray(actions[0], dtype=np.float32)
    action_b = np.asarray(actions[1], dtype=np.float32)
    reward_a = float(rewards[0]) if rewards else 0.0
    reward_b = float(rewards[1]) if rewards else 0.0
    return {
        "rgb_a": rgb_a,
        "rgb_b": rgb_b,
        "q_a": np.zeros(4, dtype=np.float32),
        "q_b": np.zeros(4, dtype=np.float32),
        "action_a": action_a,
        "action_b": action_b,
        "grip_a": float(action_a[-1]) if action_a.size else 0.0,
        "grip_b": float(action_b[-1]) if action_b.size else 0.0,
        "instruction": f"Perform {task}",
        "task": task,
        "success": bool(info.get("task_success", False)),
        "reward_a": reward_a,
        "reward_b": reward_b,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    collect(args)


if __name__ == "__main__":  # pragma: no cover
    main()
