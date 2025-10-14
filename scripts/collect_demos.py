"""Script for generating scripted demonstration episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from control.scripted import scripted_drawer_policy, scripted_handoff_policy, scripted_lift_policy
from control.phase_machine import PhaseMachine
from data.writer import EpisodeWriter
from envs import NewtonMAEnv


POLICIES = {
    "lift": scripted_lift_policy,
    "handoff": scripted_handoff_policy,
    "drawer": scripted_drawer_policy,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted TeamVLA demonstrations.")
    parser.add_argument("--out", type=Path, default=Path("data/episodes"))
    parser.add_argument("--task", choices=list(POLICIES), default="lift")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser.parse_args(argv)


def collect(args: argparse.Namespace) -> None:
    env = NewtonMAEnv({"task_name": args.task, "max_steps": args.max_steps})
    phase_machine = PhaseMachine(phases=("reach", "action"))
    policy = POLICIES[args.task]
    writer = EpisodeWriter(args.out / args.task)
    for episode_idx in range(args.episodes):
        observations = env.reset(f"Perform {args.task}")
        writer.start_episode({"task": args.task, "episode_id": f"{episode_idx:04d}", "success": False})
        for step in range(args.max_steps):
            actions = policy(env, phase_machine, observations)
            observations, _rewards, done, info = env.step(actions)
            zeros = np.zeros(7, dtype=np.float32)
            writer.add_step(
                {
                    "rgb_a": observations[0].get("rgb", np.zeros((3, 64, 64), dtype=np.uint8)),
                    "rgb_b": observations[1].get("rgb", np.zeros((3, 64, 64), dtype=np.uint8)),
                    "q_a": zeros,
                    "q_b": zeros,
                    "action_a": np.asarray(actions[0], dtype=np.float32),
                    "action_b": np.asarray(actions[1], dtype=np.float32),
                    "grip_a": 0.0,
                    "grip_b": 0.0,
                    "instruction": f"Perform {args.task}",
                    "task": args.task,
                    "success": bool(info.get("task_success", False)),
                }
            )
            if done:
                break
        writer.end_episode(success=bool(info.get("task_success", False)))
    writer.close()
    env.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collect(args)


if __name__ == "__main__":  # pragma: no cover
    main()
