"""Integration smoke tests for environment → control → data stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from control.phase_machine import PhaseMachine, phase_signals_from_state
from control.scripted import scripted_lift_policy
from data.dataloader import MultiTaskDataset
from data.writer import EpisodeWriter
from envs.core_env import NewtonMAEnv


def test_end_to_end_collection_and_loading(tmp_path: Path) -> None:
    env = NewtonMAEnv({"task_name": "lift", "max_steps": 5})
    writer = EpisodeWriter(tmp_path)
    observations = env.reset("Collect lift demo")
    phase_machine = PhaseMachine(tuple(env._phases))  # type: ignore[attr-defined]
    writer.start_episode({"task": "lift"})
    info: dict[str, object] = {}
    for _ in range(3):
        actions = scripted_lift_policy(env, phase_machine, observations)
        observations, rewards, done, info = env.step(actions)
        payload = {
            "rgb_a": observations[0].get("rgb", np.zeros((48, 48, 3), dtype=np.uint8)),
            "rgb_b": observations[1].get("rgb", np.zeros((48, 48, 3), dtype=np.uint8)),
            "q_a": np.zeros(4, dtype=np.float32),
            "q_b": np.zeros(4, dtype=np.float32),
            "action_a": np.asarray(actions[0], dtype=np.float32),
            "action_b": np.asarray(actions[1], dtype=np.float32),
            "grip_a": float(actions[0][-1]),
            "grip_b": float(actions[1][-1]),
            "instruction": "Collect lift demo",
            "task": "lift",
            "success": bool(info.get("task_success", False)),
        }
        writer.add_step(payload)
        phase_machine.step(phase_signals_from_state({}, info))
        if done:
            break
    writer.end_episode(success=bool(info.get("task_success", False)))
    env.close()

    dataset = MultiTaskDataset([tmp_path])
    sample = dataset[0]
    assert sample["task"] == "lift"
