"""Gradio demo for TeamVLA scripted and learned policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import numpy as np

from envs import NewtonMAEnv

try:  # pragma: no cover - optional torch dependency
    from models.vla_singlebrain import SingleBrainVLA
except ImportError:  # pragma: no cover
    SingleBrainVLA = None  # type: ignore


@dataclass(slots=True)
class DemoConfig:
    max_steps: int = 32
    task: str = "lift"


def load_policy(checkpoint_path: str | None = None) -> Callable[[Sequence[dict[str, Any]]], List[np.ndarray]]:
    """Load a trained policy; defaults to a zero-action stub if Torch unavailable."""

    del checkpoint_path  # Placeholder for future checkpoint support.
    if SingleBrainVLA is None:
        return _zero_policy
    try:  # pragma: no cover - executed when torch is available
        model = SingleBrainVLA({"vision_dim": 16, "text_dim": 16, "fusion_dim": 32, "action_dim": 4})
    except Exception:  # noqa: BLE001 - fall back gracefully
        return _zero_policy

    def _policy(observations: Sequence[dict[str, Any]]) -> List[np.ndarray]:
        return [np.asarray(action, dtype=np.float32) for action in model.act(observations)]

    return _policy


def run_demo_episode(instruction: str, *, cfg: DemoConfig | None = None) -> dict[str, Any]:
    """Run a short demo episode using the placeholder policy."""

    cfg = cfg or DemoConfig()
    env = NewtonMAEnv({"task_name": cfg.task, "max_steps": cfg.max_steps})
    policy = load_policy(None)
    actions_log: list[list[float]] = []
    obs = env.reset(instruction)
    info: dict[str, Any] = {}
    try:
        for _ in range(cfg.max_steps):
            acts = policy(obs)
            actions_log.append([float(x) for x in acts[0]])
            obs, _rewards, done, info = env.step(acts)
            if done:
                break
    finally:
        env.close()
    return {
        "instruction": instruction,
        "steps": len(actions_log),
        "actions": actions_log,
        "success": bool(info.get("task_success", False)),
    }


def main() -> None:
    """Launch the Gradio interface."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Gradio is required to launch the demo application.") from exc

    iface = gr.Interface(
        fn=lambda instruction: run_demo_episode(instruction, cfg=DemoConfig()),
        inputs=gr.Textbox(label="Instruction"),
        outputs=gr.JSON(label="Episode Summary"),
        title="TeamVLA Demo",
        description="Runs a short scripted rollout and reports placeholder actions.",
    )
    iface.launch()


def _zero_policy(observations: Sequence[dict[str, Any]]) -> List[np.ndarray]:
    _unused(observations)
    return [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]


def _unused(*_: Any) -> None:
    pass

