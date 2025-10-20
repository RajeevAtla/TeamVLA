"""Gradio demo for TeamVLA scripted and learned policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from envs import NewtonMAEnv

try:  # pragma: no cover - optional torch dependency
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = cast(Any, None)

if TYPE_CHECKING:  # pragma: no cover - only for static analysis
    from models.vla_singlebrain import SingleBrainVLA
else:  # pragma: no cover - runtime shims when optional deps are missing
    torch = cast(Any, _torch)
    try:
        from models.vla_singlebrain import SingleBrainVLA
    except ImportError:  # pragma: no cover
        SingleBrainVLA = cast(Any, None)


@dataclass(slots=True)
class DemoConfig:
    max_steps: int = 32
    task: str = "lift"
    checkpoint: Path | None = None


def load_policy(
    checkpoint_path: str | Path | None = None,
) -> Callable[[Sequence[dict[str, Any]]], list[np.ndarray]]:
    """Load a trained policy; defaults to a zero-action shim if unavailable."""

    if checkpoint_path is None or SingleBrainVLA is None or torch is None:
        return _zero_policy

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return _zero_policy

    payload = torch.load(checkpoint, map_location="cpu")
    model_cfg = payload.get("config", {}).get("model", {})
    model = SingleBrainVLA(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()

    def policy(observations: Sequence[dict[str, Any]]) -> list[np.ndarray]:
        actions = model.act(observations)
        return [np.asarray(action, dtype=np.float32) for action in actions]

    return policy


def run_demo_episode(instruction: str, *, cfg: DemoConfig | None = None) -> dict[str, Any]:
    """Run a short demo episode using the configured policy."""

    cfg = cfg or DemoConfig()
    env = NewtonMAEnv({"task_name": cfg.task, "max_steps": cfg.max_steps})
    policy = load_policy(cfg.checkpoint)
    actions_log: list[list[float]] = []
    obs = env.reset(instruction)
    info: dict[str, Any] = {}
    try:
        for _ in range(cfg.max_steps):
            acts = policy(obs)
            step_actions = [[float(x) for x in action] for action in acts]
            actions_log.append(step_actions[0])
            obs, _rewards, done, info = env.step(step_actions)
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


def _zero_policy(observations: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    del observations
    return [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
