"""Gradio demo stub for TeamVLA policies."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from envs import NewtonMAEnv
from models.vla_singlebrain import SingleBrainVLA


def load_policy(checkpoint_path: str | None = None) -> Callable[[Mapping[str, Any]], list[np.ndarray]]:
    """Load a trained policy; returns a no-op placeholder policy."""

    _unused(checkpoint_path)
    SingleBrainVLA({"vision_dim": 16, "text_dim": 16, "fusion_dim": 32, "action_dim": 8})

    def policy(obs: Mapping[str, Any]) -> list[np.ndarray]:
        _unused(obs)
        return [np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]

    return policy


def inference_step(
    img_a: np.ndarray,
    img_b: np.ndarray,
    text: str,
    state: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Placeholder inference step returning zero-actions and updated state."""

    _unused(img_a, img_b, text)
    state = dict(state or {})
    state.setdefault("calls", 0)
    state["calls"] += 1
    return [np.zeros(8), np.zeros(8)], state


def main() -> None:
    """Launch the Gradio interface."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Gradio is required to launch the demo application.") from exc

    env = NewtonMAEnv({"task_name": "lift"})
    iface = gr.Interface(
        fn=lambda text: inference_step(np.zeros((3, 64, 64)), np.zeros((3, 64, 64)), text, {}),
        inputs=gr.Textbox(label="Instruction"),
        outputs=gr.JSON(label="Actions"),
        title="TeamVLA Demo",
        description="Placeholder demo returning zero actions.",
    )
    env.close()
    iface.launch()


def _unused(*_: Any) -> None:
    """Helper for unused arguments."""

