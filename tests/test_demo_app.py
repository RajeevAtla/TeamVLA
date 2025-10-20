"""Tests for the demo application scaffolding."""

from __future__ import annotations

import sys
import types

import numpy as np

from demos import app


def test_load_policy_returns_callable() -> None:
    policy = app.load_policy(None)
    actions = policy(
        [
            {"rgb": np.zeros((48, 48, 3), dtype=np.uint8)},
            {"rgb": np.zeros((48, 48, 3), dtype=np.uint8)},
        ]
    )
    assert len(actions) == 2


def test_run_demo_episode_returns_summary() -> None:
    summary = app.run_demo_episode("test instruction", cfg=app.DemoConfig(max_steps=2))
    assert "actions" in summary
    assert summary["steps"] <= 2


def test_main_launches_interface(monkeypatch) -> None:
    fake_gradio = types.SimpleNamespace()
    launch_called = {"flag": False}

    class FakeInterface:
        def __init__(self, *args, **kwargs):  # noqa: D401,ANN001
            pass

        def launch(self):
            launch_called["flag"] = True

    fake_gradio.Interface = FakeInterface
    fake_gradio.Textbox = lambda **kwargs: None
    fake_gradio.JSON = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    app.main()
    assert launch_called["flag"]
