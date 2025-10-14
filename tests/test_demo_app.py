"""Tests for the demo application scaffolding."""

from __future__ import annotations

import sys
import types

import numpy as np

from demos import app


def test_load_policy_returns_callable() -> None:
    policy = app.load_policy(None)
    actions = policy({"observation": 0})
    assert len(actions) == 2


def test_inference_step_updates_state() -> None:
    actions, state = app.inference_step(np.zeros((3, 64, 64)), np.zeros((3, 64, 64)), "text", {})
    assert state["calls"] == 1
    assert len(actions) == 2


def test_main_launches_interface(monkeypatch) -> None:
    fake_gradio = types.SimpleNamespace()
    launch_called = {"flag": False}

    class FakeInterface:
        def __init__(self, *args, **kwargs):
            self.launched = False

        def launch(self):
            launch_called["flag"] = True

    fake_gradio.Interface = FakeInterface
    fake_gradio.Textbox = lambda **kwargs: None
    fake_gradio.JSON = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    app.main()
    assert launch_called["flag"]
