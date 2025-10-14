"""Smoke tests ensuring scaffolded modules import successfully."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "envs",
        "envs.core_env",
        "envs.tasks",
        "envs.tasks.base",
        "envs.tasks.lift",
        "envs.tasks.handoff",
        "envs.tasks.drawer",
        "control",
        "control.ik_utils",
        "control.phase_machine",
        "control.scripted.lift_demo",
        "control.scripted.handoff_demo",
        "control.scripted.drawer_demo",
        "data.schema",
        "data.writer",
        "data.dataloader",
        "models.encoders.vision",
        "models.encoders.language",
        "models.vla_singlebrain",
        "models.vla_msgpassing",
        "train.bc_trainer",
        "train.losses",
        "train.schedulers",
        "eval.metrics",
        "eval.rollouts",
        "eval.bench",
        "demos.app",
        "scripts.collect_demos",
        "scripts.render_videos",
    ],
)
def test_module_imports(module_name: str) -> None:
    """Ensure each scaffolded module can be imported."""

    module = importlib.import_module(module_name)
    assert module.__doc__, f"Module {module_name} should define a docstring."
