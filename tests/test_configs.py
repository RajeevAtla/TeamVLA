"""Tests ensuring configuration files provide required keys."""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_common_config_contains_logging(tmp_path: Path) -> None:
    config = _load_yaml(Path("configs/common.yaml"))
    assert "log" in config
    assert "device" in config


def test_task_configs_define_task_field() -> None:
    for name in ["lift", "handoff", "drawer"]:
        config = _load_yaml(Path(f"configs/{name}.yaml"))
        assert config["task"] == name


def test_train_config_includes_dataset_and_model() -> None:
    config = _load_yaml(Path("configs/train_bc.yaml"))
    assert "dataset" in config
    assert "model" in config
