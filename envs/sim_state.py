"""Shared simulation state dataclasses for the TeamVLA tabletop environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class SimulationModel:
    """Static configuration describing the workspace bounds and integration step."""

    workspace_min: NDArray[np.float64]
    workspace_max: NDArray[np.float64]
    action_limit: float
    dt: float


@dataclass(slots=True)
class ArmState:
    """State for a single robot arm end-effector."""

    name: str
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    gripper: float = 1.0


@dataclass(slots=True)
class SimulationObject:
    """Rigid object tracked within the tabletop workspace."""

    name: str
    position: NDArray[np.float64]
    target: NDArray[np.float64]
    radius: float
    holders: set[str] = field(default_factory=set)

    def attach(self, agent: str) -> None:
        self.holders.add(agent)

    def detach(self, agent: str) -> None:
        self.holders.discard(agent)


@dataclass(slots=True)
class SimulationState:
    """Mutable simulation snapshot advanced each environment step."""

    arms: dict[str, ArmState]
    objects: dict[str, SimulationObject]
    rng: np.random.Generator
    step_index: int = 0
    task_state: dict[str, Any] = field(default_factory=dict)

    def reset_step(self) -> None:
        self.step_index = 0


__all__ = [
    "SimulationModel",
    "ArmState",
    "SimulationObject",
    "SimulationState",
]
