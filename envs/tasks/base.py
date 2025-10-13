"""Task specification protocols and registry utilities for TeamVLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Sequence, runtime_checkable

TaskFactory = Callable[[], "TaskSpec"]


@dataclass(slots=True)
class TaskMetadata:
    """Lightweight metadata returned during task resets.

    Attributes:
        phases: Ordered list of high-level task phases.
        extras: Arbitrary task-specific metadata.
    """

    phases: Sequence[str]
    extras: Mapping[str, Any]


@runtime_checkable
class TaskSpec(Protocol):
    """Protocol describing the contract each TeamVLA task must satisfy."""

    name: str

    def build_scene(self, model: Any) -> None:
        """Populate the Newton model with task-specific geometry."""

    def phases(self) -> Sequence[str]:
        """Return the ordered list of task phases used for shaping and logging."""

    def reset(self, state: Any, rng: Any) -> TaskMetadata:
        """Randomize task state and return metadata about the new episode."""

    def reward(self, state: Any, phase: str) -> Sequence[float]:
        """Compute per-agent rewards for the provided state and phase."""

    def success(self, state: Any) -> bool:
        """Return True when the task has satisfied success conditions."""

    def scripted_action(self, obs: Mapping[str, Any], phase: str, agent_id: int) -> Sequence[float]:
        """Return a scripted action for data collection."""

    def info(self, state: Any) -> Mapping[str, Any]:
        """Return diagnostic information for logging/telemetry."""


_TASK_REGISTRY: Dict[str, TaskFactory] = {}


def register_task(name: str, factory: TaskFactory) -> None:
    """Register a new task factory under the provided name."""

    _ensure_not_registered(name)
    _TASK_REGISTRY[name] = factory


def register_task_class(task_cls: type[TaskSpec]) -> type[TaskSpec]:
    """Decorate a TaskSpec-compatible class to register it by its ``name`` attribute."""

    register_task(task_cls.name, task_cls)
    return task_cls


def deregister_task(name: str) -> None:
    """Remove a task from the registry (primarily for tests)."""

    _TASK_REGISTRY.pop(name, None)


def get_task(name: str) -> TaskSpec:
    """Instantiate a registered task by name."""

    factory = _TASK_REGISTRY.get(name)
    if factory is None:
        message = f"Task '{name}' is not registered. Available tasks: {sorted(_TASK_REGISTRY)}"
        raise KeyError(message)
    return factory()


def iter_registered_tasks() -> Iterable[str]:
    """Yield registered task names."""

    return _TASK_REGISTRY.keys()


def _ensure_not_registered(name: str) -> None:
    """Ensure the provided name is not already present in the registry."""

    if name in _TASK_REGISTRY:
        raise ValueError(f"Task '{name}' is already registered.")
