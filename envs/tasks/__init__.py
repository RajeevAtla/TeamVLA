"""Task package exposing registry utilities and default tasks."""

from .base import (
    TaskMetadata,
    TaskSpec,
    deregister_task,
    get_task,
    iter_registered_tasks,
    register_task,
    register_task_class,
)
from .drawer import DrawerTask
from .handoff import HandOffTask
from .lift import LiftTask

__all__ = [
    "TaskMetadata",
    "TaskSpec",
    "deregister_task",
    "get_task",
    "iter_registered_tasks",
    "register_task",
    "register_task_class",
    "DrawerTask",
    "HandOffTask",
    "LiftTask",
]
