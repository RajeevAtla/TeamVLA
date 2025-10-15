"""Phase machine utilities for coordinating scripted controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class PhaseMachine:
    """Finite-state machine that advances through predefined task phases."""

    phases: tuple[str, ...]
    timeouts: Mapping[str, int] | None = None
    repeat_last_on_timeout: bool = False
    _index: int = field(init=False, default=0)
    _steps_in_phase: int = field(init=False, default=0)
    _history: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("PhaseMachine requires at least one phase.")
        self.reset()

    def reset(self) -> None:
        """Return to the initial phase and clear time counters."""

        self._index = 0
        self._steps_in_phase = 0
        self._history = [self.phases[0]]

    def step(self, signals: Mapping[str, Any]) -> str:
        """Advance the phase machine using the provided signals."""

        self._steps_in_phase += 1
        if self._should_advance(signals):
            self._advance()
        return self.current()

    def current(self) -> str:
        """Return the name of the active phase."""

        return self.phases[self._index]

    def steps_in_phase(self) -> int:
        """Expose the number of consecutive steps spent in the current phase."""

        return self._steps_in_phase

    def history(self) -> tuple[str, ...]:
        """Return the visited phase history."""

        return tuple(self._history)

    def is_terminal(self) -> bool:
        """Return True when the final phase has been completed."""

        return self._index >= len(self.phases) - 1

    # ------------------------------------------------------------------#
    # Private helpers                                                   #
    # ------------------------------------------------------------------#

    def _should_advance(self, signals: Mapping[str, Any]) -> bool:
        if signals.get("collision"):
            return False
        if signals.get("task_success"):
            return True
        if signals.get("phase_complete"):
            return True
        return self._has_timed_out()

    def _has_timed_out(self) -> bool:
        if not self.timeouts:
            return False
        phase_name = self.current()
        limit = self.timeouts.get(phase_name)
        return bool(limit and self._steps_in_phase >= limit)

    def _advance(self) -> None:
        if self._index < len(self.phases) - 1:
            self._index += 1
            self._steps_in_phase = 0
            self._history.append(self.phases[self._index])
        elif not self.repeat_last_on_timeout:
            # Remain in final phase, history already includes it.
            self._steps_in_phase = 0


def phase_signals_from_state(state: Mapping[str, Any], task_info: Mapping[str, Any]) -> dict[str, bool]:
    """Derive phase transition signals from environment state and task info."""

    timeout_threshold = task_info.get("max_steps")
    current_step = task_info.get("step", 0)
    timeout_imminent = bool(timeout_threshold and current_step >= timeout_threshold)
    phase_complete = bool(task_info.get("phase_complete"))
    collision = bool(state.get("collision") or task_info.get("collision"))
    success = bool(task_info.get("task_success"))
    return {
        "phase_complete": phase_complete,
        "collision": collision,
        "timeout_imminent": timeout_imminent,
        "task_success": success,
    }
