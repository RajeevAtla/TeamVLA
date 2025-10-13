"""Phase machine utilities for coordinating scripted controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class PhaseMachine:
    """Finite-state machine that advances through predefined task phases."""

    phases: tuple[str, ...]
    timeouts: Mapping[str, int] | None = None
    _index: int = field(init=False, default=0)
    _steps_in_phase: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("PhaseMachine requires at least one phase.")
        self.reset()

    def reset(self) -> None:
        """Return to the initial phase and clear time counters."""

        self._index = 0
        self._steps_in_phase = 0

    def step(self, signals: Mapping[str, bool]) -> str:
        """Advance the phase machine using the provided signals."""

        self._steps_in_phase += 1
        if self._should_advance(signals):
            self._advance()
        return self.current()

    def current(self) -> str:
        """Return the name of the active phase."""

        return self.phases[self._index]

    def is_terminal(self) -> bool:
        """Return True when the final phase has been completed."""

        return self._index >= len(self.phases) - 1

    def _should_advance(self, signals: Mapping[str, bool]) -> bool:
        complete = signals.get("phase_complete", False)
        timed_out = self._has_timed_out()
        return bool(complete or timed_out)

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


def phase_signals_from_state(state: Mapping[str, Any], task_info: Mapping[str, Any]) -> dict[str, bool]:
    """Derive phase transition signals from environment state and task info."""

    return {
        "phase_complete": bool(task_info.get("phase_complete")),
        "collision": bool(state.get("collision", False)),
        "timeout_imminent": bool(task_info.get("step", 0) >= task_info.get("max_steps", 0)),
    }
