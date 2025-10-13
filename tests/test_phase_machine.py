"""Tests for the PhaseMachine helper."""

from __future__ import annotations

from control.phase_machine import PhaseMachine, phase_signals_from_state


def test_phase_machine_advances_on_completion() -> None:
    machine = PhaseMachine(phases=("reach", "grasp"))
    assert machine.current() == "reach"
    machine.step({"phase_complete": True})
    assert machine.current() == "grasp"
    assert machine.is_terminal()


def test_phase_machine_timeouts_trigger_advance() -> None:
    machine = PhaseMachine(phases=("reach", "grasp"), timeouts={"reach": 2})
    machine.step({"phase_complete": False})
    machine.step({"phase_complete": False})
    assert machine.current() == "grasp"


def test_phase_signals_from_state_extracts_flags() -> None:
    signals = phase_signals_from_state({"collision": True}, {"phase_complete": 1, "step": 5, "max_steps": 5})
    assert signals["phase_complete"]
    assert signals["collision"]
    assert signals["timeout_imminent"]
