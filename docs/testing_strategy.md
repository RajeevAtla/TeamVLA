# TeamVLA Testing Strategy

This guide summarizes how the repository's automated tests are organised and how to extend them safely as new features land.

## Test Categories
- **Unit** – Fast checks that exercise functions and classes in isolation (e.g., metrics, rollout utilities, scripted policies).
- **Integration** – Multi-module flows that validate interactions (`tests/test_integration_pipeline.py` covers environment → control → data).
- **CLI & Interfaces** – Argument parsing and lightweight smoke tests for command-line entry points, the benchmark CLI, and the Gradio demo.

## Shared Fixtures
- `lift_env`: Temporary `NewtonMAEnv` instance for tests that need environment context without spinning up full simulations.
- `synthetic_rollout`: Representative rollout summary dictionary used across evaluation tests.
- `synthetic_trajectory`: Observation-like payloads mirroring Track B datasets for schema and dataloader coverage.
- `config_loader`: Helper that loads YAML configs from `configs/`, ensuring CLI/tests use validated settings.

## Utility Helpers
- `tests.utils.random_rgb` generates deterministic RGB tensors for fixtures that need image content.
- `tests.utils.rollout_summary` standardises the structure expected by evaluation metrics.
- `tests.utils.assert_close` offers a concise assertion helper for numeric comparisons.

## Pytest Markers
- `slow`: Applied to longer-running end-to-end flows so they can be excluded with `pytest -m "not slow"`.
- `requires_newton`: Reserve for cases that depend on the proprietary Newton runtime (currently unused but ready).

## Running The Suite
- `uv run pytest` – Default quick pass over all tests (slow tests can be skipped as `uv run pytest -m "not slow"`).
- `uv run pytest tests/test_integration_pipeline.py -m slow` – Focus on slower integration runs when needed.
- `uv run pytest -k "demo"` – Targeted filters for feature-specific loops during development.

## Evaluation Summary Template
Downstream tracks can record benchmark outcomes using the template below. Copy the table and populate it per checkpoint.

| Date | Agent | Tasks | Episodes | Success Rate | Success@T | Avg Time | Coordination | Collision | Notes |
|------|-------|-------|----------|--------------|-----------|----------|--------------|-----------|-------|
| 2024-06-01 | Track C | lift, handoff, drawer | 5 | 0.60 | 0.60 | 87.4 | 0.82 | 0.05 | |

Document the associated checkpoint/config name alongside the table for reproducibility.
