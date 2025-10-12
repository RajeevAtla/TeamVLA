# Phase 1: Environment and Task Scaffolding

## Objectives
- Implement import-ready skeletons for `envs/core_env.py` and `envs/tasks/*.py` that strictly follow the interfaces defined in `architecture.md`.
- Set up a registration mechanism (`TASK_REGISTRY`) enabling dynamic task selection without circular imports.
- Provide richly documented placeholders for Newton integration while keeping pure-Python fallbacks for testing.
- Encode small, single-purpose helper functions (e.g., observation builders, RNG seeding) to satisfy the “small functions” mandate.

## Deliverables
- `envs/core_env.py` containing:
  - `NewtonMAEnv` class with stubbed lifecycle methods (`__init__`, `reset`, `step`, `render`, `close`, `set_task`, `get_state_dict`).
  - Helper functions: `build_multi_robot_world`, `setup_cameras`, `make_solver`, `obs_i`.
  - Structured logging placeholders and error messages that explain missing Newton dependencies.
  - Dataclass or typed configuration models (lightweight) to keep parameter passing explicit.
- `envs/tasks/base.py` defining the `TaskSpec` protocol and `register_task` function, plus utility functions for task discovery.
- Task modules (`lift.py`, `handoff.py`, `drawer.py`) implementing skeleton classes that inherit `TaskSpec` and register themselves via side effects.
- `envs/__init__.py` exposing public API for simple imports (`from envs import NewtonMAEnv`).

## Task Breakdown
1. **Core Environment Skeleton**
   - Define `NewtonMAEnv` with dependency injection for `task_name`, `cfg`, `rng_seed`.
   - Each method should raise `NotImplementedError` or return placeholder values with clear TODO comments.
   - Introduce `_validate_actions`, `_compute_rewards`, `_gather_info` helper stubs to keep main methods minimal.
   - Wrap Newton imports in try/except, set `_NEWTOWN_AVAILABLE` flag, and explain fallback path.
2. **Task Protocol and Registry**
   - `TaskSpec` to live in `envs/tasks/base.py` with `typing.Protocol`.
   - Add `TaskInfo` dataclass for metadata (phases, success thresholds) to reduce dict usage.
   - Implement `TASK_REGISTRY` as `dict[str, TaskSpec]`, plus `register_task(task_cls)` decorator that instantiates or stores class references.
   - Provide `get_task(name: str)` helper that raises descriptive `KeyError`.
3. **Individual Task Stubs**
   - Each task file defines a class (e.g., `LiftTask`) with method stubs and module-level `register_task(LiftTask)` executed at import.
   - Document expected assets, randomization knobs, and reward terms directly in docstrings.
   - Include helper placeholders (`_compute_alignment_reward`, `_plan_drawer_waypoints`) keeping logic isolated when implemented later.
4. **Configuration Coupling**
   - Reference `configs/{lift,handoff,drawer}.yaml` in docstrings to ensure future authors align parameter names (e.g., `grasp_height`, `handoff_radius`).
   - Ensure environment accepts `cfg: Mapping[str, Any]` and defers value lookups to these configs.

## Testing Strategy
- Create `tests/test_env_skeleton.py`:
  - Verify `NewtonMAEnv` can be instantiated with dummy config and mocked tasks.
  - Confirm `reset`/`step` return shape placeholders (e.g., lists of dicts) and raise `RuntimeError` if Newton missing.
  - Use stub task implementing `TaskSpec` with minimal state to assert registry mechanics.
- Add `tests/test_task_registry.py` to cover register/get logic, duplicate detection, and error messaging.
- Keep tests granular: each test function should cover one behavior (import, registry, error path).

## Quality & Maintainability Notes
- Favor dataclasses/TypedDicts for configuration and observation stubs to make type narrowing easier in later phases.
- Keep logging statements at debug level by default; configure logger in Phase 0.
- Explicitly mark unimplemented sections with `# TODO(phaseX): ...` to trace future ownership.
- Maintain deterministic behavior by seeding RNG in `__init__` and `reset`.
