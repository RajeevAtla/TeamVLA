<!-- Coordinated backlog enabling three agents to work in parallel -->

# TeamVLA Parallel Planning Backlog

This document distributes the full scope of the project across three worktracks so agents can operate concurrently with minimal contention. Each track owns implementation, testing, and documentation for its domain, while shared coordination items ensure consistency. Tasks remain exhaustive—agents should check off items as they complete them and surface blockers via the dependencies listed below.

## Shared Coordination Layer (All Agents)
- [ ] Finalize tooling baseline (`Python`, `uv`, `torch`, `numpy`, `pytest`, `ruff`, `mypy`) and record commands plus OS-specific notes in `docs/tools.md`.
- [ ] Maintain `CHANGELOG.md` capturing milestones by date, responsible agent, and summary.
- [ ] Keep `.gitignore`, `README.md`, and `pyproject.toml` in sync with cross-track changes; coordinate edits to avoid merge conflicts.
- [ ] Agree on Definition of Done: code updated, relevant tests written, `uv run ruff check .`, `uv run ty check`, and `uv run pytest` pass, documentation refreshed.
- [ ] Meet at phase boundaries to confirm integration points and update the dependency table.

### Dependency Table
| Upstream (provides) | Downstream (consumes) | Notes |
| --- | --- | --- |
| Track A: env/task interfaces | Track B: dataloaders, models | Track A must document observation/action schemas. |
| Track A scripted policies | Track B data pipeline | Waypoint outputs feed EpisodeWriter fixtures. |
| Track B schema definitions | Track C evaluation/tests | Track B provides typed schema utilities used in metrics/tests. |
| Track B checkpoints | Track C benchmark/demo | Ensure save/load formats and config structures are fixed. |
| Track C tooling configs | Tracks A & B | Any lint/type/test adjustments must be communicated to other tracks. |

---

## Track A — Simulation & Control (Agent A)
**Scope:** Repository bootstrap for environment/control folders, Newton environment scaffolding, task registry, action generation utilities, and scripted policies.

### Phase 0 Responsibilities
- [x] Create directory skeleton for `envs/` and `control/` with `__init__.py` placeholders and TODO docstrings.
- [x] Contribute to `pyproject.toml` by listing environment/control-specific dependencies (e.g., `numpy`, `scipy`) and confirm they appear in `requirements.txt`.
- [x] Add environment-focused sections to `README.md` (Newton setup, task overview, control philosophy).

### Environment & Task Scaffolding
- [x] Implement `EnvironmentConfig` dataclass and parsing helper in `envs/core_env.py`.
- [x] Author `NewtonMAEnv` lifecycle methods (`__init__`, `reset`, `step`, `render`, `close`, `set_task`, `get_state_dict`) with deterministic RNG seeding and Newton guards.
- [x] Build helper functions (`build_multi_robot_world`, `setup_cameras`, `make_solver`, `obs_i`, `render_topdown`, `_unused`).
- [x] Flesh out action-handling utilities (`_validate_actions`, `_apply_actions`, `_handle_object_attachments`, `_advance_phase`).
- [x] Implement task protocol and registry in `envs/tasks/base.py`, including `TaskSpec`, metadata dataclasses, and registry helpers (`register_task`, `deregister_task`, `get_task`, `iter_registered_tasks`).
- [x] Scaffold task modules `lift.py`, `handoff.py`, `drawer.py`; include documented phase flows, reward placeholders, and scripted action stubs.
- [x] Provide public exports in `envs/__init__.py`.

### Control Utilities & Scripted Policies
- [x] Expand `control/ik_utils.py` with IK solvers, rendezvous planning, gripper mapping, and helper functions.
- [x] Implement `PhaseMachine` dataclass in `control/phase_machine.py` with transition helpers and signal extraction.
- [x] Develop scripted policy modules in `control/scripted/` (lift, handoff, drawer): deterministic waypoint libraries, policy functions integrating IK utilities.
- [x] Ensure control modules document expected observation/action formats for Track B consumption.

### Testing & Validation
- [x] Author tests covering environment/control behavior (`tests/test_env_skeleton.py`, `tests/test_task_registry.py`, `tests/test_ik_utils.py`, `tests/test_phase_machine.py`, `tests/test_scripted_policies.py`).
- [x] Add fixtures in `tests/conftest.py` for dummy tasks/environment as soon as Track C scaffolds shared fixtures (note dependency).
- [x] Run `uv run pytest`, `uv run ruff check .`, `uv run ty check` after each major feature (baseline suite verified).

### Documentation Deliverables
- [x] Update `README.md` with environment usage examples and control policy descriptions.
- [x] Document Newton installation placeholders and how to plug in real physics once available.
- [x] Maintain task reference notes (`planning/phase_1.md`, `phase_2.md`) with cross-links to actual code sections.

### Hand-off Requirements
- [x] Deliver observation/action schema summary to Track B (`docs/interfaces/env_control.md`).
- [x] Provide deterministic scripted policy outputs for Track B's dataset fixtures (sample JSON/NPZ in `tests/fixtures/` once pipeline available).

---

## Track B — Data & Learning Pipeline (Agent B)
**Scope:** Data schema, episode writing, dataloaders, configuration files, encoder/model scaffolding, training utilities, and checkpointing.

### Phase 0 Responsibilities
- [ ] Extend directory skeleton for `data/`, `models/`, `train/`, `configs/`, and ensure `__init__.py` stubs exist.
- [ ] Populate `pyproject.toml` dependency sections for data/model/training needs (`pyyaml`, `torch`, `tqdm`, `rich`), and sync with `requirements.txt`.
- [ ] Enrich `README.md` with data pipeline overview, config usage, and training workflow summary.

### Data Schema & Pipeline
- [ ] Implement `data/schema.py`: typed structures (`StepData`, `EpisodeMeta`), validators (`validate_step`, `validate_episode_meta`), dtype helpers, and exception types.
- [ ] Build `EpisodeWriter` in `data/writer.py` with context manager support, serialization strategies, and configurable formats (NPZ placeholder).
- [ ] Create `MultiTaskDataset` and `make_dataloader` in `data/dataloader.py`; include index caching, transform chaining, collate functions, and deterministic seeding.
- [ ] Draft YAML config files (`configs/common.yaml`, `lift.yaml`, `handoff.yaml`, `drawer.yaml`, `train_bc.yaml`) with commented placeholders and consistent naming.
- [ ] Produce sample dataset artifacts for tests (e.g., synthetic episodes in `tests/fixtures/data/`).

### Models & Training Loop
- [ ] Implement encoder builders (`models/encoders/vision.py`, `models/encoders/language.py`) with configurable stubs, forward helpers, and tokenizer.
- [ ] Implement `SingleBrainVLA` with modality fusion, action heads, `act`, `loss`, and helper methods for pre-processing.
- [ ] Provide message-passing variant `MsgPassingVLA` (base class or wrapper) aligning with Track A’s environment exports.
- [ ] Author training utilities:
  - [ ] `train/losses.py` (BC loss, coordination penalties, aggregator dictionary).
  - [ ] `train/schedulers.py` (optimizer factory, cosine warmup scheduler).
  - [ ] `train/bc_trainer.py` functions (`set_seed`, `build_model`, `build_data`, `train_one_epoch`, `evaluate`, `save_checkpoint`, `load_checkpoint`, `main`) with logging.
- [ ] Ensure checkpoints store `config`, `model_state`, `optimizer_state`, optional scheduler state, and `TrainingState`.

### Testing & Validation
- [ ] Create tests: `tests/test_data_schema.py`, `tests/test_episode_writer.py`, `tests/test_dataloader.py`, `tests/test_configs.py`.
- [ ] Add model/training tests: `tests/test_encoders.py`, `tests/test_models.py`, `tests/test_losses.py`, `tests/test_bc_trainer.py`.
- [ ] Use fixtures from Track C once available; otherwise provide local fixtures for schema/model tests.
- [ ] Validate compatibility with Track A outputs (mock environment observations match expected schemas).

### Documentation Deliverables
- [ ] Produce `docs/data_pipeline.md` explaining schema, writer usage, and dataloader structure.
- [ ] Update `README.md` training section with commands (`uv run python -m train.bc_trainer`, config overrides).
- [ ] Summarize config keys and dataset expectations in `configs/README.md`.

### Hand-off Requirements
- [ ] Provide schema documentation and serialized sample episodes to Track C for evaluation/tests.
- [ ] Deliver saved checkpoints (synthetic) demonstrating `save_checkpoint`/`load_checkpoint` compatibility for Track C benchmarking.

---

## Track C — Evaluation, Tooling, & Developer Experience (Agent C)
**Scope:** Evaluation metrics, rollout utilities, benchmark CLI, demo app, scripts, test infrastructure, lint/type tooling, developer docs, and CI scaffolding.

### Phase 0 Responsibilities
- [ ] Create `eval/`, `demos/`, `scripts/`, `tests/`, `docs/` scaffolding with `__init__.py` or README placeholders as appropriate.
- [ ] Finalize `pyproject.toml` tool sections (ruff, black, mypy, pytest) and ensure shared scripts exist for linting/type/tests.
- [ ] Update `README.md` with evaluation overview, testing philosophy, and tooling usage instructions.

### Evaluation & Demo Implementation
- [ ] Implement metrics in `eval/metrics.py` (`success_at_T`, `time_to_success`, `coordination_score`, `collision_cost`, `aggregate_results`) plus helper functions.
- [ ] Build rollout utilities (`eval/rollouts.py`): `run_episode`, `run_suite`, and supporting helpers with deterministic seeding and logging.
- [ ] Implement benchmark CLI (`eval/bench.py`): argument parsing, `build_env`, `load_policy`, `benchmark`, and output serialization (JSON/Markdown).
- [ ] Develop Gradio demo (`demos/app.py`) with Blocks layout, policy loading fallback, inference step, and main entrypoint.
- [ ] Implement scripts: `scripts/collect_demos.py` (leveraging Track A/B components) and `scripts/render_videos.py` (frame rendering, CLI args).

### Testing Infrastructure & QA
- [ ] Structure `tests/` hierarchy mirroring packages; create `tests/conftest.py` with shared fixtures (dummy env, synthetic trajectory, config loader).
- [ ] Provide utility helpers in `tests/utils/` (tensor comparison, random data generators).
- [ ] Author evaluation/demo tests: `tests/test_metrics.py`, `tests/test_rollouts.py`, `tests/test_bench_cli.py`, `tests/test_demo_app.py`, `tests/test_scripts.py`.
- [ ] Coordinate with Track A/B to mock or import their outputs (fixtures for observations, episodes, checkpoints).
- [ ] Configure `pytest` markers (`slow`, `requires_newton`) and ensure default test run skips heavy/newton-dependent cases.

### Tooling & Developer Experience
- [ ] Finalize lint/type/test tooling:
  - [ ] Configure `ruff` rules, `black` settings, `mypy` strictness, and `pytest` options in `pyproject.toml`.
  - [ ] Author documentation for running `uv run ruff check .`, `uv run ty check`, `uv run pytest`, `uv run mypy`.
- [ ] Prepare CI workflow or scripted command sequence for lint → type → tests (documented even if file not committed).
- [ ] Write contributor guide (`docs/contributing.md`) covering branching strategy, review checklist, pre-commit recommendations.
- [ ] Add troubleshooting guide for tooling failures, environment setup, and common errors.

### Documentation Deliverables
- [ ] Update `README.md` evaluation/demo/testing sections with usage examples and interpretation guidance.
- [ ] Produce `docs/testing_strategy.md` summarizing test categories, fixtures, and markers.
- [ ] Provide `docs/tooling.md` referencing CLI commands, expected outputs, and maintenance tips.

### Hand-off Requirements
- [ ] Supply evaluation summary templates to Track A/B for reporting results.
- [ ] Provide automation scripts or instructions that the other tracks integrate into their workflows.

---

## Cross-Track Milestones
- **Milestone M1 (Tracks A & B):** Environment, task registry, scripted policies, schema, and episode writer complete; synthetic data generation possible.
- **Milestone M2 (Tracks B & C):** Models, training loop, and evaluation metrics wired—able to run `uv run pytest` covering data/model/eval tests.
- **Milestone M3 (All Tracks):** Benchmark CLI and demo operational with synthetic checkpoints; documentation synchronized.
- **Milestone M4 (All Tracks):** Full QA pipeline enforced (lint/type/tests/CI scripts) and contributor guide published.

## Risk & Decision Log
- [ ] Confirm Newton engine availability/licensing (Track A lead; inform Track C for skip markers).
- [ ] Decide on episode serialization format (Track B lead; Tracks A & C must adapt to any change).
- [ ] Validate torch + uv compatibility on all target OSes (Track B lead with Track C testing support).
- [ ] Establish policy for GPU usage in training/evaluation (Track B propose, Track C document, Track A ensure env compatibility).
- [ ] Plan long-term checkpoint registry and naming conventions (Track B lead; Track C update benchmark loader).

## Continuous Improvement Backlog
- [ ] Evaluate automated documentation tooling (Sphinx/MkDocs) once APIs stabilize (Track C coordinate, A/B contribute).
- [ ] Plan integration tests combining environment, scripted policies, writer, dataloader, and training loop (joint effort once core pieces exist).
- [ ] Explore additional baselines (e.g., reinforcement learning) after BC pipeline proven (Track B).
- [ ] Consider dashboard/visualization tools for benchmark history (Track C).
- [ ] Draft guidelines for extending to more agents/tasks (Track A lead with contributions from others).
