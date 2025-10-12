# Phase 3: Data Pipeline and Configuration Layer

## Objectives
- Build the data handling backbone covering schema validation, episode writing, dataset loading, and configuration management.
- Ensure all functions remain granular, deterministic, and directly testable without Newton assets.
- Define configuration files (`configs/*.yaml`) with consistent naming and clear separation between common options and task/training specifics.

## Deliverables
- `data/schema.py`:
  - `STEP_FIELDS` constant describing expected keys, shapes, and dtypes (use `TypedDict` or dataclass for clarity).
  - Validation helpers (`validate_step`, `validate_episode_meta`) with explicit error messages.
  - Utility functions for converting raw numpy arrays to tensors (placeholder stubs).
- `data/writer.py`:
  - `EpisodeWriter` class handling streaming writes to `npz`/`hdf5` (initial implementation can store data in memory and write placeholder files).
  - Small helper functions for path management (`_make_episode_dir`, `_serialize_step`).
  - Context manager support (`__enter__`, `__exit__`) to simplify usage in scripted collection scripts.
- `data/dataloader.py`:
  - `MultiTaskDataset` with configuration-driven task filtering and transform pipelines.
  - `make_dataloader` factory hooking into PyTorch `DataLoader` with deterministic seeding.
  - Private helpers (`_load_episode_index`, `_stack_batch`) to keep public methods minimal.
- Config files:
  - `configs/common.yaml` capturing seeds, device, logging.
  - Task configs (`lift.yaml`, `handoff.yaml`, `drawer.yaml`) listing randomization ranges, asset paths, reward weights.
  - `configs/train_bc.yaml` mapping model/training hyperparameters (batch size, scheduler, loss weights, dataset roots).
- Update `README.md` with configuration usage instructions and environment variables (e.g., `TEAMVLA_CONFIG` override).

## Task Breakdown
1. **Schema Definition**
   - Use `typing.TypedDict` for `StepData`, `EpisodeMeta` to enforce key presence.
   - `validate_step` iterates fields and raises `SchemaError` (custom exception) with per-field details.
   - Provide helper `ensure_np_dtype(array, dtype)` to centralize dtype casting.
2. **Episode Writer**
   - Constructor accepts `out_dir`, `fmt`, `compress`, `max_steps`.
   - `start_episode` sets up buffers; `add_step` validates via schema module; `end_episode` writes file and returns path.
   - `close` flushes and resets state; ensure idempotency.
   - Use small helpers for file naming (timestamp + uuid) to keep methods short.
3. **Dataset Loader**
   - Build an index in `__init__` (list of `(episode_path, step_idx)`).
   - `__getitem__` loads a single step, applies transforms (callable chain).
   - Provide `collate_fn` for consistent batching to be consumed by `DataLoader`.
   - Document expectation that transforms are pure functions returning dict copies.
4. **Configurations**
   - Draft YAML templates with placeholders referencing assets (e.g., `assets/README.md` for download instructions).
   - Include comments describing each parameter to guide future implementers.
   - Ensure naming consistency (`task_name`, `dataset_roots`, `lambda_sync`).

## Testing Strategy
- `tests/test_data_schema.py`:
  - Validate correct schema passes and incorrect entries raise detailed errors.
  - Keep tests modular: one test per field type validation.
- `tests/test_episode_writer.py`:
  - Use temp directories (via `tmp_path`) to ensure writer creates files and resets state.
  - Mock `numpy.savez` if necessary to avoid heavy IO; verify function call parameters.
- `tests/test_dataloader.py`:
  - Construct synthetic dataset directories with minimal `.npz` fixtures.
  - Check deterministic ordering given a fixed seed and ensure collate function preserves keys.
- `tests/test_configs.py`:
  - Load YAML files using `yaml.safe_load` and assert required keys exist.
  - Each test function should focus on a single config file to keep scope tight.

## Quality & Maintainability Notes
- Avoid monolithic methods; prefer private helpers and dataclasses for shared structures (e.g., `DatasetIndexEntry`).
- Provide module-level constants for filenames (`EPISODE_FILE_EXT = ".npz"`) to reduce magic strings.
- For YAML configs, include `TODO` comments referencing future tuning phases; keep them human-readable (sorted keys).
- Ensure all IO operations are wrapped in error handling that surfaces actionable messages for missing directories or permission issues.
