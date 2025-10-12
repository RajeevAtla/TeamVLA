# Phase 5: Evaluation Pipelines and Interactive Demo

## Objectives
- Provide scaffolding for evaluation metrics, rollout utilities, benchmarking CLI, and the Gradio/WebRTC demo app.
- Maintain tight, single-responsibility functions with extensive docstrings and TODO placeholders for integration with real environment/policy logic.
- Ensure evaluation components consume outputs that match the schema defined in earlier phases, enabling end-to-end validation once models are implemented.

## Deliverables
- `eval/metrics.py`:
  - Functions: `success_at_T`, `time_to_success`, `coordination_score`, `collision_cost`, `aggregate_results`.
  - Helper utilities (`_compute_success`, `_normalize_scores`) to keep public functions concise.
- `eval/rollouts.py`:
  - `run_episode` and `run_suite` stubs orchestrating environment-policy interaction with deterministic seeding and logging hooks.
  - Private helpers (`_reset_env`, `_collect_step`, `_finalize_episode`) for modularity.
- `eval/bench.py`:
  - CLI entrypoint using `argparse` with options for tasks, episodes, checkpoint, output file.
  - `main` should load configs, instantiate env & policy via factory functions, and call rollout utilities (placeholders for now).
- `demos/app.py`:
  - Gradio interface skeleton with `load_policy`, `inference_step`, and `main`.
  - Keep logic modular: small helper to format observation frames, another to render status text.
- Update `scripts/collect_demos.py` and `scripts/render_videos.py` to align with evaluation utilities (import stubbed functions, provide CLI placeholders).

## Task Breakdown
1. **Metrics Module**
   - Each metric function accepts clearly typed trajectories (list/dict) consistent with data schema.
   - Include early input validation and descriptive exceptions for missing keys.
   - `aggregate_results` aggregates list of per-episode dictionaries into summary stats; use helper functions for mean/std/percentile to keep code short.
2. **Rollout Utilities**
   - `run_episode`:
     - Accept `policy` callable returning actions, `instruction` string, `max_steps`.
     - Placeholder loop collecting observations into `traj` dict with TODO markers.
   - `run_suite` orchestrates multiple episodes across tasks; use helper `_iter_tasks` to handle round-robin scheduling.
   - Provide hooks for deterministic seeding and logging.
3. **Benchmark CLI**
   - `bench.py` should parse CLI arguments, configure logging level, instantiate environment/policy through factory functions (`build_env`, `load_checkpoint` placeholders).
   - Write evaluation summaries to both stdout (via logging) and optional CSV/Markdown files.
   - Keep `main` small by delegating to `_parse_args`, `_run_benchmark`.
4. **Demo App**
   - Build Gradio Blocks layout with two video feeds and instruction text box.
   - Use placeholder frames (e.g., numpy zeros) until real rendering is available.
   - `load_policy` should handle missing checkpoint gracefully and log instructions for users.
   - Provide `if __name__ == "__main__": main()` guard.
5. **Scripts**
   - `collect_demos.py`: CLI for generating scripted demonstrations; tie into `data.EpisodeWriter` and `control.scripted` modules.
   - `render_videos.py`: CLI for transforming episodes to MP4 using OpenCV placeholders.
   - Both scripts should have `main` functions calling small helper functions to maintain readability.

## Testing Strategy
- `tests/test_metrics.py`:
  - Create synthetic trajectories and verify metric outputs (success rates, coordination).
  - Each metric tested independently; include edge cases (empty traj, missing keys).
- `tests/test_rollouts.py`:
  - Utilize mock environment/policy to confirm `run_episode` returns dict with required keys and `run_suite` iterates tasks properly.
  - Ensure deterministic behavior with fixed RNG seeds.
- `tests/test_bench_cli.py`:
  - Use `argparse` test helpers or `pytest` `capsys` to validate CLI argument parsing and high-level flow.
  - Mock heavy dependencies to keep tests fast.
- `tests/test_demo_app.py`:
  - Verify Gradio Blocks instantiation and callback wiring using monkeypatch/mocks.
  - Ensure `load_policy` returns placeholder objects even when checkpoint missing.
- `tests/test_scripts.py`:
  - Write targeted tests for CLI functions (argument parsing, main entrypoints) with dependency injection.

## Quality & Maintainability Notes
- Document expected trajectory dict structure at top of `metrics.py` / `rollouts.py` for cross-reference with Phase 3.
- Keep CLI interfaces consistent (shared `--config`, `--seed`, `--device` flags) to reduce user friction.
- Provide fallback behaviors when optional packages (e.g., Gradio) aren’t installed; raise informative `ImportError` with install instructions.
- Use logging extensively to trace evaluation steps without relying on prints.
