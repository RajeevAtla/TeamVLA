# TeamVLA Troubleshooting Guide

Use this checklist when local tooling or tests misbehave.

## Environment & Dependency Issues
- **`uv sync` fails** – Ensure you're using Python 3.10+; run `uv python install 3.10` if needed.
- **Missing optional deps (torch, gradio, imageio)** – Commands and tests degrade gracefully, but install the extras when exercising the full pipeline: `uv add torch --index-url https://download.pytorch.org/whl/cpu`.
- **Namespace clashes** – Delete stale `*.egg-info` directories (`rm -rf teamvla.egg-info`) after refactoring package names, then rerun `uv sync`.

## Tooling Failures
- **`uv run ty check` reports namespace errors** – Regenerate stub caches by removing `.mypy_cache/` and confirm `tool.setuptools.packages` lists the new module.
- **Ruff ignores configuration changes** – Run `uv run ruff --show-files` to verify discovery; ensure `pyproject.toml` is at the repo root.
- **Black reformats large diffs unexpectedly** – Use `black --diff` to preview changes; pin to the repo's configured version to avoid style drift.

## Test Failures
- **Integration tests are slow** – Skip them during iteration with `uv run pytest -m "not slow"`; run the full suite before merging.
- **Newton-dependent tests fail** – Set `NEWTON_AVAILABLE=0` or mark the tests with `@pytest.mark.requires_newton` to gate them until the SDK is available.
- **Flaky random-based tests** – Prefer fixtures from `tests.utils` (deterministic RNG) to stabilise expectations.

## CLI & Script Problems
- **Benchmark CLI cannot load checkpoints** – Confirm the checkpoint path exists and the payload includes `config` + `model_state` fields; fall back to the zero policy otherwise.
- **Gradio demo crashes on import** – Install `gradio` (`uv add gradio`) or use `python -m demos.app --help` to verify the module resolves before launching the UI.
- **Episode rendering fails** – Install `imageio[ffmpeg]` for MP4 support or rely on the `.npy` fallback written by `scripts.render_videos`.
