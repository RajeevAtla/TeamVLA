# Contributing to TeamVLA

Thank you for helping build the TeamVLA benchmark. This document summarizes the key practices we follow across the repository.

## Development Workflow

- Work phase-by-phase (`planning/phase_*.md`). Each phase defines deliverables, tests, and quality bars.
- Keep functions short and single-purpose. Extract helpers to maintain readability and simplify testing.
- Use type hints and docstrings for all public functions and classes.
- Prefer dependency injection over globals. Pass configuration explicitly.
- Log with the standard `logging` module; avoid printing in library code.

## Testing & Tooling

- Run the full test suite before submitting changes:

  ```bash
  uv run pytest
  ```

- Lint and format:

  ```bash
  uv run ruff check .
  uv run black --check .
  uv run mypy .
  ```

- Torch-heavy tests set `OMP_NUM_THREADS=1` via `tests/conftest.py` to keep runtimes predictable.
- Mark Newton-dependent tests with `@pytest.mark.requires_newton` so they can be skipped when the engine is absent.

## Pull Request Checklist

1. Phase roadmap updated (if scope changes).
2. Unit tests added or updated.
3. Documentation reflects new behavior or CLI options.
4. Runbook / configs adjusted when introducing new parameters.
5. Ensure new dependencies are captured in `pyproject.toml` and `requirements.txt`.

## Additional Notes

- Place large assets under `assets/` (ignored by git). Document download instructions there.
- Configuration defaults live under `configs/`; prefer referencing them from code rather than duplicating literals.
- For CI integration, mirror the `pytest` + `ruff` + `mypy` steps above in your workflow.
