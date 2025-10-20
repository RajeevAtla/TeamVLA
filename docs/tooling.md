# TeamVLA Tooling Reference

This document captures the agreed-upon commands and expectations for linting, static analysis, and testing. All commands run via `uv` so they resolve the virtual environment and dependency groups automatically.

## Core Commands
- `uv run ruff check .` – Style and correctness linting (`tool.ruff` controls rule selection and ignore lists).
- `uv run black --check .` – Formatting check (mirrors `tool.black` settings).
- `uv run ty check` – Type-checking through the `ty` runner; wraps `mypy` using the configuration in `pyproject.toml`.
- `uv run mypy .` – Direct `mypy` invocation when deeper debugging output is required.
- `uv run pytest` – Execute the full pytest suite; combine with markers such as `-m "not slow"` for faster iterations.

## Usage Patterns
- Run `uv run ruff check . && uv run black --check . && uv run ty check && uv run pytest && uv run mypy .` before submitting changes.
- CI (`.github/workflows/ci.yaml`) mirrors the same sequence, so surprises at push time are rare.
- Add `-- --maxfail=1 -x` after any command to halt on first failure during debugging (e.g., `uv run pytest -- --maxfail=1 -x`).

## Maintenance Tips
- Keep `pyproject.toml` as the single source of truth for tool versions and configuration-update it rather than sprinkling command-line flags in scripts.
- When introducing new optional dependencies, document skip behaviour in `tests/testing_strategy.md` and gate tests with markers as appropriate.
- If `ty` or `mypy` emits namespace-package warnings, verify that the package is listed under `[tool.setuptools]` and that `__init__.py` files exist.
- Ruff rule adjustments should be discussed across tracks; add comments in `pyproject.toml` if temporarily suppressing checks.

## OS-Specific Notes

### Windows
- Prefer PowerShell terminals; prefix commands with `uv run` as shown above.
- When installing optional packages (e.g., torch CPU wheels) use `uv pip install --index-url https://download.pytorch.org/whl/cpu torch`.
- Add `%APPDATA%\uv\python\<version>\Scripts` to your `PATH` if you want direct access to `pytest`/`ruff` shims outside of `uv run`.

### macOS
- Ensure Xcode Command Line Tools are installed (`xcode-select --install`) before compiling optional wheels.
- For Apple Silicon, pick the `macosx_11_0_arm64` torch wheels via `uv pip install --index-url https://download.pytorch.org/whl/cpu torch`.
- Use `arch -arm64 uv run ...` when forcing Apple Silicon execution from Intel Rosetta shells.

### Linux
- Install system build essentials (`build-essential`, `python3-dev`, `libgl1`) before running the full test suite with vision dependencies.
- If running inside containers, mount the repo at `/workspace` and rely on `uv run` to bootstrap virtual envs under `.venv`.
- CUDA-enabled hosts should set `UV_PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124` (for example) prior to `uv pip install torch`.
