# Phase 6: Testing, Quality Assurance, and Tooling Integration

## Objectives
- Build out the automated testing framework, lint/format tooling, and continuous integration scaffolding to enforce the “small functions, testable units” mandate.
- Ensure every module from previous phases has corresponding unit tests that can run without Newton assets.
- Document workflows for running tests, linters, and type checkers locally and in CI.

## Deliverables
- Comprehensive `tests/` suite covering imports, env/tasks, control, data, models, training, evaluation, demos, and scripts (as outlined in earlier phases).
- `conftest.py` with shared pytest fixtures (e.g., temporary config loader, synthetic trajectories, fake environment).
- `pytest.ini` or equivalent configuration (integrated into `pyproject.toml`) specifying test discovery patterns, markers (`@pytest.mark.slow`, `@pytest.mark.requires_newton` placeholder).
- Lint/format/type tooling configuration:
  - `ruff` ruleset for linting (enable flakes, import sorting, complexity checks).
  - `black` formatting profile (line length 100).
  - `mypy` configuration targeting strict optional checks where feasible.
- Optional GitHub Actions workflow (`.github/workflows/ci.yaml`) or documented script in README guiding setup (actual workflow file can be deferred if repository policy discourages yet).
- Developer guide (`docs/contributing.md` or README section) describing coding standards, review checklist, and test matrix.

## Task Breakdown
1. **Pytest Suite Completion**
   - Implement missing tests referenced in Phases 0-5; confirm each test module targets a single domain.
   - Use fixtures to avoid repetition (e.g., `dummy_env_cfg`, `mock_task_class`, `synthetic_episode`).
   - Mark Newton-dependent tests with `pytest.mark.requires_newton` so they can be skipped when engine unavailable.
   - Ensure each test function is concise and asserts only a narrow behavior.
2. **Shared Utilities**
   - Create `tests/utils/` with helpers for generating random tensors, sample configs, and verifying logging output.
   - Provide `assert_tensor_close` helper for numeric comparisons without relying on PyTorch built-ins (keeps tests framework-agnostic).
3. **Tooling Setup**
   - Configure `ruff` to enforce import order, avoid unused variables, and limit function complexity (e.g., `max-complexity = 10`).
   - Integrate `mypy` with modules typed at least to `--strict` minus known blockers; specify per-module overrides if necessary.
   - Document commands in README:
     - `python -m pytest`
     - `ruff check .`
     - `black --check .`
     - `mypy .`
4. **Continuous Integration Preparation**
   - Draft GitHub Actions workflow (if allowed) or provide script snippet demonstrating sequential execution of lint -> type check -> tests.
   - Ensure workflow/test instructions include caching pip dependencies and using matrix for multiple Python versions (3.10/3.11).
5. **Developer Workflow Documentation**
   - Add `docs/contributing.md` (or README section) with:
     - Branching strategy suggestions.
     - Code review checklist (unit tests updated, docstrings present, logging used, functions small).
     - How to run selective test markers (`pytest -m "not requires_newton"`).
     - Guidance on adding new modules (update `__all__`, add tests, update configs).

## Testing Strategy
- Run entire pytest suite locally to ensure coverage across modules; document expected run time and dependencies.
- Provide instructions for optional coverage reporting (`pytest --cov=teamvla`), even if coverage tool not yet configured.
- For lint/type tools, include smoke tests (e.g., `tests/test_tooling_configs.py`) that parse config files and assert key settings.
- Encourage pre-commit hooks (document `pre-commit` optional integration) to run the tool stack before commits.

## Quality & Maintainability Notes
- Maintain high cohesion: tests should mirror module structure (`tests/envs/test_core_env.py`, etc.) for navigability.
- Keep fixtures lightweight and pure; avoid expensive setup/teardown operations.
- Establish naming conventions for test data directories (e.g., `tests/fixtures/episodes/`).
- Ensure documentation emphasizes continuous enforcement of “small, testable functions” and warns against introducing God methods.
