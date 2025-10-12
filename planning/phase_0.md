# Phase 0: Bootstrap Foundation

## Objectives
- Establish the full repository directory tree exactly as defined in `architecture.md` / `prompt.md`.
- Provide package metadata (`pyproject.toml` or `setup.cfg` + `requirements.txt`) aligned with the documented dependency set (`torch`, `numpy`, `opencv-python`, `gradio`, Newton SDK placeholder).
- Seed each package with `__init__.py` markers, module docstring scaffolds, and TODO placeholders so future phases can plug in functionality without restructuring.
- Extend the root `README.md` with installation prerequisites (Python version, Newton engine notes, optional GPU support), repository orientation, and contribution/testing guidelines.

## Deliverables
- Directory skeleton with empty or lightly scaffolded Python files that mirror the target layout under `envs/`, `control/`, `data/`, `models/`, `train/`, `eval/`, `demos/`, `scripts/`, `configs/`, `tests/`, `assets/`.
- `pyproject.toml` including tool configuration (setuptools project metadata, black/isort/ruff placeholders for later lint integration) and pinned dependency minimums; fallback `requirements.txt` for quick installs.
- Updated `README.md` describing project scope, environment setup, and the multi-phase roadmap summary.
- `.gitignore` entries covering Python builds, virtual environments, Newton caches, dataset outputs (`runs/`, `episodes/`, etc.).

## Task Breakdown
1. **Repository Layout**
   - Create directories and nested packages; ensure each Python directory has an `__init__.py` exporting nothing yet (keeps imports predictable and enables namespace package extension later).
   - Insert concise module-level docstrings referencing TODO phases to keep future authors oriented.
2. **Configuration & Tooling**
   - Author `pyproject.toml`:
     - Project metadata (name `teamvla`, version `0.1.0`, description, authors placeholder).
     - Dependencies: `torch`, `numpy`, `opencv-python`, `gradio`, `pyyaml`, `tqdm`, `rich`, `scipy`, plus `typing-extensions` for compatibility. Document the optional Newton SDK requirement in README rather than listing (cannot be pip-installed).
     - Optional groups: `dev` (`pytest`, `mypy`, `ruff`, `black`) to support Phase 6 testing/tooling.
     - Tool sections for `ruff`, `mypy`, `pytest` with minimal defaults so future phases can incrementally adopt them.
   - Provide `requirements.txt` mirroring runtime deps with comments referencing `pip install -r requirements.txt`.
3. **Documentation**
   - Expand `README.md`:
     - Overview of TeamVLA, goals from architecture doc, high-level module map.
     - Setup instructions (Python version, recommended virtual environment, pip install commands, Newton engine install placeholder).
     - Roadmap table referencing `phase_*.md` documents for detailed guidance.
     - Placeholder sections for demos/results to be filled in later phases.
4. **Baseline Conventions**
   - Define coding standards section referencing type hints, docstrings, logging over prints, deterministic tests.
   - Document repository-level environment variables and configuration search order (e.g., `configs/common.yaml`).

## Testing Strategy
- Add Phase 0 import check test (under `tests/test_imports.py`) that verifies each top-level package imports successfully and modules expose docstrings. Keep functions tiny: e.g., one helper that walks directories, one assertion function.
- Include CI placeholder instructions in README for future GitHub Actions integration; no workflow files yet.

## Quality & Maintainability Notes
- Mandate that subsequent functions stay single-responsibility—call it out in README and future module docstrings.
- Ensure `pyproject.toml` enforces consistent formatting (e.g., `tool.black.line-length = 100`) to simplify later diffs.
- Preserve ASCII-only scaffolding unless a module already requires otherwise (per developer instructions).
- Avoid adding runtime code beyond docstrings/TODOs; Phase 0 is structural only.
