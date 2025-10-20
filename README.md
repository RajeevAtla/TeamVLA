# TeamVLA – Two-Arm Newton-Guided Operations

TeamVLA is a benchmark and reference implementation for coordinating two robot arms in a shared Newton physics scene. The system targets three cooperative manipulation tasks—lift-and-place, hand-off, and bimanual drawer—conditioned on multimodal (vision + language) inputs and producing per-arm joint deltas plus gripper commands.

> The repository is scaffolded in phases. Each phase document lives under `planning/` and captures objectives, deliverables, task breakdowns, and testing strategy.

## Getting Started

- **Python**: 3.10 or newer (managed by `uv`).  
- **Newton SDK**: Not bundled; follow vendor instructions and ensure headers/binaries are on your library path once you reach the implementation phases.

```bash
# Create an isolated environment (uv-managed)
uv venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

# Install runtime dependencies from pyproject.toml / uv.lock
uv sync

# (Optional) install developer tooling (adds [dependency-groups.dev])
uv sync --dev
```

> We default to `uv` for package management; feel free to translate the commands to your preferred tooling if necessary.

## Repository Structure

```
TeamVLA/
  planning/             # Phase docs, architecture notes, prompt
  envs/                 # Newton environment and task registry (Phase 1+)
  control/              # IK utilities, phase machine, scripted controllers (Phase 2+)
  data/                 # Schema, writer, dataloader (Phase 3+)
  models/               # Vision/language encoders and VLA policies (Phase 4+)
  train/                # Losses, schedulers, BC trainer (Phase 4+)
  eval/                 # Metrics, rollouts, benchmarking utilities
  demos/                # Gradio demo entry-point
  scripts/              # CLI helpers for data collection/rendering
  configs/              # YAML configuration stubs (Phase 3+)
  tests/                # Pytest suite (Phase 0+)
  assets/               # Placeholder for Newton assets (not tracked)
```

Most packages now expose concrete functionality—encoders/models, scripted control, data pipelines, evaluation.  The architecture still follows the phase roadmap, so extending a module rarely requires reshaping its neighbours.

## Development Workflow

1. Follow the phase roadmap in order; each step builds on prior scaffolding.
2. Keep functions short and single-purpose—prefer private helpers over long public methods.
3. Add or update tests alongside code. Every new feature must be covered by a granular unit test.
4. Use logging (`logging` module) for runtime diagnostics; avoid printing in library code.
5. Run the tooling stack regularly (install `torch` and `pytest` locally for the full test suite):
   - `uv run pytest`
   - `uv run ruff check .`
   - `uv run black .`
   - `uv run mypy .`

Refer to `docs/contributing.md` for the full contributor checklist and tooling expectations.

## Phase Roadmap

| Phase | Description | Reference |
|-------|-------------|-----------|
| 0 | Bootstrap structure, metadata, documentation, baseline tests | `planning/phase_0.md` |
| 1 | Environment core + task registry scaffolding | `planning/phase_1.md` |
| 2 | IK utilities, phase machine, scripted demos | `planning/phase_2.md` |
| 3 | Data schema, writer, dataloader, configs | `planning/phase_3.md` |
| 4 | Models, encoders, training loop | `planning/phase_4.md` |
| 5 | Evaluation suite, benchmark CLI, Gradio demo | `planning/phase_5.md` |
| 6 | Comprehensive testing, linting, CI guidance | `planning/phase_6.md` |

Consult the architecture overview (`planning/architecture.md`) and the prompt scaffold (`planning/prompt.md`) for fine-grained interface requirements.

## Data Collection & Training

- **Environment & Control Interfaces**: See `docs/interfaces/env_control.md` for a field-by-field breakdown of observations, action vectors, and scripted policy outputs owned by the simulation/control track.

- Generate scripted demonstrations:

  ```bash
  python -m scripts.collect_demos --task lift --episodes 10 --out data/episodes
  ```

- Inspect or convert collected episodes to videos (falls back to `.npy` if `imageio` is unavailable):

  ```bash
  python -m scripts.render_videos --episodes data/episodes --out videos
  ```

- Launch a behaviour-cloning run (requires `torch`):

  ```bash
  python -m train.bc_trainer --config configs/train_bc.yaml
  ```

- Evaluate a policy (currently uses a placeholder zero-action policy until checkpoints are produced):

  ```bash
  python -m eval.bench --tasks lift handoff drawer --episodes 2 --output results/summary.json
  ```

## Demo

Launch the Gradio-based interactive demo:

```bash
python -m demos.app
```

It resets the Newton environment per request and reports the placeholder actions returned by the policy shim.  Once checkpoints are available, `demos.app.load_policy` can be extended to load them.

See `docs/workflow.md` for a detailed walkthrough that ties together setup, scripted data generation, training, evaluation, and demo usage.

## Testing

The unit suite now covers control, data, models, evaluation, scripts, and an end-to-end smoke test linking the pipeline.  Install `torch` locally to run the full suite:

```bash
pytest
```

Tests that rely on optional dependencies fall back gracefully if the packages are missing.  When authoring new tests, continue to mark Newton-dependent cases with `@pytest.mark.requires_newton`.

## License

MIT License. See `LICENSE` for details.
