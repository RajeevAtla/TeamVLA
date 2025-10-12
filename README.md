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

# Install runtime dependencies
uv pip install -r requirements.txt

# (Optional) install developer tooling
uv pip install -r requirements.txt -r <(printf 'pytest>=8.0\npytest-cov>=4.1\nmypy>=1.9\nruff>=0.4\nblack>=24.3\n')
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
  eval/                 # Metrics, rollouts, benchmarking (Phase 5+)
  demos/                # Gradio demo (Phase 5+)
  scripts/              # CLI helpers for data collection/rendering (Phase 5+)
  configs/              # YAML configuration stubs (Phase 3+)
  tests/                # Pytest suite (Phase 0+)
  assets/               # Placeholder for Newton assets (not tracked)
```

All Python packages export empty `__all__` lists today. Subsequent phases will populate modules with concrete implementations while preserving import stability.

## Development Workflow

1. Follow the phase roadmap in order; each step builds on prior scaffolding.
2. Keep functions short and single-purpose—prefer private helpers over long public methods.
3. Add or update tests alongside code. Every new feature must be covered by a granular unit test.
4. Use logging (`logging` module) for runtime diagnostics; avoid printing in library code.
5. Run the tooling stack regularly:
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

## Testing

Initial tests focus on import sanity and configuration validation. As phases progress, each module gains focused tests:

- Environment & tasks: deterministic reset/step behavior
- Control utilities: IK shape checks, phase transitions
- Data pipeline: schema validation, writer round-trip, dataloader indexing
- Models & training: forward passes, loss outputs, trainer wiring
- Evaluation & demo: metric calculations, CLI parsing, Gradio wiring

Mark Newton-dependent tests with `@pytest.mark.requires_newton` so they can be skipped when the physics engine is unavailable.

## License

MIT License. See `LICENSE` for details.
