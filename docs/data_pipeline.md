TeamVLA Data Pipeline
=====================

This document captures the scope that Track B owns: schemas, serialization,
dataloading, and training artefacts.  It is designed to be read alongside the
code so that other tracks (simulation and evaluation) understand what the data
stack guarantees.

Overview
--------

The data pipeline is responsible for turning scripted trajectories produced by
Track A into reproducible datasets, and for feeding those datasets into the
behaviour-cloning trainers implemented in `train/`.  The main components are:

- **Schema definitions** (`data/schema.py`): typed structures, validation
  helpers, and dtype utilities.
- **Episode writer** (`data/writer.py`): streaming helper that validates every
  step and persists `.npz` bundles.
- **Dataset/dataloader** (`data/dataloader.py`): a manifest-driven iterable that
  surfaces episode steps with optional transforms and deterministic indexing.
- **Configurations** (`configs/*.yaml`): YAML files wiring the pipeline together
  for the default lift/handoff/drawer tasks.
- **Model and trainer entrypoints** (`models/`, `train/`): encoders, policy
  heads, and the behaviour-cloning trainer that consumes the dataloader.

Schema Summary
--------------

### Step Structure

`StepData` is represented as a plain mapping validated via `FieldSpec`.  Every
step must include:

- `rgb_a`, `rgb_b` (`uint8[H×W×3]`): per-arm RGB observations.
- `q_a`, `q_b` (`float32[N]`): joint configurations.
- `action_a`, `action_b` (`float32[N]`): previous action targets (for teacher
  forcing).
- `grip_a`, `grip_b` (`float32`): normalised gripper commands.
- `instruction` (`str`): natural-language prompt supplied to the episode.
- `task` (`str`): canonical task identifier (`lift`, `handoff`, `drawer`, …).
- `success` (`bool`): step-level success flag.

Optional fields (depth maps, phase labels, timestamps) are described in
`OPTIONAL_FIELDS`.  Any unrecognised fields pass through untouched as long as
they are not NumPy scalars, enabling future extensions without breaking older
code.

### Episode Metadata

`EpisodeMeta` captures high-level information:

- `task`: task slug.
- `episode_id`: unique identifier per episode.
- `success`: overall success flag.
- `num_steps`: derived automatically during finalisation.
- `version`: file-format version (`EPISODE_FILE_VERSION`).

`validate_episode_meta` coerces metadata to the dataclass and raises
`SchemaError` on invalid payloads.

Episode Writer
--------------

`EpisodeWriter` provides a context-manager interface:

```python
from data.writer import EpisodeWriter

with EpisodeWriter(\"data/episodes\", fmt=\"npz\") as writer:
    meta = writer.start_episode({\"task\": \"lift\", \"success\": True})
    writer.add_step(step_dict)
    path = writer.end_episode(success=True)
```

- Deterministic episode identifiers can be auto-generated or supplied.
- Steps are validated on insertion.
- Episodes are stored as `.npz` files with object arrays for metadata and step
  payloads; compression is enabled by default.
- `EpisodeWriter` can be reused across episodes and reset via `.close()`.

A synthetic fixture is checked in at
`tests/fixtures/data/lift_sample.npz` for downstream tests and documentation.

Dataset & DataLoader
--------------------

`MultiTaskDataset` reads one or more episode roots, scans for `.npz` files,
validates metadata, and builds a step-level index.  Key features:

- **Task filtering**: restrict to selected task slugs.
- **Deterministic indexing**: stable ordering regardless of filesystem state.
- **Transform chaining**: callables applied sequentially to every step.
- **Task limits**: cap the number of episodes per task to balance datasets.

To obtain a PyTorch-style dataloader, use `make_dataloader` with the YAML config
payload:

```python
from data.dataloader import make_dataloader
from yaml import safe_load

cfg = safe_load(Path(\"configs/train_bc.yaml\").read_text())
dataloader = make_dataloader(cfg[\"dataset\"])
```

If PyTorch is not installed the helper raises `ImportError`, matching the
behaviour of the model and trainer modules.  Unit tests that depend on torch are
therefore marked with `pytest.importorskip(\"torch\")`.

Configurations
--------------

`configs/` supplies a layered configuration scheme:

- `common.yaml` holds session-wide defaults (seed, logging verbosity, dataset
  root).
- Task-specific YAML (`lift.yaml`, `handoff.yaml`, `drawer.yaml`) describe the
  scripted policy assets and reward-weight placeholders Track A can refine.
- `train_bc.yaml` combines dataset, model, optimiser, and scheduler settings for
  behaviour cloning.

Every YAML file now carries comments describing the intent of each field.  The
configs are documented in `configs/README.md`, including override guidance.

Models & Trainer
----------------

Track B provides lightweight encoders (`models/encoders/`) and two policy
variants:

- `SingleBrainVLA`: a shared fusion tower for both arms.
- `MsgPassingVLA`: extends the single-brain architecture with simple message
  passing layers.

The trainer (`train/bc_trainer.py`) exposes modular builders (`build_model`,
`build_data`) alongside the main training loop.  Checkpoints store model,
optimiser, scheduler, training state, and configuration.  When PyTorch is not
available the save/load helpers transparently fall back to Python `pickle`,
enabling fixtures on systems without GPU tooling.

Compute & GPU Policy
--------------------

- CPU is the default target to keep the bootstrap path simple.
- To enable GPUs, set `device: cuda` (or `cuda:0`, etc.) in `configs/common.yaml`
  or provide `--device cuda` overrides.  The trainer automatically forwards the
  device to the instantiated model.
- Install PyTorch via `uv pip install --index-url https://download.pytorch.org/whl/cpu torch`
  (choose the appropriate wheel for your platform).  We verified that `uv run
  pytest` covers the schema/writer/dataloader suite under CPU-only conditions
  and documented the pickle fallback when torch is absent.
- Track C can extend this policy by introducing GPU-only evaluation configs; the
  pipeline remains agnostic as long as configs surface the `device` knob.

Checkpoint Registry
-------------------

- Synthetic fixtures live under `tests/fixtures/checkpoints/`.
- For real training outputs, store checkpoints in `checkpoints/<task>/` using the
  pattern `epoch_{epoch:04d}.ckpt` (compressed via `save_checkpoint`).  This
  convention is documented for Track C so their loaders can glob predictable
  filenames.
- Every checkpoint written through `save_checkpoint` contains the YAML config
  snapshot, ensuring reproducibility when the registry grows.

Future Baselines
----------------

With behaviour cloning in place, the next candidates are:

1. **DAgger / Online Imitation** – extend the trainer with data aggregation
   hooks that request corrective demonstrations from scripted policies.
2. **SAC (Soft Actor-Critic)** – reuse the existing encoders as perception
   modules and introduce actor/critic heads; leverage the dataloader for replay
   buffer seeding.
3. **BC-RL Hybrid** – fine-tune the behaviour-cloning policy with a KL-regularised
   policy-gradient stage (e.g., AWAC).

These options are scoped so that Track B can iterate once Track A finalises the
scripted data and Track C publishes evaluation harnesses.  Each proposal should
reuse the existing config system (add `train_sac.yaml`, etc.) and document the
changes in this file when the implementation lands.

Fixtures & Hand-off Artefacts
-----------------------------

Track B delivers the following artefacts for downstream tracks:

- `tests/fixtures/data/lift_sample.npz`: minimal valid episode for schema and
  dataloader tests.
- `tests/fixtures/checkpoints/bc_stub.ckpt`: synthetic checkpoint saved through
  `save_checkpoint`, loadable with or without PyTorch installed.
- Schema documentation (this file) to guide evaluation tooling authored by
  Track C.

For Track C benchmarking, `load_checkpoint` automatically falls back to the
pickle representation if `torch.load` fails, ensuring the synthetic fixture can
bootstrap tests even before full torch installations are available.

Testing Matrix
--------------

Relevant pytest modules:

- `tests/test_data_schema.py`: field validations and dtype helpers.
- `tests/test_episode_writer.py`: episode lifecycle and serialization.
- `tests/test_dataloader.py`: manifest indexing and transform flow.
- `tests/test_configs.py`: YAML sanity checks and merging logic.
- `tests/test_losses.py`, `tests/test_models.py`, `tests/test_bc_trainer.py`:
  model/training behaviour (skip if torch missing).

Run the full Track B suite after edits:

```
uv run pytest tests/test_data_schema.py tests/test_episode_writer.py tests/test_dataloader.py
uv run pytest tests/test_models.py  # requires torch
```

Compatibility Checklists
------------------------

- Confirm new episode fields are added to `OPTIONAL_FIELDS` with clear
  descriptions.
- When introducing new configs, update `configs/README.md` and include default
  values.
- For new checkpoints, always use `save_checkpoint` to preserve structure across
  tracks.

These guardrails keep Tracks A and C aligned with Track B’s contract while the
codebase evolves.
