# TeamVLA Workflow Guide

This document captures the recommended end-to-end flow for exercising the repository once you have access to PyTorch and, optionally, the Newton SDK.

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install tooling / tests (PyTorch wheel depends on your platform)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pytest pytest-cov ruff black mypy
```

> Substitute the correct PyTorch wheel URL for your CUDA/CPU environment.

## 2. Collect Scripted Demonstrations

```bash
python -m scripts.collect_demos --task lift --episodes 20 --out data/episodes
python -m scripts.collect_demos --task handoff --episodes 20 --out data/episodes
python -m scripts.collect_demos --task drawer --episodes 20 --out data/episodes

# Optional: render quick-look videos
python -m scripts.render_videos --episodes data/episodes --out videos --fps 15
```

Dataset files land under `data/episodes/<task>/<task>_<episode_id>.npz`.  They conform to the schema in `data/schema.py` and can be indexed via `MultiTaskDataset`.

## 3. Train a Behaviour-Cloning Policy

```bash
python -m train.bc_trainer --config configs/train_bc.yaml
```

Key configuration knobs:

- `dataset.roots`: list of episode directories.
- `model.action_dim`: matches the 4-D action (dx, dy, dz, gripper).
- `epochs`, `optimizer`, `scheduler`: standard training hyper-parameters.

Checkpoints are saved with `train.bc_trainer.save_checkpoint`.  Update `load_policy` in `demos/app.py` to point at the produced checkpoint when ready.

## 4. Evaluate Policies

```bash
python -m eval.bench --tasks lift handoff drawer --episodes 5 --max-steps 200 --output results/summary.json
```

The benchmark CLI currently uses a placeholder zero-action policy.  After training, load your checkpoint inside `eval.bench.load_policy`.

The summary includes success rate, success-at-horizon, average time-to-success, mean coordination, and collision cost.

## 5. Launch the Demo

```bash
python -m demos.app
```

The interface runs scripted/environment rollouts and surfaces placeholder actions.  Once a trained policy is available, updating `load_policy` allows live playback via Gradio.

## 6. Test & Lint

```bash
pytest
ruff check .
black --check .
mypy .
```

PyTorch-dependent tests call `pytest.importorskip("torch")`, so they are skipped automatically if Torch is missing.  Running the full suite on a Torch-enabled machine is strongly recommended before publishing changes.

## 7. Continuous Integration (Recommended)

- Configure a GitHub Actions workflow that installs the dependencies above and runs the toolchain (`ruff`, `black --check`, `mypy`, `pytest`).
- Gate Newton-dependent tests behind `@pytest.mark.requires_newton`, keeping the default CI run CPU-only.

This workflow aligns with the phase roadmap and demonstrates an end-to-end path from scripted data collection to evaluation.
