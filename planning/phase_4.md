# Phase 4: Model Architectures and Training Loop

## Objectives
- Scaffold the vision-language-action (VLA) model implementations and shared training components, emphasizing modularity and testability.
- Ensure every function/method focuses on a single responsibility (e.g., encoding, fusion, loss computation).
- Integrate configuration-driven instantiation so training scripts can select models and hyperparameters without code changes.

## Deliverables
- `models/encoders/vision.py` and `models/encoders/language.py`:
  - Builder functions (`build_vision_encoder`, `build_text_encoder`) returning lightweight placeholder modules (e.g., simple CNN/Transformer stubs) until real backbones are integrated.
  - Helper functions (`forward_vision`, `forward_text`, `tokenize`) with docstrings and minimal placeholder behavior.
  - Private utilities for weight initialization to keep builders slim.
- `models/vla_singlebrain.py` and `models/vla_msgpassing.py`:
  - Classes implementing `forward`, `act`, and `loss` stubs with structured return dicts.
  - Composition of encoder builders via dependency injection (pass modules in `__init__` arguments derived from config).
  - Helper methods (`_fuse_modalities`, `_action_head`, `_message_passing_block`) to maintain small functions.
- `train/losses.py`, `train/schedulers.py`, `train/bc_trainer.py`:
  - Loss utility functions (Huber, BCE, sync loss, collision penalty) with placeholder arithmetic and TODO markers.
  - Scheduler helpers (cosine warmup) returning PyTorch scheduler instances or stub objects.
  - `bc_trainer` providing functions: `set_seed`, `build_model`, `build_data`, `train_one_epoch`, `evaluate`, `save_checkpoint`, `load_checkpoint`, `main`.
  - Logging via Python `logging` module, no prints.
- Update `README.md` to document model options and training entrypoint.

## Task Breakdown
1. **Encoder Modules**
   - Implement minimal placeholder networks using PyTorch (e.g., `nn.Sequential` with linear layers) to keep tests executable.
   - `tokenize` should accept list of strings and produce dictionary of tensors (`input_ids`, `attention_mask`) using a simple vocab stub.
   - Provide explicit shape comments in docstrings for clarity.
2. **Model Classes**
   - `SingleBrainVLA`:
     - Accept config dict (model type, hidden sizes, dropout).
     - Compose encoders, fusion module, and two action heads (one per agent).
     - `forward` returns dict containing `actions`, `logits`, `aux` placeholders.
     - `act` converts observations to model inputs; keep logic short by splitting into `_prepare_batch` helper.
     - `loss` delegates to `train.losses`.
   - `MsgPassingVLA`:
     - Extend baseline with message passing modules; stub to highlight TODO for future attention layers.
     - Provide `_exchange_messages` helper returning placeholder tensors.
3. **Training Utilities**
   - `losses.py`: ensure each function receives Torch tensors and returns scalar tensors; include parameter docs.
   - `schedulers.py`: implement `cosine_with_warmup` using PyTorch scheduler or stub with TODO; `make_optimizer` selecting AdamW/SGD based on config; keep helper functions small.
   - `bc_trainer.py`:
     - Modular functions with well-defined inputs/outputs.
     - `main` loads YAML config, constructs components, orchestrates training loop skeleton (no heavy work yet).
     - Introduce `TrainingState` dataclass capturing epoch, best metrics, etc.
     - Use logging to report progress; no prints.

## Testing Strategy
- `tests/test_encoders.py`:
  - Instantiate encoder builders with minimal configs; verify output tensor shapes.
  - Test tokenizer handles varied text lengths; ensure deterministic results for given vocab.
- `tests/test_models.py`:
  - Check `SingleBrainVLA` and `MsgPassingVLA` forward pass shapes using synthetic batches.
  - Confirm `act` returns list of numpy arrays with correct lengths.
  - Validate `loss` returns dict of floats/tensors with required keys.
- `tests/test_losses.py`:
  - Ensure each loss function produces expected scalar values on small synthetic inputs.
- `tests/test_bc_trainer.py`:
  - Mock dataset/env to verify build/train/evaluate functions wire together without running heavy compute.
  - Confirm `set_seed` sets NumPy, random, and Torch seeds.
- Keep tests minimal and focused; one behavior per test function.

## Quality & Maintainability Notes
- Follow PyTorch best practices: register buffers where needed, avoid global state, keep modules simple.
- Document expected config keys within each module’s docstrings to reduce duplication.
- Encapsulate repeated tensor operations in helper functions to respect the “smallest possible function” guideline.
- Ensure checkpoint paths and run directories are configurable, defaulting to `runs/<timestamp>` but overridable via config.
