# Project: **TeamVLA – Two-Arm Newton-Guided Operations**

*(Multi-agent, multi-task VLA benchmark and baseline trainer on Newton)*

## 0) High-level goals

* Two robot arms in **one Newton scene** solving three cooperative tasks: **lift-and-place**, **hand-off**, **bimanual drawer**.
* Language-conditioned, vision+text → **joint actions (Δq) + gripper** per agent.
* Unified dataset schema, multi-task BC trainer, evaluation, and web demo.

---

## 1) Repository layout

```
TeamVLA/
  envs/
    core_env.py
    tasks/
      base.py
      lift.py
      handoff.py
      drawer.py
  control/
    ik_utils.py
    phase_machine.py
    scripted/
      lift_demo.py
      handoff_demo.py
      drawer_demo.py
  data/
    schema.py
    writer.py
    dataloader.py
  models/
    encoders/
      vision.py
      language.py
    vla_singlebrain.py
    vla_msgpassing.py
  train/
    bc_trainer.py
    losses.py
    schedulers.py
  eval/
    metrics.py
    bench.py
    rollouts.py
  demos/
    app.py
  scripts/
    collect_demos.py
    render_videos.py
  configs/
    common.yaml
    lift.yaml
    handoff.yaml
    drawer.yaml
    train_bc.yaml
  tests/
    test_env_core.py
    test_ik_phase.py
    test_writer_loader.py
  assets/
    README.md  # (placeholder for USD/URDF assets, not tracked)
  README.md
  pyproject.toml  # or setup.cfg/requirements.txt
```

---

## 2) envs/core_env.py

**Purpose:** Core multi-agent Newton environment with a Gym-like API and task plugin hooks.

### Classes & Functions

* `class NewtonMAEnv:`

  * `__init__(self, cfg: dict) -> None`
    Set up Newton model, two arm articulations, cameras, solver, collision filters, seed RNG, and load a `TaskSpec` from `TASK_REGISTRY`.
  * `reset(self, instruction: str) -> list[dict]`
    Calls `task.reset(...)`, randomizes scene, returns per-agent observations `[obs_A, obs_B]`.
  * `step(self, actions: list[np.ndarray]) -> tuple[list[dict], list[float], bool, dict]`
    Apply per-agent actions, advance physics **once**, compute per-agent rewards via `task.reward`, termination via `task.success`.
  * `render(self, mode: str="rgb_array") -> dict`
    Return a dict of current rendered frames per camera/agent.
  * `close(self) -> None`
    Dispose viewers and state.
  * `set_task(self, name: str) -> None`
    Swap in a new `TaskSpec`.
  * `get_state_dict(self) -> dict`
    Lightweight snapshot for reproducibility (seed, task name, randomization params).

* `def build_multi_robot_world(cfg: dict) -> tuple[Model, State]`
  Create Newton `Model`, add two arm articulations, table, lights; return `(model, state)`.

* `def setup_cameras(model: Model, cfg: dict) -> dict`
  Mount per-agent cameras (eye-in-hand + overhead).

* `def make_solver(cfg: dict) -> Any`
  Instantiate Newton solver backend (e.g., MuJoCo/Featherstone) with substeps.

* `def obs_i(state, cams_i, instruction) -> dict`
  Build per-agent observation: `rgb`, optional `depth`, `q`, `dq`, `gripper`, `peer_summary`, `instruction`.

* `TASK_REGISTRY: dict[str, TaskSpec]`
  Populated by `tasks/*.py`.

---

## 3) envs/tasks/base.py

**Purpose:** Define the task plugin interface.

### Classes & Protocols

* `class TaskSpec(Protocol):`

  * `name: str`
  * `def build_scene(self, model: Model) -> None` — add shelves/drawers/rendezvous frames.
  * `def phases(self) -> list[str]` — ordered list like `["reach","align","grasp","move","place","release"]`.
  * `def reset(self, state: State, rng: np.random.RandomState) -> dict` — place objects, set targets, return task metadata.
  * `def reward(self, state: State, phase: str) -> tuple[float,float]` — per-agent reward shaping.
  * `def success(self, state: State) -> bool` — terminal success condition.
  * `def scripted_action(self, obs: dict, phase: str, agent_id: int) -> np.ndarray` — IK/waypoint policy for data collection.
  * `def info(self, state: State) -> dict` — debugging signals (distances, contacts, joint positions).

* `def register_task(task: TaskSpec) -> None`
  Adds to `TASK_REGISTRY`.

---

## 4) envs/tasks/lift.py, handoff.py, drawer.py

**Purpose:** Implement each cooperative task using `TaskSpec`.

### Shared Function Signatures (each file)

* `class LiftTask(TaskSpec)` / `HandOffTask(TaskSpec)` / `DrawerTask(TaskSpec)`

  * `build_scene(self, model)` — add shelf/bar; add rendezvous frame; add cabinet + prismatic drawer.
  * `phases(self)` — phase graph per task (e.g., hand-off adds `handover`).
  * `reset(self, state, rng)` — randomize object lengths/weights/poses.
  * `reward(self, state, phase)` — shaped rewards (alignment, COM height, drawer open distance).
  * `success(self, state)` — goal AABB or drawer threshold with stability K frames.
  * `scripted_action(self, obs, phase, agent_id)` — IK waypoints; synchronized grasping.
  * `info(self, state)` — metrics (alignment error, collision impulse, contact slips).

---

## 5) control/ik_utils.py

**Purpose:** Shared control utilities.

### Functions

* `solve_ik(q_init: np.ndarray, target_pose: np.ndarray, limits: dict, max_iters: int=100) -> np.ndarray`
  Iterative IK, returns target joint config.
* `ee_pose_from_state(state, agent_id: int) -> np.ndarray`
  Extract end-effector 6D pose.
* `plan_rendezvous(eeA: np.ndarray, eeB: np.ndarray, obj_pose: np.ndarray) -> tuple[np.ndarray,np.ndarray]`
  Compute mirrored grasp poses and meeting point.
* `gripper_command(open_ratio: float) -> float`
  Normalize to actuator range.
* `clamp_action(delta_q: np.ndarray, max_norm: float) -> np.ndarray`
  Safety clamp for joint deltas.

---

## 6) control/phase_machine.py

**Purpose:** Generic finite-state machine for phases.

### Classes & Functions

* `class PhaseMachine:`

  * `__init__(self, phases: list[str], timeouts: dict[str,int])`
  * `reset(self) -> None`
  * `step(self, signals: dict) -> str` — advance phase based on conditions (distance < ε, contact acquired, etc.).
  * `current(self) -> str`
  * `is_terminal(self) -> bool`

* `def phase_signals_from_state(state, task_info: dict) -> dict`
  Build signals used to progress phases.

---

## 7) control/scripted/*.py

**Purpose:** Scripted demo policies (data collection).

### Functions (each task file)

* `def scripted_policy(env: NewtonMAEnv, phase_machine: PhaseMachine, obs: list[dict]) -> list[np.ndarray]`
  Returns `[action_A, action_B]` per step using IK waypoints + synchronized grasp/hand-off logic.
* `def waypoint_library(cfg: dict) -> dict`
  Predefined poses for reach/rendezvous/place per task.

---

## 8) data/schema.py

**Purpose:** Canonical episode/step schema.

### Constants & Helpers

* `STEP_FIELDS = {...}`
  Keys: `rgb_a`, `rgb_b`, `q_a`, `dq_a`, `q_b`, `dq_b`, `action_a`, `action_b`, `grip_a`, `grip_b`, `ee_a`, `ee_b`, `obj_pose`, `contacts`, `text_global`, `text_a`, `text_b`, `role_a`, `role_b`, `phase`, `task`, `success`.
* `def validate_step(step: dict) -> None`
  Assert shapes/dtypes.

---

## 9) data/writer.py

**Purpose:** Efficient on-disk episode writer.

### Classes

* `class EpisodeWriter:`

  * `__init__(self, out_dir: str, fmt: str="npz", compress: bool=True)`
  * `start_episode(self, meta: dict) -> None`
  * `add_step(self, step: dict) -> None`
  * `end_episode(self, success: bool) -> str` — write to disk, return path.
  * `close(self) -> None`

---

## 10) data/dataloader.py

**Purpose:** Multi-task dataset iterator for training.

### Classes & Functions

* `class MultiTaskDataset(torch.utils.data.Dataset):`

  * `__init__(self, roots: list[str], transforms=None, tasks: list[str]|None=None)`
  * `__len__(self) -> int`
  * `__getitem__(self, idx: int) -> dict` — returns fused sample with tensors.
* `def make_dataloader(cfg: dict) -> torch.utils.data.DataLoader`

---

## 11) models/encoders/vision.py

**Purpose:** Vision backbone factory.

### Functions

* `def build_vision_encoder(name: str="vit_b16", pretrained: bool=True, out_dim: int=768) -> torch.nn.Module`
  Returns encoder producing token sequence or pooled feature.
* `def forward_vision(encoder, images: torch.Tensor) -> torch.Tensor`

## 12) models/encoders/language.py

**Purpose:** Text encoder factory.

### Functions

* `def build_text_encoder(name: str="transformer_small", vocab_size: int=30522, d_model: int=512, n_layers: int=6) -> torch.nn.Module`
* `def tokenize(texts: list[str], vocab: Any) -> dict[str,torch.Tensor]`
* `def forward_text(encoder, tokens: dict) -> torch.Tensor`

---

## 13) models/vla_singlebrain.py

**Purpose:** Single-brain, multi-head VLA baseline.

### Classes

* `class SingleBrainVLA(torch.nn.Module):`

  * `__init__(self, cfg: dict)` — build vision/text encoders; fusion via cross-attention; heads per agent.
  * `forward(self, batch: dict) -> dict`
    Inputs: `rgb_a`, `rgb_b`, `text`, optional `proprio`, `role_tokens`, `task_tokens`, `phase_tokens`.
    Outputs: `pred_a(Δq, grip)`, `pred_b(Δq, grip)`, optional attention maps.
  * `act(self, obs: list[dict]) -> list[np.ndarray]` — inference wrapper.
  * `loss(self, batch: dict) -> dict[str, torch.Tensor]`
    Huber on Δq, BCE on grip, optional coordination loss (see `train/losses.py`).

---

## 14) models/vla_msgpassing.py

**Purpose:** Message-passing multi-agent VLA.

### Classes

* `class MsgPassingVLA(torch.nn.Module):`

  * `__init__(self, cfg: dict)` — add learnable 32-d message per agent; cross-attend to peer message.
  * `forward(self, batch: dict) -> dict` — same IO as `SingleBrainVLA` with `msg_a`, `msg_b`.
  * `act(self, obs)` — runtime message exchange.
  * `loss(self, batch)` — includes `λ_sync * sync_loss`.

---

## 15) train/losses.py

**Purpose:** Common training losses.

### Functions

* `def huber_action_loss(pred: torch.Tensor, target: torch.Tensor, delta: float=1.0) -> torch.Tensor`
* `def grip_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor`
* `def sync_loss(ee_a: torch.Tensor, ee_b: torch.Tensor, mode: str, phase: torch.Tensor) -> torch.Tensor`
  Penalize misalignment during co-grasp/handover phases.
* `def collision_penalty(collisions: torch.Tensor, alpha: float) -> torch.Tensor`

---

## 16) train/schedulers.py

**Purpose:** LR schedulers & warmups.

### Functions

* `def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int)`
* `def make_optimizer(model, cfg: dict) -> torch.optim.Optimizer`

---

## 17) train/bc_trainer.py

**Purpose:** End-to-end Behavior Cloning trainer.

### Functions

* `def set_seed(seed: int) -> None`
* `def build_model(cfg: dict) -> torch.nn.Module` — choose single-brain or msg-passing.
* `def build_data(cfg: dict) -> DataLoader` — multi-task.
* `def train_one_epoch(model, loader, optim, sched, cfg) -> dict`
* `def evaluate(model, env, tasks: list[str], cfg: dict) -> dict` — runs rollouts via `eval/rollouts.py`.
* `def save_checkpoint(path: str, model, optim, sched, step: int, cfg: dict) -> None`
* `def load_checkpoint(path: str, model, optim=None, sched=None) -> int`
* `def main(cfg_path: str="configs/train_bc.yaml") -> None`

CLI:

```
python -m train.bc_trainer --config configs/train_bc.yaml
```

---

## 18) eval/metrics.py

**Purpose:** Task and coordination metrics.

### Functions

* `def success_at_T(trajs: list[dict], horizon: int) -> float`
* `def time_to_success(traj: dict) -> float|None`
* `def coordination_score(traj: dict, epsilon: float) -> float` — % steps both grippers satisfy role constraints.
* `def collision_cost(traj: dict) -> float` — sum of collision impulses.
* `def aggregate_results(results: list[dict]) -> dict`

## 19) eval/rollouts.py

**Purpose:** Run policy rollouts in env.

### Functions

* `def run_episode(env: NewtonMAEnv, policy, instruction: str, max_steps: int) -> dict`
* `def run_suite(env: NewtonMAEnv, policy, tasks: list[str], n_eps: int, unseen: bool) -> list[dict]`

## 20) eval/bench.py

**Purpose:** Benchmark runner (one command).

### Functions

* `def main(args: argparse.Namespace) -> None`
  Loads checkpoints, runs `run_suite`, writes CSV/Markdown table of metrics.

CLI:

```
python -m eval.bench --tasks lift handoff drawer --episodes 200 --unseen
```

---

## 21) demos/app.py

**Purpose:** Gradio/WebRTC app to type an instruction and watch both arms act.

### Functions

* `def load_policy(checkpoint_path: str)`
* `def inference_step(img_a, img_b, text, state) -> (frames, state)`
* `def main()` — launch Gradio with two video panes + text box.

---

## 22) scripts/collect_demos.py

**Purpose:** Vectorized scripted data collection (multi-task).

### Functions

* `def collect(env: NewtonMAEnv, task_names: list[str], episodes_per_task: int, out_root: str, seed: int)`
  Interleave tasks round-robin; write episodes via `EpisodeWriter`.
* `def main()` — parse `--tasks`, `--episodes`, `--out`, `--seed`.

## 23) scripts/render_videos.py

**Purpose:** Convert episodes to MP4 snippets for README.

### Functions

* `def render_episode(ep_path: str, out_path: str, fps: int=20)`
* `def main()` — glob episodes, render all.

---

## 24) configs/*.yaml

**Purpose:** Centralize hyperparams & randomization.

* `common.yaml` — seeds, device, logging, viewer.
* `lift.yaml` / `handoff.yaml` / `drawer.yaml` — assets, thresholds, randomization ranges.
* `train_bc.yaml` — model type, encoders, batch size, LR, augments, loss weights (e.g., `lambda_sync`, `lambda_collision`), datasets.

---

## 25) tests/*

**Purpose:** Fast headless unit tests.

* `test_env_core.py`

  * Ensures `reset/step` return valid shapes; single step is deterministic with fixed seed.
* `test_ik_phase.py`

  * IK reaches pose within ε; phase machine progresses on signals.
* `test_writer_loader.py`

  * Round-trip write/read; schema validation passes.

---

## 26) README.md (Codex should write this)

* Project summary, install, quick start (run an example env; collect demos; train; benchmark), results table, demo GIFs, config explanations.

---

## 27) Implementation notes for Codex

* Use **type hints** and **docstrings** for every public function.
* Keep **no external internet calls**; everything local.
* Depend on `torch`, `numpy`, `opencv-python`, `gradio` (for demo).
* Wrap Newton imports behind a guard like `try: import newton ... except ImportError: raise ...` with setup instructions in README.
* Ensure **device-agnostic** (CPU ok for unit tests; GPU preferred for training).
* Log with `logging` (not prints).
* Save checkpoints to `runs/<timestamp>/` with config snapshot.

---
