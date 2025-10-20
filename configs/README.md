TeamVLA Configuration Guide
===========================

The YAML files in this directory act as the declarative interface for the data
and training pipeline implemented by Track B.  They are designed to be composed:
`common.yaml` supplies defaults, task YAMLs specialise them, and
`train_bc.yaml` wires everything together for behaviour cloning.

File Overview
-------------

- **`common.yaml`** – Session defaults: random seeds, logging verbosity, viewer
  options, and default dataset root.
- **`lift.yaml` / `handoff.yaml` / `drawer.yaml`** – Task-specific knobs for the
  scripted data generators (randomisation ranges, reward weights, controller
  assets).  Track A updates these as the environments mature.
- **`train_bc.yaml`** – End-to-end behaviour-cloning recipe covering dataset
  roots, dataloader arguments, model configuration, optimiser, scheduler, and
  loss weights.

Common Fields
-------------

| Key                     | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `seed`                  | Integer used by `train.bc_trainer.set_seed` for deterministic runs.        |
| `dataset.roots`         | List of directories scanned by `MultiTaskDataset`.                         |
| `dataset.tasks`         | Optional list of task slugs to filter.                                     |
| `dataset.limit_per_task`| Cap on episodes per task (null means unlimited).                           |
| `dataset.batch_size`    | Batch size fed into the PyTorch `DataLoader`.                              |
| `model.type`            | Policy architecture (`single_brain` or `msg_passing`).                     |
| `model.action_dim`      | Action dimensionality per arm.                                             |
| `optimizer.name`        | Optimiser factory registered in `train.schedulers`.                        |
| `scheduler.warmup_steps`| Number of warm-up steps for cosine schedule.                               |
| `loss_weights.*`        | Scalars applied to component losses (`actions`, `grip`, etc.).             |

Extending Configurations
------------------------

1. Add new keys with descriptive inline comments so the intent is obvious.
2. Mirror any new sections in the `configs/` README table above.
3. Update `docs/data_pipeline.md` if the change affects inter-track contracts.
4. Keep naming consistent (`snake_case` keys, task slugs matching Track A).

Usage
-----

```bash
# Launch behaviour cloning with the default recipe
python -m train.bc_trainer --config configs/train_bc.yaml

# Override values from the CLI (uses yaml.safe_load internally)
python -m train.bc_trainer --config configs/train_bc.yaml --override dataset.batch_size=16
```

Config parsing is performed with `PyYAML` (`yaml.safe_load`), so anchors and
environment-variable interpolation are available if the project needs them
later.
