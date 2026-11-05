# Training with start/goal context

`train_conditional.py` follows `train.py` and reuses the existing trajectory
dataset, temporal U-Net, Gaussian diffusion loss, trainer, EMA, and summaries.
`launch_train_conditional_01.py` follows `launch_train_01.py`, with a separate
experiment name. No original files need to be changed.

## Conditioning

The dataset already supplies `task_normalized`: the concatenated start and goal
**positions**, normalized with the dataset's task normalizer. For a planar robot
this is `[start_x, start_y, goal_x, goal_y]`. Its dimension is inferred from the
dataset, independently of whether trajectory states include velocity.

The existing `TaskModel` passes this vector through unchanged. With
`conditioning_type='default'`, the U-Net concatenates it with the diffusion-step
embedding and injects the combined vector into every residual block. The existing
start/goal hard conditions, including zero endpoint velocities when enabled, are
retained. The script sets `dataset.variable_environment = False` on the loaded
dataset instance so the existing context builder works for both loss computation
and trajectory summaries.

Training starts with fresh weights unless `resume_checkpoint` is supplied. The U-Net backbone and diffusion process are
the same as the original; the residual conditioning projections have additional
input columns for the task vector. Unconditioned state dictionaries therefore do
not load directly into this model.

## Run

Use the original MMD environment and installation instructions, with this checkout
installed as the `mmd` package. Datasets use the original directory layout under
`mmd/data_trajectories/`; `--dataset_subdir` also accepts an absolute dataset path.

From the `mmd/` repository root:

```bash
conda activate mmd
cd scripts/train_diffusion
python train_conditional.py \
    --dataset_subdir EnvEmptyNoWait2D-RobotPlanarDisk \
    --include_velocity True \
    --conditioning_type default \
    --unet_dim_mults_option 0 \
    --batch_size 128 \
    --lr 0.0003 \
    --num_train_steps 500000 \
    --steps_til_summary 20000 \
    --steps_til_ckpt 20000 \
    --wandb_mode disabled \
    --debug False \
    --seed 0 \
    --results_dir logs_conditional
```

These settings match the original launcher's training configuration. For the
original local/Slurm experiment workflow, edit the dataset list and resource
settings in the new launcher and run:

```bash
python launch_train_conditional_01.py
```

The launcher uses the experiment name `train_diffusion_conditional` and disables
WandB by default. Enable it and set your entity if desired. The direct script
retains the original training script's model/optimizer defaults; the launcher
retains the original launcher's overrides.

## Checkpoints and inference

To continue training from saved weights, set `resume_checkpoint` in
`launch_train_conditional_01.py` to the absolute path of the previous run's
`model_current_state_dict.pth`, then run that launcher again. Its timestamp creates
a separate output directory. Leave `resume_checkpoint = ''` for fresh training.

For direct execution, add:

```bash
--resume_checkpoint /path/to/previous/run/0/checkpoints/model_current_state_dict.pth \
--results_dir logs_conditional_continued
```

Use the same dataset, normalization, conditioning mode, U-Net dimensions, and
diffusion configuration as the source run. Loading is strict, so incompatible
state-dictionary keys or tensor shapes fail. This restores model weights only:
the optimizer, step counter, and EMA history restart (EMA initializes from the
loaded model). `num_train_steps` specifies the number of additional steps.
The shared hard-conditioning function retains its current behavior.

The direct command writes arguments and split indices under
`logs_conditional/0/`, and the original checkpoint formats under
`logs_conditional/0/checkpoints/`, including `model_current_state_dict.pth`,
`ema_model_current_state_dict.pth`, and full-model `.pth` files. Use a fresh
results directory for each run to keep earlier weights.

For state-dictionary loading, reconstruct the U-Net with the saved training
arguments plus `conditioning_embed_dim=dataset.fields[dataset.field_key_task].shape[-1]`,
and give the diffusion wrapper a `TaskModel` with that same input/output dimension.
Full-model checkpoints already contain the conditioning architecture.

For an individual task, supply the same normalized context during inference:

```python
sample = dataset[trajectory_id]
context = {'tasks': sample[f'{dataset.field_key_task}_normalized']}
trajs_normalized = model.run_inference(
    context=context,
    hard_conds=sample['hard_conds'],
    n_samples=25,
    horizon=dataset.n_support_points,
)
```

Here the task vector is unbatched; `run_inference` repeats it for each sample.
For a new start/goal pair, concatenate their positions and use
`dataset.normalize_tasks(...)`. Endpoint hard conditions use the **trajectory**
normalizer via `dataset.get_hard_conditions(..., normalize=True)`.
The original planners pass `context=None`, so using these conditional weights in
those planners requires passing task context and constructing the matching model.
