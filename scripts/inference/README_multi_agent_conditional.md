# Conditional multi-agent inference

`inference_multi_agent_conditional.py` is a copy of the original multi-agent
entry point using the new `MPDEnsembleConditional` planner. The original files
are unchanged. This supports multiple planar robots in **one map**, with one
trajectory segment per robot. Multiple tiles are rejected because their unknown
junction positions need additional context handling.

From `mmd/scripts/inference`, run:

```bash
conda activate mmd
python inference_multi_agent_conditional.py \
    --model_dir /absolute/path/to/conditional/training/run/0 \
    --checkpoint model_current_state_dict.pth \
    --num_agents 3
```

`--model_dir` must contain `args.yaml` and `checkpoints/`, including all parameter
folders and the seed folder. It is the same directory accepted by
`inference_single_agent.py`. `--checkpoint` accepts a filename inside
`checkpoints/` or an absolute state-dictionary path. If omitted, EMA weights are
selected when the training configuration enables EMA.

Each robot gets its own normalized `[start_position, goal_position]` context.
The context is used for warmup, initial diffusion sampling, and local replanning
with conflict constraints. The U-Net conditioning mode and dimensions come from
the saved training configuration and dataset. Dataset normalization uses the
training data; `--dataset_subdir` can override its location if the data moved.

The entry point retains the original circle scenario: robots start on a circle
of radius 0.8 and aim for opposite points. Edit `start_state_pos_l` and
`goal_state_pos_l` in the new entry point to use other starts and goals.

The original multi-agent search, collision/smoothness guidance, and shared
`apply_hard_conditioning()` function are reused. No independent endpoint clamping
is added. A shared helper that only returns `x` continues to disable inpainting;
start/goal context remains active.

Other options:

- `--planner XECBS`: also accepts ECBS, XCBS, CBS, and PP.
- `--n_samples 64`: candidate trajectories per agent.
- `--runtime_limit 180`: planning time limit in seconds (model setup is separate).
- `--seed 18`: random seed.
- `--no-render`: skip the plot and animation.
- `--compile_model`: opt into `torch.compile`; disabled by default.

Like the original ensemble planner, this workflow requires CUDA. The current
planner requires velocity-enabled trajectories and the configured MMD horizon
(64 by default). Results use the original result-directory layout with the
instance name `conditional_single_tile` and planner name `MPDEnsembleConditional`.
