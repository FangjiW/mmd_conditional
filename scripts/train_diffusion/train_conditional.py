"""
Adapted from train.py and https://github.com/jacarvalho/mpd-public
Train with normalized start/goal positions as task context.
"""
import os
import torch

from experiment_launcher import single_experiment_yaml, run_experiment
from mmd import trainer
from mmd.models import UNET_DIM_MULTS, TemporalUnet, TaskModel
from mmd.trainer import get_dataset, get_model, get_loss, get_summary
from mmd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


@single_experiment_yaml
def experiment(
    ########################################################################
    # Dataset
    dataset_subdir: str = 'EnvSimple2D-RobotPointMass',
    # dataset_subdir: str = 'EnvSpheres3D-RobotPanda',
    include_velocity: bool = True,

    ########################################################################
    # Diffusion Model
    diffusion_model_class: str = 'GaussianDiffusionModel',
    variance_schedule: str = 'exponential',  # cosine
    n_diffusion_steps: int = 25,
    predict_epsilon: bool = True,

    # Unet
    unet_input_dim: int = 32,
    unet_dim_mults_option: int = 1,
    conditioning_type: str = 'default',

    ########################################################################
    # Loss
    loss_class: str = 'GaussianDiffusionLoss',

    # Training parameters
    batch_size: int = 32,
    lr: float = 1e-4,
    num_train_steps: int = 500000,
    resume_checkpoint: str = '',  # Optional state dictionary; optimizer and step counter restart.

    use_ema: bool = True,
    use_amp: bool = False,

    # Summary parameters
    steps_til_summary: int = 10,
    summary_class: str = 'SummaryTrajectoryGeneration',

    steps_til_ckpt: int = 50000,

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    ########################################################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = 'logs_conditional',

    ########################################################################
    # WandB
    wandb_mode: str = 'disabled',  # "online", "offline" or "disabled"
    wandb_entity: str = 'scoreplan',
    wandb_project: str = 'train_diffusion_conditional',
    **kwargs
):
    if conditioning_type not in ('default', 'concatenate', 'attention'):
        raise ValueError('Conditional training requires default, concatenate, or attention conditioning.')
    if num_train_steps <= 0:
        raise ValueError('num_train_steps must be positive.')

    fix_random_seed(seed)

    device = get_torch_device(device=device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=include_velocity,
        dataset_subdir=dataset_subdir,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args
    )

    dataset = train_subset.dataset

    # The existing loss and summary use build_context(), which expects this flag.
    # These datasets describe a fixed environment; only the task is context.
    dataset.variable_environment = False
    conditioning_embed_dim = dataset.fields[dataset.field_key_task].shape[-1]

    # Model
    diffusion_configs = dict(
        variance_schedule=variance_schedule,
        n_diffusion_steps=n_diffusion_steps,
        predict_epsilon=predict_epsilon,
    )

    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=unet_input_dim,
        dim_mults=UNET_DIM_MULTS[unet_dim_mults_option],
        conditioning_type=conditioning_type,
        conditioning_embed_dim=conditioning_embed_dim,
    )

    model = get_model(
        model_class=diffusion_model_class,
        model=TemporalUnet(**unet_configs),
        context_model=TaskModel(in_dim=conditioning_embed_dim, out_dim=conditioning_embed_dim),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )

    # Continue from weights before the trainer initializes its optimizer and EMA.
    if resume_checkpoint:
        checkpoint_path = os.path.abspath(os.path.expanduser(resume_checkpoint))
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        print(f'Loaded training weights from {checkpoint_path}')
        print(f'Starting a fresh optimizer and EMA for {num_train_steps} additional training steps.')

    # Loss
    loss_fn = val_loss_fn = get_loss(
        loss_class=loss_class
    )

    # Summary
    summary_fn = get_summary(
        summary_class=summary_class,
    )

    # Train
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        epochs=get_num_epochs(num_train_steps, batch_size, len(train_subset)),
        max_steps=num_train_steps,
        model_dir=results_dir,
        summary_fn=summary_fn,
        lr=lr,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        steps_til_summary=steps_til_summary,
        steps_til_checkpoint=steps_til_ckpt,
        clip_grad=True,
        use_ema=use_ema,
        use_amp=use_amp,
        debug=debug,
        tensor_args=tensor_args
    )


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
