"""
Single-agent diffusion-prior inference, adapted from MMD's sampling pipeline.

From scripts/inference:
    python inference_single_agent.py --start 0.8 0.0 --goal -0.8 0.0

Supports original and start/goal-conditioned state-dictionary checkpoints.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from mmd.datasets import TrajectoryDataset
from mmd.models import TemporalUnet, TaskModel, UNET_DIM_MULTS
from mmd.trainer import get_model
from mmd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed


REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def sample_trajectories(model, dataset, start, goal, n_samples=16):
    # Endpoint conditions use trajectory normalization; context uses task normalization.
    hard_conds = dataset.get_hard_conditions(torch.stack((start, goal)), normalize=True)

    context = None
    if model.context_model is not None:
        task = dataset.normalize_tasks(torch.cat((start, goal)))
        context = {'tasks': task}

    x = model.run_inference(
        context=context,
        hard_conds=hard_conds,
        n_samples=n_samples,
        horizon=dataset.n_support_points,
    )

    if not torch.isfinite(x).all():
        raise RuntimeError('Sampling produced non-finite states; check weights and dataset normalization.')
    return dataset.unnormalize_trajectories(x)


def run_single_agent(args):
    fix_random_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}
    model_dir = Path(args.model_dir).expanduser().resolve()
    train_args = load_params_from_yaml(model_dir / 'args.yaml')

    dataset_path = Path(args.dataset_subdir or train_args['dataset_subdir']).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / 'data_trajectories' / dataset_path
    dataset = TrajectoryDataset(
        dataset_subdir=str(dataset_path),
        include_velocity=train_args['include_velocity'],
        tensor_args=tensor_args,
    )

    conditioning_type = train_args.get('conditioning_type')
    conditional = conditioning_type not in (None, 'None')
    context_dim = dataset.fields[dataset.field_key_task].shape[-1]
    model = get_model(
        model_class=train_args['diffusion_model_class'],
        model=TemporalUnet(
            state_dim=dataset.state_dim,
            n_support_points=dataset.n_support_points,
            unet_input_dim=train_args['unet_input_dim'],
            dim_mults=UNET_DIM_MULTS[train_args['unet_dim_mults_option']],
            conditioning_type=conditioning_type,
            conditioning_embed_dim=context_dim,
        ),
        context_model=TaskModel(in_dim=context_dim, out_dim=context_dim) if conditional else None,
        variance_schedule=train_args['variance_schedule'],
        n_diffusion_steps=train_args['n_diffusion_steps'],
        predict_epsilon=train_args['predict_epsilon'],
        tensor_args=tensor_args,
    )
    checkpoint_name = args.checkpoint or (
        'ema_model_current_state_dict.pth' if train_args.get('use_ema', True)
        else 'model_current_state_dict.pth'
    )
    checkpoint = Path(checkpoint_name).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = model_dir / 'checkpoints' / checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    start = torch.tensor(args.start, **tensor_args)
    goal = torch.tensor(args.goal, **tensor_args)
    if start.numel() != dataset.robot.q_dim or goal.numel() != dataset.robot.q_dim:
        raise ValueError(f'Start and goal must each contain {dataset.robot.q_dim} position coordinates.')

    print(f'Checkpoint: {checkpoint}')
    print(f'Context conditioning: {conditioning_type}')
    print(f'Requested start: {start}; requested goal: {goal}')
    trajectories = sample_trajectories(
        model, dataset, start, goal, n_samples=args.n_samples,
    )

    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories.cpu(), results_dir / 'trajectories.pt')
    with open(results_dir / 'args.yaml', 'w') as stream:
        yaml.safe_dump({**vars(args), 'device': device, 'checkpoint': str(checkpoint),
                        'dataset_subdir': str(dataset_path), 'conditioning_type': conditioning_type}, stream)
    if not args.no_plot:
        fig, _ = dataset.planner_visualizer.render_robot_trajectories(
            trajs=trajectories, start_state=start, goal_state=goal,
        )
        fig.savefig(results_dir / 'trajectories.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'Saved {tuple(trajectories.shape)} trajectories to {results_dir / "trajectories.pt"}')
    return trajectories


def parse_args():
    parser = argparse.ArgumentParser(description='Sample single-agent trajectories from one diffusion model.')
    parser.add_argument('--model_dir', default=str(REPO_ROOT / 'data_trained_models/EnvEmptyNoWait2D-RobotPlanarDisk'),
                        help='Training run directory containing args.yaml and checkpoints/.')
    parser.add_argument('--dataset_subdir', default=None,
                        help='Override the training dataset name or absolute path; use the same data for normalization.')
    parser.add_argument('--checkpoint', default=None,
                        help='State-dictionary filename under checkpoints/, or an absolute path. Defaults to EMA if trained.')
    parser.add_argument('--start', type=float, nargs='+', default=[0.8, 0.0])
    parser.add_argument('--goal', type=float, nargs='+', default=[-0.8, 0.0])
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--device', default=None, help='Default: CUDA when available, otherwise CPU.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--results_dir', default='results/single_agent')
    parser.add_argument('--no-plot', action='store_true', help='Skip trajectory visualization.')
    args = parser.parse_args()
    if args.n_samples <= 0:
        parser.error('--n_samples must be positive.')
    return args


if __name__ == '__main__':
    run_single_agent(parse_args())
