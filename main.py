import DiffusionBridge as db
import torch
import numpy as np
import argparse
from pathlib import Path
import time
from rich.panel import Panel
from rich.pretty import Pretty
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from utils.data import Gaussian, twoGaussian, sourceDiagonalMatching, targetDiagonalMatching, Gaussian2d, fourGaussian2d, S, circle, sourceMoon, targetMoon
from utils.plot import plot_bridge, plot_t, save_gif_frame, save_gif_traj


def get_dist(task):
    if task == 'gaussian2twogaussian':
        return Gaussian(), twoGaussian()
    elif task == 't':
        return sourceDiagonalMatching(), targetDiagonalMatching()
    elif task == 'gaussian2fourgaussian2d':
        return Gaussian2d(), fourGaussian2d()
    elif task == 'gaussian2fourgaussian2dfar':
        return Gaussian2d(), fourGaussian2d(m=20)
    elif task == 'gaussian2S':
        return Gaussian2d(), S()
    elif task == 'gaussian2Sfar':
        return Gaussian2d(mu=50), S()
    elif task == 'gaussian2ring':
        return Gaussian2d(), circle()
    elif task == 'moon2moon':
        return sourceMoon(), targetMoon()


def main():
    parser = argparse.ArgumentParser(description='Train Gaussian2twoGaussian')

    parser.add_argument('--task', type=str,
                        default='gaussian2twogaussian', required=True)
    # parser.add_argument('--model', type=str, default='unet++', required=True)

    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--checkpoint_score', type=str, default=None)
    parser.add_argument('--checkpoint_marginal', type=str, default=None)
    parser.add_argument('--continue_score', action='store_true')
    parser.add_argument('--continue_marginal', action='store_true')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('-f', '--full_model', action='store_true')
    # parser.add_argument('--iter_nums', type=int, default=1)
    # parser.add_argument('--epoch_nums', type=int, default=3)
    parser.add_argument('-bt', '--batch_size_tran', type=int, default=5000)
    parser.add_argument('-bm', '--batch_size_marg', type=int, default=5000)
    parser.add_argument('-it', '--iters_tran', type=int, default=500)
    parser.add_argument('-im', '--iters_marg', type=int, default=500)
    parser.add_argument('--num_test_samples', type=int, default=1000)
    parser.add_argument('--device', type=str)
    # parser.add_argument('-n','--normalize', action='store_true')
    # parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--drift', type=str, default=None, choices=[None, 'bridge', 'process'])
    parser.add_argument('--noforward', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    experiment_name = args.task
    if args.drift is not None:
        experiment_name += f'_drift_{args.drift}'
        
    if args.debug:
        log_dir = Path('experiments') / 'debug' / 'train' / \
            time.strftime("%Y-%m-%d/%H_%M_%S/")
    else:
        log_dir = Path('experiments') / experiment_name / \
            'train' / time.strftime("%Y-%m-%d/%H_%M_%S/")

    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir

    args.checkpoint_score = Path(
        args.checkpoint_score) if args.checkpoint_score is not None else None
    args.checkpoint_marginal = Path(
        args.checkpoint_marginal) if args.checkpoint_marginal is not None else None

    args.device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'cpu') if args.device is None else args.device
    main_worker(args)


def main_worker(args):

    console = Console(record=True, color_system='truecolor')
    pretty = Pretty(args.__dict__, expand_all=True)
    panel = Panel(pretty, title='Arguments', expand=False, highlight=True)
    console.log(panel)
    console.log(f"Saving to {Path.absolute(args.log_dir)}")

    # problem settings
    source_dist, target_dist = get_dist(args.task)
    assert source_dist.d == target_dist.d
    d = source_dist.d
    sigma = torch.tensor(1.0)
    T = torch.tensor(1.0)
    M = args.M

    # f For Schrodinger Bridge
    def f(x, t): return 0

    diffusion = db.diffusion.model(f, sigma, d, T, M)

    # source_dist = Gaussian()
    # target_dist = twoGaussian()
    ############################ Model ########################################
    epsilon = 0.
    ema_momentum = 0.99

    if args.checkpoint_score is None:
        output = diffusion.my_learn_full_score_transition(
            source_dist, target_dist, epsilon, 1, args.batch_size_tran, args.iters_tran, args.lr, ema_momentum, scheduler=args.scheduler, device=args.device, drift=args.drift, full_model=args.full_model)
    else:
        if args.continue_score:
            output = diffusion.my_learn_full_score_transition(
                source_dist, target_dist, epsilon, 1, args.batch_size_tran, args.iters_tran, args.lr, ema_momentum, scheduler=args.scheduler, device=args.device, drift=args.drift, full_model=args.full_model, checkpoint=args.checkpoint_score)
        else:
            output = diffusion.load_checkpoint_score(args.checkpoint_score, full_model=args.full_model)
            console.log(
                f"Loaded score checkpoint from {Path.absolute(args.checkpoint_score)}")
    score_transition_net = output['net']
    console.log(
        f"Transition Model {score_transition_net.__class__.__name__} Parameters: {float(sum(p.numel() for p in score_transition_net.parameters())/1e6)}M")
    torch.save(score_transition_net.state_dict(),
               args.log_dir / 'score_transition_net.pt')

    if not args.noforward:
        if args.checkpoint_marginal is None:
            output = diffusion.my_learn_score_marginal(score_transition_net, source_dist, target_dist, epsilon, 1,
                                                    args.batch_size_marg, args.iters_marg, args.lr, ema_momentum, num_workers=args.num_workers, scheduler=args.scheduler, device=args.device)
        else:
            if args.continue_marginal:
                output = diffusion.my_learn_score_marginal(score_transition_net, source_dist, target_dist, epsilon, 1,
                                                        args.batch_size_marg, args.iters_marg, args.lr, ema_momentum, num_workers=args.num_workers, scheduler=args.scheduler, device=args.device, checkpoint=args.checkpoint_marginal)
            else:
                output = diffusion.load_checkpoint_marginal(
                    args.checkpoint_marginal)
            console.log(
                f"Loaded marginal checkpoint from {Path.absolute(args.checkpoint_marginal)}")
        score_marginal_net = output['net']
        console.log(
            f"Marginal Model {score_marginal_net.__class__.__name__} Parameters: {float(sum(p.numel() for p in score_marginal_net.parameters())/1e6)}M")
        torch.save(score_marginal_net.state_dict(),
                args.log_dir / 'score_marginal_net.pt')

    score_transition_net = score_transition_net.cpu()
    score_marginal_net = score_marginal_net.cpu() if not args.noforward else None

    ############################# Test ########################################
    num_test_samples = args.num_test_samples
    source_sample = source_dist(num_test_samples)
    target_sample = target_dist(num_test_samples)
    backward_out = diffusion.my_simulate_bridge_backwards(
        score_transition_net, source_sample, target_sample, epsilon, modify=False, full_score=True, full_model=args.full_model)
    np.save('score_evaluations.npy', backward_out['score_evaluations'].detach().numpy())
    if not args.noforward:
        forward_out = diffusion.my_simulate_bridge_forwards(score_transition_net, score_marginal_net, source_sample,
                                                            target_sample, epsilon, num_samples=1, modify=True, full_score=True, new_num_steps=None)
    if args.drift is not None:
        real_out = diffusion.my_simulate_bridge(source_sample, target_sample) if args.drift == 'bridge' else diffusion.my_simulate_process(source_sample, target_sample)
    else:
        real_out = diffusion.simulate_process(source_sample)

    console.rule("Results")

    if args.task == 'gaussian2twogaussian':
        fig, _ = plot_bridge(diffusion.time.numpy(), backward_out['trajectories'][:, :, 0].detach(
        ).numpy().T, source_sample.numpy(), target_sample.numpy(), show_rate=1, show_gt=True)
        fig.savefig(args.log_dir / 'bridge_backward.jpg')
        if not args.noforward:
            fig, _ = plot_bridge(diffusion.time.numpy(), forward_out['trajectories'][:, :, 0].detach(
            ).numpy().T, source_sample.numpy(), target_sample.numpy(), show_rate=1, show_gt=False)
            fig.savefig(args.log_dir / 'bridge_forward.jpg')
    
        fig, _ = plot_bridge(diffusion.time.numpy(), real_out['trajectories'][:, :, 0].detach(
        ).numpy().T, source_sample.numpy(), target_sample.numpy(), show_rate=1, show_gt=False)
        fig.savefig(args.log_dir / 'bridge_real.jpg')

    elif args.task in ['t', 'gaussian2fourgaussian2dfar', 'gaussian2Sfar']:
        fig, _ = plot_t(backward_out['trajectories'].detach().numpy(), bound=backward_out['trajectories'].abs().max().item()+1)
        fig.savefig(args.log_dir / 'bridge_backward.jpg')
        
        if not args.noforward:
            fig, _ = plot_t(forward_out['trajectories'].detach().numpy(), bound=backward_out['trajectories'].abs().max().item()+1)
            fig.savefig(args.log_dir / 'bridge_forward.jpg')

        fig, _ = plot_t(real_out['trajectories'].detach().numpy(), bound=backward_out['trajectories'].abs().max().item()+1)
        fig.savefig(args.log_dir / 'bridge_real.jpg')
        save_gif_traj(backward_out['trajectories'][:,:-1, :].detach().numpy(), save_path=args.log_dir, save_name='backward_bridge_traj.gif', bound=backward_out['trajectories'].abs().max().item()+1)
        
        fig, _ = plot_t(diffusion.simulate_process(source_sample)['trajectories'].detach().numpy(), bound=backward_out['trajectories'].abs().max().item()+1)
        fig.savefig(args.log_dir / 'process.jpg')

    elif args.task in ['gaussian2fourgaussian2d', 'gaussian2S', 'gaussian2ring', 'moon2moon']:
        save_gif_frame(backward_out['trajectories'][:,:-1, :].detach().numpy(), save_path=args.log_dir, save_name='backward_bridge.gif')
        if not args.noforward:
            save_gif_frame(forward_out['trajectories'][:,:-1, :].detach().numpy(), save_path=args.log_dir, save_name='forward_bridge.gif')
        save_gif_frame(real_out['trajectories'][:,:-1, :].detach().numpy(), save_path=args.log_dir, save_name='real_bridge.gif')
    
        save_gif_traj(backward_out['trajectories'][:,:-1, :].detach().numpy(), save_path=args.log_dir, save_name='backward_bridge_traj.gif', bound=10)
        

if __name__ == '__main__':
    main()
