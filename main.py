import DiffusionBridge as db
import torch
import numpy as np
import argparse
from pathlib import Path
import time

from utils.data import gaussian, two_gaussian
from utils.plot import plot_bridge
def get_d(task):
    if task == 'gaussian2twogaussian':
        return 1
    else:
        raise NotImplementedError(f'Unknown task: {task}')

def main():
    parser = argparse.ArgumentParser(description='Train Gaussian2twoGaussian')
    
    parser.add_argument('--task', type=str, default='gaussian2twogaussian', required=True)
    # parser.add_argument('--model', type=str, default='unet++', required=True)

    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--checkpoint_score', type=str, default=None)
    parser.add_argument('--checkpoint_marginal', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--iter_nums', type=int, default=1)
    # parser.add_argument('--epoch_nums', type=int, default=3)
    # parser.add_argument('-b','--batch_size', type=int, default=8000)
    # parser.add_argument('-n','--normalize', action='store_true')
    # parser.add_argument('--num_workers', type=int, default=20)
    
    parser.add_argument('--debug', action='store_true')
    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    experiment_name = args.task
    if args.change_epsilons:
        experiment_name += '_change_epsilons'
    if args.filter_number is not None and 'mnist' in args.task:
        experiment_name += f'_filter{args.filter_number}'
    
    if args.debug:
        log_dir = Path('experiments') / 'debug' / 'train' / time.strftime("%Y-%m-%d/%H_%M_%S/")  
    else:
        log_dir = Path('experiments') / experiment_name / 'train' / time.strftime("%Y-%m-%d/%H_%M_%S/") 
    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir


    # problem settings
    d = get_d(args.task)
    sigma = torch.tensor(1.0)
    T = torch.tensor(1.0)
    M = 1000
    
    # f For Schrodinger Bridge
    f = lambda x, t: 0

    diffusion = db.diffusion.model(f, sigma, d, T, M)

    sigma = 1
    epsilon = 0.001
    T = 1


   

    source_dist = gaussian()
    target_dist = two_gaussian()

    epsilon = 1e-3
    minibatch = 1
    num_initial_per_batch = 5000
    num_iterations = 2000
    learning_rate = 0.01
    ema_momentum = 0.99
    if args.checkpoint_score is None:
        output = diffusion.my_learn_full_score_transition(source_dist, 
                                                        target_dist, 
                                                        epsilon, 
                                                        minibatch, 
                                                        num_initial_per_batch, 
                                                        num_iterations, 
                                                        learning_rate, 
                                                        ema_momentum,
                                                        scheduler='cos',
                                                        device=device)
        torch.save(score_transition_net.state_dict(), args.log_dir / 'score_transition_net.pt')
    else:
        output = diffusion.load_checkpoint_score(args.checkpoint_score)
    score_transition_net = output['net']



# plot_bridge(ts, bridge, source_sample, target_sample, show_rate=1)

    score_transition_net = score_transition_net.cpu()
    num_test_samples = 500
    source_sample = source_dist(num_test_samples)
    target_sample = target_dist(num_test_samples)
    out = diffusion.my_simulate_bridge_backwards(score_transition_net, source_sample, target_sample, 1e-3, modify = True, full_score = True)

    fig, _ = plot_bridge(diffusion.time.numpy(), out['trajectories'][:,:,0].detach().numpy().T, source_sample.numpy(), target_sample.numpy(), show_rate=1, show_gt=True)
    fig.savefig(args.log_dir / 'bridge_backward.jpg')

    out = diffusion.my_simulate_process(source_sample, target_sample)
    fig, _ = plot_bridge(diffusion.time.numpy(), out['trajectories'][:,:,0].detach().numpy().T, source_sample.numpy(), target_sample.numpy(), show_rate=1, show_gt=False)
    fig.savefig(args.log_dir / 'bridge_real.jpg')

    out = diffusion.my_learn_score_marginal(score_transition_net, source_dist, target_dist, epsilon, minibatch, num_initial_per_batch, num_iterations, learning_rate, ema_momentum, scheduler='cos')



