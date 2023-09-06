import torch

class marginalDataset(torch.utils.data.Dataset):
    def __init__(self, data, num_batches) -> None:
        self.data = data
        self.num_batches = num_batches
        
    def __len__(self):
        return self.num_batches

    def __getitem__(j):
        pass

class bridgeBackwardsDataset(torch.utils.data.Dataset):
    def __init__(self, diffusion, score_transition_net ,simulate_initial_state ,simulate_terminal_state, num_repeats, num_initial_per_batch, minibatch, device, epsilon, num_samples, N, d, M):
        self.diffusion = diffusion
        self.score_transition_net = score_transition_net
        self.simulate_initial_state = simulate_initial_state
        self.simulate_terminal_state = simulate_terminal_state
        self.num_repeats = num_repeats
        self.num_initial_per_batch = num_initial_per_batch
        self.minibatch = minibatch
        self.device = device
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.N = N
        self.d = d
        self.M = M
        
    def __len__(self):
        return self.num_repeats
    
    def __getitem__(self, i):
        initial_states = self.simulate_initial_state(self.num_initial_per_batch).repeat((self.minibatch,1)) # size (N, d)
        initial_states_repeated = initial_states.reshape((self.N,1,self.d)).repeat((1,self.M,1)) # size (N, M, d)
        initial_states_flatten = initial_states_repeated.flatten(start_dim = 0, end_dim = 1) # size (N*M, d)

        terminal_states = self.simulate_terminal_state(self.num_initial_per_batch).repeat((self.minibatch,1)) # size (N, d)
        terminal_states_repeated = terminal_states.reshape((self.N,1,self.d)).repeat((1,self.M,1)) # size (N, M, d)
        terminal_states_flatten = terminal_states_repeated.flatten(start_dim = 0, end_dim = 1) # size (N*M, d)
        
        # simulate trajectories from approximate diffusion bridge process backwards
        with torch.no_grad():
            simulation_output = self.diffusion.my_simulate_bridge_backwards(self.score_transition_net, initial_states_flatten, terminal_states_flatten, self.epsilon, self.num_samples, modify = True, full_score = True)
        trajectories = simulation_output['trajectories']
        scaled_brownian = simulation_output['scaled_brownian']
        
        return trajectories, scaled_brownian

class Gaussian:
    def __init__(self, mu=0, sigma=1) -> None:
        self.mu = mu
        self.sigma = sigma
        pass
    
    def __call__(self, num_samples):
        return torch.normal(self.mu, self.sigma, size=(num_samples,1))

class twoGaussian:
    def __init__(self, mu1=-8, mu2=8,sigma=1) -> None:
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma
        pass
    
    def __call__(self, num_samples):
        half_num_samples = num_samples // 2
        samples1 = torch.normal(self.mu1, self.sigma, size=(half_num_samples,1))
        samples2 = torch.normal(self.mu2, self.sigma, size=(num_samples-half_num_samples,1))
        return torch.concatenate([samples1, samples2])
    
    
    
class DiagonalMatching:

    def __init__(self, ):
        pass
        

    def sample(self, n_samples):
        n = n_samples // 2 + 1

        left_square = torch.stack([
            torch.rand(size=(n,)) * 0.2 - 1.2,
            torch.linspace(-0.1, 0.5, n)
        ], dim=1) * torch.tensor([4.0, 4.0]) + torch.tensor([-5.0, 3.0])

        right_square = torch.stack([
            torch.rand(size=(n,)) * 0.2 + 1.0,
            torch.linspace(-0.1, 0.5, n)
        ], dim=1) * torch.tensor([4.0, 4.0]) + torch.tensor([5.0, 3.0])

        top_square = torch.stack([
            torch.linspace(-0.3, 0.3, n),
            torch.rand(size=(n,)) * 0.2 + 0.8
        ], dim=1) * torch.tensor([4.0, 2.0]) + torch.tensor([0.0, 3.0])

        bottom_square = torch.stack([
            torch.linspace(-0.3, 0.3, n),
            torch.rand(size=(n,)) * 0.2 - 1.5
        ], dim=1) * torch.tensor([4.0, 2.0]) + torch.tensor([0.0, -3.0])

        # rand_shuffling = torch.randperm(n_samples)

        initial = torch.cat([left_square, top_square], dim=0)[:n_samples]
        final = torch.cat([right_square, bottom_square], dim=0)[:n_samples]

        return {
            "initial": initial,
            "final": final
        }

class sourceDiagonalMatching(DiagonalMatching):
    
    def __init__(self, ):
        super().__init__()
        
    def __call__(self, num_samples):
        return self.sample(num_samples)['initial']
    
class targetDiagonalMatching(DiagonalMatching):
    
    def __init__(self, ):
        super().__init__()
        
    def __call__(self, num_samples):
        return self.sample(num_samples)['final']
    
    
if __name__ == "__main__":
    test = sourceDiagonalMatching()
    print(test(10).shape)
    pass