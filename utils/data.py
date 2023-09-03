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

class gaussian:
    def __init__(self, mu=0, sigma=1) -> None:
        self.mu = mu
        self.sigma = sigma
        pass
    
    def __call__(self, num_samples):
        return torch.normal(self.mu, self.sigma, size=(num_samples,1))

class two_gaussian:
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