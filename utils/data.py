from typing import Any
import torch
from sklearn.datasets import make_s_curve, make_circles, make_moons

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

class oneDimension:
    def __init__(self,) -> None:
        self.d = 1
        pass
    
class twoDimension:
    def __init__(self,) -> None:
        self.d = 2
        pass

class Gaussian(oneDimension):
    def __init__(self, mu=0, sigma=1) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        pass
    
    def __call__(self, num_samples):
        return torch.normal(self.mu, self.sigma, size=(num_samples,1))

class Gaussian2d(twoDimension):
    def __init__(self, mu=0, sigma=1) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        pass
    
    def __call__(self, num_samples):
        return torch.normal(self.mu, self.sigma, size=(num_samples,2))


class twoGaussian(oneDimension):
    def __init__(self, mu1=-8, mu2=8,sigma=1) -> None:
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma
        pass
    
    def __call__(self, num_samples):
        half_num_samples = num_samples // 2
        samples1 = torch.normal(self.mu1, self.sigma, size=(half_num_samples,1))
        samples2 = torch.normal(self.mu2, self.sigma, size=(num_samples-half_num_samples,1))
        return torch.concatenate([samples1, samples2])
    

class fourGaussian2d(twoDimension):
    def __init__(self, m=8) -> None:
        super().__init__()
        self.mus = [(m, m), (m, -m), (-m, m), (-m, -m)]
    
    def __call__(self, num_samples):
        samples = int(num_samples / 4)
        cov_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        L = torch.linalg.cholesky(cov_matrix)

        target_dist_list = []

        # 循环遍历每一个均值向量，并生成相应的多元正态分布样本集
        for mu in self.mus:
            # 生成均值为0，标准差为1的正态分布样本
            Z = torch.normal(mean=0.0, std=1.0, size=(samples, 2))
            
            # 使用线性变换得到具有给定均值和协方差矩阵的多元正态分布样本
            X = torch.tensor(mu) + Z.matmul(L.T)
            
            # 将生成的样本集添加到列表中
            target_dist_list.append(X)
        return torch.concatenate(target_dist_list)

class S(twoDimension):
    def __init__(self, scale=3, drift=[0, 0]) -> None:
        super().__init__()
        self.scale = scale
        self.drift = torch.tensor(drift)
        pass  
    
    def __call__(self, num_samples):
        return torch.tensor(make_s_curve(n_samples=num_samples)[0][:, ::2], dtype=torch.float32) * self.scale + self.drift

class fourS(twoDimension):
    def __init__(self, scale=3, drift_all=0) -> None:
        super().__init__()
        self.scale = scale
        self.Ss = []
        for i in [(drift_all, drift_all), (drift_all, -drift_all), (-drift_all, drift_all), (-drift_all, -drift_all)]:
            self.Ss.append(S(scale, i))
        pass  
    
    def __call__(self, num_samples):
        ret = []
        for s in self.Ss:
            ret.append(s(int(num_samples/4)))
        return torch.concat(ret, dim=0)


class circle(twoDimension):
    def __init__(self, scale=8) -> None:
        super().__init__()
        self.scale = scale
        pass  
    
    def __call__(self, num_samples) -> Any:
        return torch.tensor(make_circles(n_samples=num_samples)[0], dtype=torch.float32) * self.scale


class DiagonalMatching(twoDimension):

    def __init__(self, ):
        super().__init__()
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

def rotate2d(x, radians):
    """Build a rotation matrix in 2D, take the dot product, and rotate using PyTorch."""
    radians = torch.tensor(radians, dtype=x.dtype)
    c, s = torch.cos(radians), torch.sin(radians)
    j = torch.tensor([[c, s], [-s, c]], dtype=x.dtype)
    m = torch.matmul(j, x)
    return m

class Moon(twoDimension):
    def __init__(self, ):
        super().__init__()
        pass
    def sample(self, n_samples):
        n = n_samples
        x = torch.linspace(0, torch.pi, n // 2)
        u = torch.stack([torch.cos(x) + 0.5, -torch.sin(x) + 0.2], dim=1) * 10.0
        u += 0.5 * torch.normal(mean=0.0, std=1.0, size=u.shape)
        u /= 3
        v = torch.stack([torch.cos(x) - 0.5, torch.sin(x) - 0.2], dim=1) * 10.0
        v += 0.5 * torch.normal(mean=0.0, std=1.0, size=v.shape)
        v /= 3
        samples = torch.cat([u, v], dim=0)

        # rotate and shrink samples
        samples_t = torch.zeros(samples.shape)
        for i, v in enumerate(samples):
            samples_t[i] = rotate2d(v, 180)

        return {"initial": samples_t.float(),
                "final": samples.float()}

class sourceMoon(Moon):
    def __init__(self,) -> None:
        super().__init__()
        pass  
    
    def __call__(self, num_samples) -> Any:
        return self.sample(num_samples)['initial']
    
class targetMoon(Moon):
    def __init__(self,) -> None:
        super().__init__()
        pass  
    
    def __call__(self, num_samples) -> torch.Tensor:
        return self.sample(num_samples)['final']
    
if __name__ == "__main__":
    test = sourceDiagonalMatching()
    print(test(10).shape)
    pass