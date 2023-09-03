import torch

# class myDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data
#         pass

#     def __

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