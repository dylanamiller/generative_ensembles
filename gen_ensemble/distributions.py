import torch
from torch.distributions import MultivariateNormal


class Gaussian:
    def __init__(self, dim):
        self.dim = dim
        self.dist = MultivariateNormal

    def __call__(self, mu, sigma):
        print(mu.size(), sigma.size())
        print(torch.exp(sigma).size(), torch.eye(self.dim).size())
        sigma = torch.exp(sigma) # * torch.eye(self.dim)
        return self.dist(mu, sigma)