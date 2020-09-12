import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from models import Encoder, Decoder
from distributions import Gaussian


class DynamicsEnsemble(nn.Module):
    """ Model of dynamics
    
        Single encoder, multiple decoder. Idea is to generate
        next state prediction, allowing for interpolation in
        regions of poor data coverage.
    """
    def __init__(self, 
                 ensemble_size, 
                 in_dim, 
                 out_dim, 
                 encoder_hidden_dim, 
                 decoder_hidden_dim, 
                 latent_dim,
                 n_hidden,
                 training=False,
                 learn_rate=0.0001,
                 ):

        super(DynamicsEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.training = training

        self.encoder = Encoder(in_dim, encoder_hidden_dim, latent_dim)
        self.decoders = self.build_decoder_ensemble(out_dim, decoder_hidden_dim, latent_dim, n_hidden)
        self.opt = optim.Adam(self.parameters(), lr=learn_rate)

        self.gaussian = Gaussian(latent_dim)

    def build_decoder_ensemble(self, out_dim, hidden_dim, latent_dim, n_hidden):
        return [
            Decoder(out_dim, hidden_dim, latent_dim) for _ in range(self.ensemble_size)
        ]

    def training_pass(self, z):
        return torch.stack([
            decoder(z) for decoder in self.decoders
        ])

    def squared_error(self, y_hat, y):
        return (0.5*(y_hat-y)**2).mean(1).mean(1)

    def update(self, y, y_hat):
        self.opt.zero_grad()
        y = torch.cat(5*[y.unsqueeze(0)], dim=0)
        loss = self.squared_error(y_hat[0], y)
        mse_loss = loss.mean()
        mse_loss.backward()
        self.opt.step()
        return loss, mse_loss

    def random_decoder(self, z):
        return self.decoders[np.random.choice(self.ensemble_size)](z)

    def sample_gaussian(self, mu, sigma):
        return self.gaussian(mu, sigma)

    def forward(self, x):
        """
            Get mu and sigma from encoder, sample from 
            gaussian and center distribution. Use new
            distribution for decoder.

            If training, pass z to all decoders, else
            pass z to randomly chosen decoder.
        """
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        decoder = self.training_pass if self.training else self.random_decoder
        next_obs_hat = decoder(z)

        return next_obs_hat, mu, logsigma