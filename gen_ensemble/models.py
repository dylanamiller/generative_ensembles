"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer(dim1, dim2):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.Tanh(),
    )

def initialize_model(dims):
    return [layer(dims[i], dims[i+1]) for i in range(len(dims)-1)]


class MLP(nn.Module):
    def __init__(self, scale_factor=0.5, n_hidden=2):
        super(MLP, self).__init__()
        self.scale_factor = scale_factor
        self.n_hidden = n_hidden

    def set_orthogonal(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def _initialize_model(self, dims):
        return nn.Sequential(*initialize_model(dims))

    def get_hidden_dims(self, hidden_dim):
        return [int(hidden_dim*self.scale_factor**i) for i in range(self.n_hidden)]

    def forward(self):
        ...

class Decoder(MLP):
    """ VAE decoder """
    def __init__(self, out_dim, hidden_dim, latent_dim, scale_factor=2, n_hidden=2):
        super(Decoder, self).__init__(scale_factor, n_hidden)
        hidden_dims = self.get_hidden_dims(hidden_dim)
        dims = [latent_dim, *hidden_dims, out_dim]
        self._decoder = self._initialize_model(dims)

        self.set_orthogonal()

        print(self)

    def forward(self, x):
        return self._decoder(x)


class Encoder(MLP):
    """ VAE encoder """
    def __init__(self, in_dim, hidden_dim, latent_dim, scale_factor=0.5, n_hidden=2):
        super(Encoder, self).__init__(scale_factor, n_hidden)
        hidden_dims = self.get_hidden_dims(hidden_dim)
        dims = [in_dim, *hidden_dims]
        self._encoder = self._initialize_model(dims)

        # latent mean and log_variance
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logsigma = nn.Linear(hidden_dims[-1], latent_dim)

        self.set_orthogonal()

        print(self)

    def forward(self, x):
        x = self._encoder(x)
        
        mu = self.mu(x)
        logsigma = self.logsigma(x)

        return mu, logsigma
