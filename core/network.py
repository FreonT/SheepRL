import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

def weights_init_xavier(m):
    if isinstance(m, nn.Linear)\
            or isinstance(m, nn.Conv2d)\
            or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def create_linear_network(input_dim, output_dim, hidden_units=[256, 256],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=weights_init_xavier):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units, bias=False))
        model.append(hidden_activation)
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(output_activation)

    return nn.Sequential(*model).apply(initializer)

class Gaussian(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_units=[256, 256],
                 std=None, leaky_slope=0.2):
        super(Gaussian, self).__init__()
        self.net = create_linear_network(
            input_dim, 2*output_dim if std is None else output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(leaky_slope),
            initializer=weights_init_xavier)

        self.std = std

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return Normal(loc=mean, scale=std)


class ConstantGaussian(nn.Module):

    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def initialize(self, inputs, batch_size):

        return dict(core_state=self.initial_state(batch_size))

    def initial_state(self, batch_size):    
        return tuple()