import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from core.network import BaseNet

class RandomAgent_discrete(BaseNet):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(RandomAgent_discrete, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

    def initial_state(self, batch_size):
        return tuple()
        
    
    def forward(self, inputs, agent_state=dict(core_state=(), )):
        action = torch.Tensor(np.random.randint([self.num_actions])).int().view(1,1)
        
        return (
            dict(action=action),
            dict(core_state=agent_state["core_state"],),
        )
    
    def act(self, inputs, agent_state=dict(core_state=(), )):
        return self.forward(inputs, agent_state)

    def initialize(self, inputs, batch_size):
        return dict(core_state=tuple(),)