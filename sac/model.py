import torch
import torch.nn as nn
from torch.distributions import Normal

from core.network import weights_init_xavier, create_linear_network, BaseNet


class TwinnedQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = create_linear_network(
            num_inputs+num_actions, 1, hidden_units=hidden_units,
            initializer=initializer)
        self.Q2 = create_linear_network(
            num_inputs+num_actions, 1, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states, actions):
        T, B, *_ = states.shape
        states = torch.flatten(states, 0, 1)
        states = states.float()
        states = states.view(T * B, -1)

        actions = torch.flatten(actions, 0, 1)
        actions = actions.float()
        actions = actions.view(T * B, -1)

        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)

        q1 = q1.view(T, B, 1)
        q2 = q2.view(T, B, 1)

        return q1, q2


class GaussianPolicy(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(GaussianPolicy, self).__init__()
        self.num_actions = num_actions
        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.policy = create_linear_network(
            num_inputs, num_actions*2, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        T, B, *_ = states.shape
        x = torch.flatten(states, 0, 1)
        x = x.float()
        x = x.view(T * B, -1)

        mean, log_std = torch.chunk(self.policy(x), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        mean = mean.view(T, B, self.num_actions)
        log_std = log_std.view(T, B, self.num_actions)

        return mean, log_std

    def sample(self, states):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        action_log_probs = normals.log_prob(xs)
        log_probs = action_log_probs\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=2, keepdim=True)

        return actions, entropies, torch.tanh(means), action_log_probs

    def sample_mean(self, states):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        #xs = normals.rsample()
        actions = torch.tanh(means)
        # calculate entropies
        action_log_probs = normals.log_prob(means)
        log_probs = action_log_probs\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=2, keepdim=True)
        return actions, entropies, torch.tanh(means), action_log_probs


    def get_action_log_probs(self, states, actions):
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        action_log_probs = normals.log_prob(self.atanh(actions))
        return action_log_probs

    @staticmethod
    def atanh(x):
        return 0.5*torch.log((1+x)/(1-x))

    

class SACNet(BaseNet):
    def __init__(self, observation_shape, num_actions, use_lstm=False, hidden_units=[256, 256], initializer='xavier'):
        super(SACNet, self).__init__()
        self.num_actions = num_actions
        self.policy = GaussianPolicy(
            observation_shape[0],
            num_actions,
            hidden_units=hidden_units)
        self.critic = TwinnedQNetwork(
            observation_shape[0],
            num_actions,
            hidden_units=hidden_units)
        self.critic_target = TwinnedQNetwork(
            observation_shape[0],
            num_actions,
            hidden_units=hidden_units)

    
    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, _, action_log_probs = self.policy.sample(state)
        return action.cpu().reshape(-1), action_log_probs

    def act(self, inputs, core_state=()):
        x = inputs["frame"]

        x = torch.flatten(x, 0, 1)
        
        states = x.float()
        action, action_log_probs = self.explore(states)
        return (
            dict(action=action, action_log_probs=action_log_probs),
            core_state,
        )
    