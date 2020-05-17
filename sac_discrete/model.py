import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.nn import functional as F

from core.network import weights_init_xavier, create_linear_network, BaseNet

class TwinnedQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(TwinnedQNetwork, self).__init__()
        self.num_actions = num_actions
        self.Q1 = create_linear_network(
            num_inputs, num_actions, hidden_units=hidden_units,
            initializer=initializer)
        self.Q2 = create_linear_network(
            num_inputs, num_actions, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        T, B, *_ = states.shape
        states = torch.flatten(states, 0, 1)
        states = states.float()
        states = states.view(T * B, -1)

        q1 = self.Q1(states)
        q2 = self.Q2(states)

        q1 = q1.view(T, B, self.num_actions)
        q2 = q2.view(T, B, self.num_actions)

        return q1, q2

class CateoricalPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(CateoricalPolicy, self).__init__()
        self.policy = create_linear_network(
            num_inputs, num_actions, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        T, B, *_ = states.shape
        x = torch.flatten(states, 0, 1)
        x = x.float()
        x = x.view(T * B, -1)

        action_logits = self.policy(x)

        action_logits = action_logits.view(T, B, self.num_actions)

        return action_logits

    def act(self, states):
        # act with greedy policy
        action_logits = self.forward(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        # act with exploratory policy
        action_probs = F.softmax(self.policy(states), dim=2)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample()#.view(-1, 1)

        # avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class SACNet(BaseNet):
    def __init__(self, observation_shape, num_actions, use_lstm=False, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(SACNet, self).__init__()
        self.num_actions = num_actions
        self.policy = CateoricalPolicy(
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
            action, _, action_log_probs = self.policy.sample(state)
        return action.cpu().reshape(-1), action_log_probs

    def act(self, inputs, core_state=()):
        x = inputs["frame"]
        x = torch.flatten(x, 0, 1)
        
        states = x.float()
        action, action_log_probs = self.explore(states)
        return (
            dict(action=action.view(-1, 1), action_log_probs=action_log_probs),
            core_state,
        )