import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.nn import functional as F

from core.network import weights_init_xavier, create_linear_network, Gaussian, ConstantGaussian, BaseNet


class Decoder(nn.Module):

    def __init__(self, input_dim=288, output_dim=3, std=1.0, leaky_slope=0.2):
        super(Decoder, self).__init__()
        self.std = std

        """
        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, 3, 5, 2, 2, 1),
            nn.LeakyReLU(leaky_slope)
        ).apply(weights_init_xavier)
        """
        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, input_dim, 3),
            nn.LeakyReLU(leaky_slope),
            # (32+256, 4, 4) -> (256, 12, 12)
            nn.ConvTranspose2d(input_dim, 256, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (256, 12, 12) -> (128, 21, 21)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 2),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(leaky_slope)
        ).apply(weights_init_xavier)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x,  dim=-1)

        num_batches, num_sequences, latent_dim = x.size()
        x = x.view(num_batches * num_sequences, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(num_batches, num_sequences, C, W, H)
        return Normal(loc=x, scale=torch.ones_like(x) * self.std)


class Encoder(nn.Module):

    def __init__(self, input_dim=3, output_dim=256, leaky_slope=0.2):
        super(Encoder, self).__init__()
        """
        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(leaky_slope)
        ).apply(weights_init_xavier)
        """
        self.net = nn.Sequential(
            # (3, 84, 84) -> (32, 42, 42)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(leaky_slope),
            # (32, 42, 42) -> (64, 21, 21)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (64, 21, 21) -> (128, 24, 24)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 24, 24) -> (256, 12, 12)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (256, 12, 12) -> (256, 3, 3)
            nn.Conv2d(256, output_dim, 3),
            nn.LeakyReLU(leaky_slope),
            # (256, 3, 3) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(leaky_slope),
        ).apply(weights_init_xavier)

    def forward(self, x):
        num_batches, num_sequences, C, H, W = x.size()
        x = x.view(num_batches * num_sequences, C, H, W)
        x = self.net(x)
        x = x.view(num_batches, num_sequences, -1)

        return x


class LatentNetwork(nn.Module):

    def __init__(self, observation_shape, action_shape, feature_dim=256,
                 latent1_dim=32, latent2_dim=256, hidden_units=[256, 256],
                 leaky_slope=0.2):
        super(LatentNetwork, self).__init__()
        # NOTE: We encode x as the feature vector to share convolutional
        # part of the network with the policy.

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            latent1_dim, latent2_dim, hidden_units, leaky_slope=leaky_slope)
        # p(z1(t+1) | z2(t), a(t))
        self.latent1_prior = Gaussian(
            latent2_dim + action_shape[0], latent1_dim, hidden_units,
            leaky_slope=leaky_slope)
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_prior = Gaussian(
            latent1_dim + latent2_dim + action_shape[0], latent2_dim,
            hidden_units, leaky_slope=leaky_slope)

        
        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(
            feature_dim, latent1_dim, hidden_units, leaky_slope=leaky_slope)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.latent1_posterior = Gaussian(
            feature_dim + latent2_dim + action_shape[0], latent1_dim,
            hidden_units, leaky_slope=leaky_slope)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward_predictor = Gaussian(
            2 * latent1_dim + 2 * latent2_dim + action_shape[0],
            1, hidden_units, leaky_slope=leaky_slope)

        # feat(t) = x(t) : This encoding is performed deterministically.
        self.encoder = Encoder(
            observation_shape[0], feature_dim, leaky_slope=leaky_slope)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            latent1_dim + latent2_dim, observation_shape[0],
            std=np.sqrt(0.1), leaky_slope=leaky_slope)

    def sample_prior(self, actions_seq, init_features=None):
        ''' Sample from prior dynamics (with conditionning on the initial frames).
        Args:
            actions_seq   : (N, S, *action_shape) tensor of action sequences.
            init_features : (N, *) tensor of initial frames or None.
        Returns:
            latent1_samples : (N, S+1, L1) tensor of sampled latent vectors.
            latent2_samples : (N, S+1, L2) tensor of sampled latent vectors.
            latent1_dists   : (S+1) length list of (N, L1) distributions.
            latent2_dists   : (S+1) length list of (N, L2) distributions.
        '''
        num_sequences = actions_seq.size(1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t == 0:
                # Condition on initial frames.
                if init_features is not None:
                    # q(z1(0) | feat(0))
                    latent1_dist = self.latent1_init_posterior(init_features)
                    latent1_sample = latent1_dist.rsample()
                    # q(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_posterior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

                # Not conditionning.
                else:
                    # p(z1(0)) = N(0, I)
                    latent1_dist = self.latent1_init_prior(actions_seq[t])
                    latent1_sample = latent1_dist.rsample()
                    # p(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_prior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

            else:
                # p(z1(t) | z2(t-1), a(t-1))
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t-1], actions_seq[t-1]])
                latent1_sample = latent1_dist.rsample()
                # p(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_prior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)

    def sample_posterior(self, features_seq, actions_seq):
        ''' Sample from posterior dynamics.
        Args:
            features_seq : (N, S+1, 256) tensor of feature sequenses.
            actions_seq  : (N, S, *action_space) tensor of action sequenses.
        Returns:
            latent1_samples : (N, S+1, L1) tensor of sampled latent vectors.
            latent2_samples : (N, S+1, L2) tensor of sampled latent vectors.
            latent1_dists   : (S+1) length list of (N, L1) distributions.
            latent2_dists   : (S+1) length list of (N, L2) distributions.
        '''
        num_sequences = actions_seq.size(1)
        features_seq = torch.transpose(features_seq, 0, 1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t == 0:
                # q(z1(0) | feat(0))
                latent1_dist = self.latent1_init_posterior(features_seq[t])
                latent1_sample = latent1_dist.rsample()
                # q(z2(0) | z1(0))
                latent2_dist = self.latent2_init_posterior(latent1_sample)
                latent2_sample = latent2_dist.rsample()
            else:
                # q(z1(t) | feat(t), z2(t-1), a(t-1))
                latent1_dist = self.latent1_posterior(
                    [features_seq[t], latent2_samples[t-1], actions_seq[t-1]])
                latent1_sample = latent1_dist.rsample()
                # q(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_posterior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)





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
        
        q1 = self.Q1(states)
        q2 = self.Q2(states)

        
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
        action_probs = F.softmax(self.policy(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample()#.view(-1, 1)

        # avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs



class SACNet(BaseNet):
    def __init__(self, observation_shape, num_actions, 
                feature_dim=256, latent1_dim=32, latent2_dim=256,
                num_sequences=8, use_lstm=False, hidden_units=[256, 256],
                leaky_slope=0.2, initializer=weights_init_xavier):
        super(SACNet, self).__init__()
        self.num_actions = num_actions
        self.num_sequences = num_sequences
        self.observation_shape = observation_shape
        self.action_shape = 1

        self.latent = LatentNetwork(
            observation_shape, [num_actions], feature_dim,
            latent1_dim, latent2_dim, hidden_units, leaky_slope
            )
        self.policy = CateoricalPolicy(
            num_sequences * feature_dim
            + (num_sequences-1) * num_actions,
            num_actions,
            hidden_units=hidden_units)
        self.critic = TwinnedQNetwork(
            latent1_dim + latent2_dim,
            num_actions,
            hidden_units=hidden_units)
        self.critic_target = TwinnedQNetwork(
            latent1_dim + latent2_dim,
            num_actions,
            hidden_units=hidden_units)

    def initialize(self, inputs, batch_size):
        states = inputs["frame"]
        states = torch.ByteTensor(states).squeeze(0).squeeze(0).numpy()

        state_deque, action_deque = self.reset_deque(states)
        return dict(state_deque=state_deque, action_deque=action_deque, core_state=self.initial_state(batch_size))
    


    def reset_deque(self, states):
        
        state_deque = deque(maxlen=self.num_sequences)
        action_deque = deque(maxlen=self.num_sequences-1)
        
        for _ in range(self.num_sequences-1):
            state_deque.append(
                np.zeros(self.observation_shape, dtype=np.uint8))
            action_deque.append(
                np.zeros((self.num_actions), dtype=np.uint8))
        state_deque.append(states)
        return state_deque, action_deque
        

    def deque_to_batch(self, agent_state):
        # Convert deques to batched tensor.
        #state = np.array(self.state_deque, dtype=np.uint8)
        #state = torch.ByteTensor(
        #    state).unsqueeze(0).float() / 255.0

        state = torch.Tensor(np.array(agent_state["state_deque"], dtype=np.float32)).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            feature = self.latent.encoder(state)
            feature = feature.view(1, -1)

        action = np.array(agent_state["action_deque"], dtype=np.float32)
        action = torch.FloatTensor(action).view(1, -1)
        feature_action = torch.cat([feature, action], dim=-1)
        return feature_action

    def explore(self, agent_state):
        # Act with randomness
        feature_action = self.deque_to_batch(agent_state)
        with torch.no_grad():
            actions, action_probs, action_log_probs = self.policy.sample(feature_action)
        return actions.cpu().reshape(-1), action_log_probs

    def act(self, inputs, agent_state=dict(core_state=(), )):
        states = inputs["frame"]
        states = torch.ByteTensor(states).squeeze(0).squeeze(0).numpy()
        agent_state["state_deque"].append(states)
        
        action, action_log_probs = self.explore(agent_state)
        action = action[0]
        action_one_hot = np.eye(self.num_actions)[action]
        agent_state["action_deque"].append(action_one_hot)

        return (
            dict(action=action, action_log_probs=action_log_probs),
            dict(core_state=agent_state["core_state"], 
                 state_deque=agent_state["state_deque"],
                 action_deque=agent_state["action_deque"]),
        )
    
   