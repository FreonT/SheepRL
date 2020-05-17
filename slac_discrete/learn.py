import threading
import typing

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.distributions.kl import kl_divergence


Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def calc_policy_loss(self, latents, feature_actions, weights, alpha):
    # Re-sample actions to calculate expectations of Q.
    _, action_probs, log_action_probs = self.policy.sample(feature_actions)
    # E[Q(z(t), a(t))]
    q1, q2 = self.critic(latents)

    # expectations of entropies
    entropies = -torch.sum(
        action_probs * log_action_probs, dim=1, keepdim=True)
    # expectations of Q
    q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

    # Policy objective is maximization of (Q + alpha * entropy) with
    # priority weights.
    policy_loss = (weights * (- q - alpha * entropies)).mean()
    

    return policy_loss, entropies

def calc_critic_loss(model, latents, next_latents, actions, next_feature_actions, rewards, weights, alpha, gamma_n):
        
        # Q(z(t), a(t))
        curr_q1, curr_q2 = model.critic(latents)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())

        # E[Q(z(t+1), a(t+1)) + alpha * H(pi)]
        with torch.no_grad():
            next_actions, next_entropies, _ =\
                model.policy.sample(next_feature_actions)
            next_q1, next_q2 = model.critic(next_latents)
            next_q1 = next_q1.gather(1, next_actions.unsqueeze(1).long())
            next_q2 = next_q2.gather(1, next_actions.unsqueeze(1).long())
            next_q = torch.min(next_q1, next_q2)# + alpha * next_entropies
        # r(t) + gamma * E[Q(z(t+1), a(t+1)) + alpha * H(pi)]
        target_q = rewards.view_as(next_q) + gamma_n * next_q

        # Critic losses are mean squared TD errors.
        q1_loss = 0.5 * torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = 0.5 * torch.mean((curr_q2 - target_q).pow(2))

        return q1_loss, q2_loss

def calc_entropy_loss(log_alpha, target_entropy, entropy, weights):
    # Intuitively, we increse alpha when entropy is less than target
    # entropy, vice versa.
    entropy_loss = -torch.mean(
        log_alpha * (target_entropy - entropy).detach()
        * weights)
    return entropy_loss


def calc_kl_divergence(p_list, q_list):
    assert len(p_list) == len(q_list)

    kld = 0.0
    for i in range(len(p_list)):
        # (N, L) shaped array of kl divergences.
        _kld = kl_divergence(p_list[i], q_list[i])
        # Average along batches, sum along sequences and elements.
        kld += _kld.mean(dim=0).sum()

    return kld

def calc_latent_loss(flags, model, images_seq, actions_seq, rewards_seq, dones_seq, images):
    features_seq = get_feature_seq(flags, model, images)
    #features_seq = model.latent.encoder(images_seq)

    #features_seq = features_seq.view(num_batches, num, num_sequences, -1)

    # Sample from posterior dynamics.
    (latent1_post_samples, latent2_post_samples),\
        (latent1_post_dists, latent2_post_dists) =\
        model.latent.sample_posterior(features_seq, actions_seq)
    # Sample from prior dynamics.
    (latent1_pri_samples, latent2_pri_samples),\
        (latent1_pri_dists, latent2_pri_dists) =\
        model.latent.sample_prior(actions_seq)

    # KL divergence loss.
    kld_loss = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

    # Log likelihood loss of generated observations.
    images_seq_dists = model.latent.decoder(
        [latent1_post_samples, latent2_post_samples])
    
    log_likelihood_loss = images_seq_dists.log_prob(
        images_seq).mean(dim=0).sum()
    
    # Log likelihood loss of genarated rewards.
    rewards_seq_dists = model.latent.reward_predictor([
        latent1_post_samples[:, :-1],
        latent2_post_samples[:, :-1],
        actions_seq, latent1_post_samples[:, 1:],
        latent2_post_samples[:, 1:]])
    reward_log_likelihoods =\
        rewards_seq_dists.log_prob(rewards_seq) * (1.0 - dones_seq)
    reward_log_likelihood_loss = reward_log_likelihoods.mean(dim=0).sum()

    latent_loss =\
        kld_loss - log_likelihood_loss - reward_log_likelihood_loss

    return latent_loss, kld_loss, log_likelihood_loss, reward_log_likelihood_loss


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def unbatch(flags, batch):
    images = batch["frame"]
    
    actions = batch["action"][1:]
    #action_log_probs = batch["action_log_probs"][1:]

    rewards = batch["reward"][1:]
    if flags.reward_clipping == "abs_one":
        rewards = torch.clamp(rewards, -1, 1)
    elif flags.reward_clipping == "none":
        rewards = rewards
    dones = batch["done"][1:].float()

    return images, actions, rewards, dones

def get_feature_seq(flags, model, images):
    
    features = model.latent.encoder(images)

    batch_size_1 = flags.unroll_length + 1  - flags.num_sequences
    batch_size_2 = flags.batch_size

    features_seq = torch.Tensor(np.empty((
        batch_size_1, batch_size_2, flags.num_sequences+1, 256),
        dtype=np.uint8)).to(flags.device)

    for i_1 in range(batch_size_1):
        for i_2 in range(batch_size_2):
            features_seq[i_1, i_2, ...]  = features[i_1:i_1+flags.num_sequences+1, i_2]

    num_batches, num, num_sequences, C = features_seq.size()
    features_seq = features_seq.view(num_batches * num, num_sequences, C)
    return features_seq

def get_sequence(flags, batch):
    
    images, actions, rewards, dones = unbatch(flags, batch)
    
    batch_size_1 = flags.unroll_length + 1  - flags.num_sequences
    batch_size_2 = flags.batch_size

    empty_im = np.empty((
        batch_size_1, batch_size_2, flags.num_sequences+1, *flags.observation_shape),
        dtype=np.uint8)
    images_seq = torch.Tensor(empty_im)
    images_seq = images_seq.to(flags.device)
    actions_one_hot_seq = torch.Tensor(np.empty((
        batch_size_1, batch_size_2, flags.num_sequences, flags.action_shape),
        dtype=np.float32)).to(flags.device)
    actions_step = torch.Tensor(np.empty((
        batch_size_1, batch_size_2, 1),
        dtype=np.float32)).to(flags.device)
    rewards_seq = torch.Tensor(np.empty((
        batch_size_1, batch_size_2, flags.num_sequences), dtype=np.float32)).to(flags.device)
    dones_seq = torch.Tensor(np.empty((
        batch_size_1, batch_size_2, flags.num_sequences), dtype=np.float32)).to(flags.device)
    rewards_step = torch.Tensor(np.empty((
        batch_size_1, batch_size_2, 1), dtype=np.float32)).to(flags.device)

    for i_1 in range(batch_size_1):
        for i_2 in range(batch_size_2):
            images_seq[i_1, i_2, ...]  = images[i_1:i_1+flags.num_sequences+1, i_2]
            actions_one_hot_seq[i_1, i_2, ...] = torch.eye(flags.action_shape)[actions[i_1:i_1+flags.num_sequences, i_2]]
            actions_step[i_1, i_2, ...] = actions[i_1+flags.num_sequences-1, i_2]
            rewards_seq[i_1, i_2, ...] = rewards[i_1:i_1+flags.num_sequences, i_2]
            dones_seq[i_1, i_2, ...]   = dones[i_1:i_1+flags.num_sequences, i_2]
            rewards_step[i_1, i_2, ...] = rewards[i_1+flags.num_sequences-1, i_2]

    num_batches, num, num_sequences, C, H, W = images_seq.size()
    images_seq = images_seq.view(num_batches * num, num_sequences, C, H, W).float()/255

    num_batches, num, num_sequences, C = actions_one_hot_seq.size()
    actions_one_hot_seq = actions_one_hot_seq.view(num_batches * num, num_sequences, C)

    num_batches, num, num_sequences = rewards_seq.size()
    rewards_seq = rewards_seq.view(num_batches * num, num_sequences, 1)

    num_batches, num, num_sequences = dones_seq.size()
    dones_seq = dones_seq.view(num_batches * num, num_sequences, 1)

    num_batches, num, num_sequences = actions_step.size()
    actions_step = actions_step.view(num_batches * num, num_sequences)

    num_batches, num, num_sequences = rewards_step.size()
    rewards_step = rewards_step.view(num_batches * num, num_sequences, 1)
    
    images = images.float()/255

    return images_seq, actions_one_hot_seq, actions_step, rewards_seq, dones_seq, rewards_step, images

def create_feature_actions(features_seq, actions_seq):
    N = features_seq.size(0)

    # sequence of features
    f = features_seq[:, :-1].view(N, -1)
    n_f = features_seq[:, 1:].view(N, -1)
    # sequence of actions
    a = actions_seq[:, :-1].view(N, -1)
    n_a = actions_seq[:, 1:].view(N, -1)

    # feature_actions
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)

    return fa, n_fa


def learn(
    flags,
    actor_model,
    learner_model,
    batch,
    initial_agent_state,
    optim,
    step,
    lock=threading.Lock(),  # noqa: B008
    weights=1,
):
    """Performs a learning (optimization) step."""
    with lock:
        
        images_seq, actions_one_hot_seq, actions, rewards_seq, dones_seq, rewards, images = get_sequence(flags, batch)
        
        latent_loss, kld_loss, log_likelihood_loss, reward_log_likelihood_loss = calc_latent_loss(
            flags, learner_model, images_seq, actions_one_hot_seq, rewards_seq, dones_seq, images)
        
        
        update_params(
            optim.latent_optim, learner_model.latent, latent_loss, flags.grad_norm_clipping)
        
        
        # NOTE: Don't update the encoder part of the policy here.
        with torch.no_grad():
            # f(1:t+1)
            features_seq = get_feature_seq(flags, learner_model, images)
            #features_seq = learner_model.latent.encoder(images_seq)
            latent_samples, _ = learner_model.latent.sample_posterior(
                features_seq, actions_one_hot_seq)

        # z(t), z(t+1)
        latents_seq = torch.cat(latent_samples, dim=-1)
        latents = latents_seq[:, -2]
        next_latents = latents_seq[:, -1]
        # a(t)
        
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_actions, next_feature_actions =\
            create_feature_actions(features_seq, actions_one_hot_seq)

        q1_loss, q2_loss = calc_critic_loss(
            learner_model, latents, next_latents, actions, next_feature_actions, rewards, weights, optim.alpha, flags.discounting**flags.multi_step)
        policy_loss, entropies = calc_policy_loss(
            learner_model, latents, feature_actions, weights, optim.alpha)

        
        update_params(
            optim.q1_optim, learner_model.critic.Q1, q1_loss, flags.grad_norm_clipping)
        update_params(
            optim.q2_optim, learner_model.critic.Q2, q2_loss, flags.grad_norm_clipping)
        update_params(
            optim.policy_optim, learner_model.policy, policy_loss, flags.grad_norm_clipping)



        if optim.entropy_tuning:
            entropy_loss = calc_entropy_loss(optim.log_alpha, optim.target_entropy, entropies, weights)
            update_params(optim.alpha_optim, None, entropy_loss)
            optim.alpha = optim.log_alpha.exp()
        
        actor_model.policy.load_state_dict(learner_model.policy.state_dict())
        actor_model.latent.load_state_dict(learner_model.latent.state_dict())
        
        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "00_episode_returns": tuple(episode_returns.cpu().numpy()),
            "01_mean_episode_return": torch.mean(episode_returns).item(),
            "02_q1_loss": q1_loss.item(),
            "03_q2_loss": q2_loss.item(),
            "04_policy_loss": policy_loss.item(),
            "05_entropy_loss": entropy_loss.item(),
            "06_alpha": optim.alpha.item(),
            "07_entoropy": entropies.mean().item(),
            "08_latent_loss":latent_loss.item(),
            "09_kld_loss":kld_loss.item(),
            "10_log_likelihood_loss":log_likelihood_loss.item(),
            "11_reward_log_likelihood_loss":reward_log_likelihood_loss.item(),
        }
        return stats
        

        


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        action_log_probs=dict(size=(T + 1, num_actions), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


class optimizer:
    def __init__(self, flags, model, entropy_tuning=True):
        self.policy_optim = Adam(model.policy.parameters(), lr=flags.learning_rate)
        self.q1_optim = Adam(model.critic.Q1.parameters(), lr=flags.learning_rate)
        self.q2_optim = Adam(model.critic.Q2.parameters(), lr=flags.learning_rate)
        self.latent_optim = Adam(model.latent.parameters(), lr=flags.latent_learning_rate)

        self.entropy_tuning = entropy_tuning
        self.device = flags.device

        if self.entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                flags.action_shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=flags.learning_rate)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)


