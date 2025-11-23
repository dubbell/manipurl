import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class Actor(nn.Module):

    def __init__(self, in_features, out_features, max_action = 1.0, min_scale = 1e-3):
        super().__init__()
        self.max_action = max_action
        self.min_scale = min_scale
        self.body = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU())
        
        self.mu = nn.Linear(256, out_features)
        self.std = nn.Linear(256, out_features)

    def forward(self, states):
        proj_states = self.body(states)
        mu = self.mu(proj_states)
        std = nn.functional.softplus(self.std(proj_states)) + self.min_scale

        action_normal_distr = Normal(mu, std)

        # unconstrained action z
        z = action_normal_distr.rsample()  

        # constrained actions
        mean_action = torch.tanh(mu) * self.max_action
        sampled_action = torch.tanh(z) * self.max_action

        # log probability for unconstrained action
        log_prob_z = action_normal_distr.log_prob(z).sum(-1, keepdim=True)

        # loss term for tanh transformation, with correction to avoid numerical instability
        log_det_jacobian = 2 * (np.log(2) - z - nn.functional.softplus(-2 * z)).sum(-1, keepdim=True)
      
        if self.max_action != 1.0:
             log_det_jacobian += torch.log(self.max_action) * z.shape[-1]

        log_prob = log_prob_z - log_det_jacobian

        return mean_action, sampled_action, log_prob


class Critic(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        
    def forward(self, states, actions):
        state_actions = torch.concatenate([states, actions], axis=-1)
        return self.net(state_actions)


class DoubleCritic(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.critic1 = Critic(in_features)
        self.critic2 = Critic(in_features)

    def forward(self, states, actions):
        return self.critic1(states, actions), self.critic2(states, actions)