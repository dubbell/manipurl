import torch
import torch.nn as nn



class Actor(nn.Module):

    def __init__(self, in_features, out_features):
        self.body = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU())
        
        self.mu = nn.Linear(out_features)
        self.std = nn.Linear(out_features)

    def forward(self, states):
        proj_states = self.body(states)
        return self.mu(proj_states), self.std(proj_states)


class Critic(nn.Module):

    def __init__(self, in_features):
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(1))
        
    def forward(self, states, actions):
        state_actions = torch.concatenate([states, actions], axis=-1)
        return self.net(state_actions)


class DoubleCritic(nn.Module):

    def __init__(self, in_features):
        self.critic1 = Critic(in_features)
        self.critic2 = Critic(in_features)


    def forward(self, states, actions):
        state_actions = torch.concatenate([states, actions], axis=-1)
        return self.critic1(state_actions), self.critic2(state_actions)