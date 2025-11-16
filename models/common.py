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

    def forward(self, x):
        proj_x = self.body(x)
        return self.mu(proj_x), self.std(proj_x)


class Critic(nn.Module):

    def __init__(self, in_features, out_features):
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(out_features))
        
    def forward(self, x):
        return self.net(x)