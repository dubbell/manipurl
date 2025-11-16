import torch



class SACAgent:
    def __init__(self, obs_dim, act_dim, tau = 0.005, gamma = 0.99, lr = 3e-4):
        