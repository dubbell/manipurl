import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class ReplayBatch:
    states : torch.Tensor
    next_states : torch.Tensor
    actions : torch.Tensor
    rewards : torch.Tensor
    done : torch.Tensor


class ReplayBuffer:
    states : torch.Tensor
    next_states : torch.Tensor
    actions : torch.Tensor
    rewards : torch.Tensor
    done : torch.Tensor

    def __init__(self, state_dim, action_dim, buffer_size = int(1e6)):
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.done = torch.zeros((buffer_size, 1), dtype=torch.uint8)

        self.max_size = buffer_size
        self.size = 0
        self.insert_pointer = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def insert(self, state, next_state, action, reward, done):
        self.states[self.insert_pointer] = torch.tensor(state) if not isinstance(state, torch.Tensor) else state
        self.next_states[self.insert_pointer] = torch.tensor(next_state) if not isinstance(next_state, torch.Tensor) else next_state
        self.actions[self.insert_pointer] = torch.tensor(action) if not isinstance(action, torch.Tensor) else action
        self.rewards[self.insert_pointer] = torch.tensor(reward) if not isinstance(reward, torch.Tensor) else reward
        self.done[self.insert_pointer] = torch.tensor(done) if not isinstance(done, torch.Tensor) else done

        self.size = min(self.size + 1, self.max_size)
        self.insert_pointer = (self.insert_pointer + 1) % self.max_size


    def sample(self, batch_size = 256):
        if batch_size < self.size:
            sampled_idxs = np.random.choice(np.arange(self.size), size=batch_size, replace=False)
        else:
            sampled_idxs = np.arange(self.size)
        return ReplayBatch(
            self.states[sampled_idxs].to(device=self.device),
            self.next_states[sampled_idxs].to(device=self.device),
            self.actions[sampled_idxs].to(device=self.device),
            self.rewards[sampled_idxs].to(device=self.device),
            self.done[sampled_idxs].to(device=self.device))

