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
    embeddings : torch.Tensor


class ReplayBuffer:
    states : torch.Tensor
    next_states : torch.Tensor
    actions : torch.Tensor
    rewards : torch.Tensor
    done : torch.Tensor

    def __init__(self, state_dim, action_dim, buffer_size = int(1e6), store_embeddings = False, emb_dim = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((buffer_size, 1), dtype=torch.uint8, device=self.device)

        self.store_embeddings = store_embeddings
        if store_embeddings:
            self.embeddings = torch.zeros((buffer_size, emb_dim), dtype=torch.float32, device=self.device)

        self.max_size = buffer_size
        self.size = 0
        self.insert_pointer = 0


    def insert(self, state, next_state, action, reward, done, embedding = None):
        self.states[self.insert_pointer].copy_(torch.as_tensor(state).pin_memory(), non_blocking=True)
        self.next_states[self.insert_pointer].copy_(torch.as_tensor(next_state).pin_memory(), non_blocking=True)
        self.actions[self.insert_pointer].copy_(torch.as_tensor(action).pin_memory(), non_blocking=True)
        self.rewards[self.insert_pointer].copy_(torch.as_tensor(reward).pin_memory(), non_blocking=True)
        self.done[self.insert_pointer].copy_(torch.as_tensor(done).pin_memory(), non_blocking=True)
        
        if self.store_embeddings:
            self.embeddings[self.insert_pointer] \
                .copy_(torch.as_tensor(embedding).pin_memory(), non_blocking=True)

        self.size = min(self.size + 1, self.max_size)
        self.insert_pointer = (self.insert_pointer + 1) % self.max_size


    def sample(self, batch_size = 256):
        if batch_size < self.size:
            sampled_idxs = torch.randperm(self.size, device=self.device)[:batch_size]
        else:
            sampled_idxs = torch.arange(self.size, device=self.device)
        
        torch.cuda.synchronize()  # since insertions are non-blocking

        return ReplayBatch(
            self.states[sampled_idxs],
            self.next_states[sampled_idxs],
            self.actions[sampled_idxs],
            self.rewards[sampled_idxs],
            self.done[sampled_idxs],
            None if not self.store_embeddings else self.embeddings[sampled_idxs])