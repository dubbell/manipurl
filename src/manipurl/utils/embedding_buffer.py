import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingBatch:
    pos_embeddings : torch.Tensor
    prev_pos_embeddings : torch.Tensor
    neg_embeddings : torch.Tensor
    prev_neg_embeddings : torch.Tensor


class EmbeddingBuffer:

    def __init__(self, emb_dim = 512, buffer_size = int(1e5)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = buffer_size
        
        self.pos_buffer = torch.zeros((buffer_size, emb_dim), dtype=torch.float32, device=self.device)
        self.neg_buffer = torch.zeros((buffer_size, emb_dim), dtype=torch.float32, device=self.device)

        self.pos_ptr = 0
        self.neg_ptr = 0
        
        # -1 means non-filled slot
        self.pos_ep_steps = torch.zeros(buffer_size, dtype=torch.uint8, device=self.device) - 1
        self.neg_ep_steps = torch.zeros(buffer_size, dtype=torch.uint8, device=self.device) - 1

    
    def insert_episode(self, episode : torch.Tensor, success : bool):
        buffer, ptr, ep_steps = (self.pos_buffer, self.pos_ptr, self.pos_ep_steps) if success \
                      else (self.neg_buffer, self.neg_ptr, self.neg_ep_steps)
        episode_length = episode.shape[0]
        self.pos_ep_count += int(success)
        self.neg_ep_count += 1 - int(success)

        cur_ep_steps = torch.arange(episode_length, device=torch.device('cuda'))

        # normal insertion
        if ptr + episode_length <= self.size:
            # store embeddings
            buffer[ptr:ptr+episode_length].copy_(episode, non_blocking=True)
            # store corresponding episode environment step
            ep_steps[ptr:ptr+episode_length].copy_(cur_ep_steps, non_blocking=True)

        # wrap-around
        else:
            # size of first and second part of wrap-around
            first = self.size - ptr
            second = episode_length - first
            # store embeddings
            buffer[ptr:].copy_(episode[:first], non_blocking=True)
            buffer[:second].copy_(episode[first:], non_blocking=True)
            # store corresponding episode environment step
            ep_steps[ptr:].copy_(cur_ep_steps[:first], non_blocking=True)
            ep_steps[:second].copy(cur_ep_steps[first:], non_blocking=True)
 
        # update pointer
        if success:
            self.pos_ptr = (ptr + episode_length) % self.size
        else:
            self.neg_ptr = (ptr + episode_length) % self.size

    

    def sample_batch(self, batch_size = 256, neg_gap = None, pos_gap = None):
        torch.cuda.synchronize()  # since insertions are non-blocking

        # mask positive indices: (not within `pos_gap` of episode start) and (not non-filled slot in buffer)
        pos_mask = (self.pos_ep_steps != -1) & (1 if pos_gap is None else (self.pos_ep_steps >= pos_gap))
        pos_idxs = torch.where(pos_mask)
        sampled_pos_idxs = pos_idxs[torch.randperm(len(pos_idxs))[:batch_size]]

        # mask negative indices: (not within `neg_gap` of episode start) and (not non-filled slot in buffer)
        neg_mask = (self.neg_ep_steps != -1) & (1 if neg_gap is None else (self.neg_ep_steps >= neg_gap))
        neg_idxs = torch.where(neg_mask)
        sampled_neg_idxs = neg_idxs[torch.randperm(len(neg_idxs))[:batch_size]]

        pos_embeddings = self.pos_buffer[sampled_pos_idxs]
        prev_pos_embeddings = None if pos_gap is None else self.pos_buffer[(sampled_pos_idxs - pos_gap) % self.size]

        neg_embeddings = self.neg_buffer[sampled_neg_idxs]
        prev_neg_embeddings = None if neg_gap is None else self.neg_buffer[(sampled_neg_idxs - neg_gap) % self.size]

        return EmbeddingBatch(pos_embeddings, prev_pos_embeddings, neg_embeddings, prev_neg_embeddings)