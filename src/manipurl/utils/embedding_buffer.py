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
        self.size = buffer_size
        
        self.pos_buffer = torch.zeros((buffer_size, emb_dim), dtype=torch.float32).cuda()
        self.neg_buffer = torch.zeros((buffer_size, emb_dim), dtype=torch.float32).cuda()

        self.pos_ptr = torch.tensor(0).cuda()
        self.neg_ptr = torch.tensor(0).cuda()

        self.pos_ep_count = 0
        self.neg_ep_count = 0

        self.pos_part = torch.zeros(buffer_size).cuda()
        self.neg_part = torch.zeros(buffer_size).cuda()

    
    def insert_episode(self, episode : torch.Tensor, success : bool):
        buffer, ptr = (self.pos_buffer, self.pos_ptr) if success \
                      else (self.neg_buffer, self.neg_ptr)
        episode_length = episode.shape[0]
        self.pos_ep_count += int(success)
        self.neg_ep_count += 1 - int(success)

        # normal insertion
        if ptr + episode_length <= self.size:
            buffer[ptr:ptr+episode_length].copy_(episode, non_blocking=True)

            if success:
                self.pos_part[ptr:ptr+episode_length] = self.pos_ep_count
            else:    
                self.neg_part[ptr:ptr+episode_length] = self.neg_ep_count

        # wrap-around
        else:
            first = self.size - ptr
            second = episode_length - first
            buffer[ptr:self.size].copy_(episode[:first], non_blocking=True)
            buffer[:second].copy_(episode[first:], non_blocking=True)
            
            if success:
                self.pos_part[ptr:self.size] = self.pos_ep_count
                self.pos_part[:second] = self.pos_ep_count
            else:
                self.neg_part[ptr:self.size] = self.neg_ep_count
                self.neg_part[:second] = self.neg_ep_count

        ptr.copy_((ptr+episode_length) % self.size)
    

    def sample_batch(self, batch_size = 256, neg_gap = None, pos_gap = None):

        pos_mask = (self.pos_part != 0)
        if pos_gap is not None:
            pos_lagged_part = self.pos_part[(torch.arange(self.size) - pos_gap) % self.size]
            pos_mask = pos_mask & (self.pos_part == pos_lagged_part)

        pos_idxs = torch.where(pos_mask)[0]
        sampled_pos_idxs = pos_idxs[torch.randperm(len(pos_idxs))[:batch_size]]

        neg_mask = (self.neg_part != 0)
        if neg_gap is not None:
            neg_lagged_part = self.neg_part[(torch.arange(self.size) - neg_gap) % self.size]
            neg_mask = neg_mask & (self.neg_part == neg_lagged_part)

        neg_idxs = torch.where(neg_mask)[0]
        sampled_neg_idxs = neg_idxs[torch.randperm(len(neg_idxs))[:batch_size]]

        pos_embeddings = self.pos_buffer[sampled_pos_idxs]
        prev_pos_embeddings = None if pos_gap is None else self.pos_buffer[(sampled_pos_idxs - pos_gap) % self.size]

        neg_embeddings = self.neg_buffer[sampled_neg_idxs]
        prev_neg_embeddings = None if neg_gap is None else self.neg_buffer[(sampled_neg_idxs - neg_gap) % self.size]

        return EmbeddingBatch(pos_embeddings, prev_pos_embeddings, neg_embeddings, prev_neg_embeddings)