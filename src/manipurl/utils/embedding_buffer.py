import torch
import numpy as np


class EmbeddingBuffer:

    def __init__(self, emb_dim = 512, buffer_size = int(1e5)):
        self.size = buffer_size
        
        self.pos_buffer = torch.zeros((buffer_size, emb_dim), dtype=torch.float32).cuda()
        self.neg_buffer = torch.zeros((buffer_size, emb_dim), dtype=torch.float32).cuda()

        self.pos_ptr = 0
        self.neg_ptr = 0

        self.pos_starts = []
        self.neg_starts = []

    
    def insert_episode(self, episode : torch.Tensor, success : bool, ):
        buffer, ptr, starts = (self.pos_buffer, self.pos_ptr, self.pos_starts) if success \
                      else (self.neg_buffer, self.neg_ptr, self.neg_starts)
        episode_length = episode.shape[0]

        if ptr + episode_length <= self.buffer_size:
            buffer[ptr:ptr+episode_length].copy_(episode, non_blocking=True)
            starts = list(filter(lambda start: start < ptr and start >= ptr + episode_length, starts))
        else:
            remainder = self.buffer_size - (ptr + episode_length)
            buffer[ptr:self.buffer_size].copy_(episode[:self.buffer_size-remainder], non_blocking=True)
            buffer[:remainder].copy_(episode[self.buffer_size-remainder:], non_blocking=True)
