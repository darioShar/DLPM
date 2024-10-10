import os
import numpy as np
import torch
import scipy

import torch.nn as nn
import torch.optim as optim
import shutil

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_time_steps, embedding_size, device, n = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2).to(device)
        k = torch.arange(max_time_steps).unsqueeze(dim=1).to(device)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False).to(device)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if len(t.shape) > 1:
            return self.pos_embeddings[t, :].squeeze()
        return self.pos_embeddings[t, :]

# Just a simple MLP
class LearnableEmbedding(nn.Module):
    def __init__(self, in_features, out_features, device) :
        super().__init__()
        
        self.mlp = nn.Linear(in_features, out_features).to(device)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)
    