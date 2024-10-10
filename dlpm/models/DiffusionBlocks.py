import os
import numpy as np
import torch
import scipy

import torch.nn as nn
import torch.optim as optim
import shutil

''' Blocks; either simple, time conditioned, or time and a_t's conditioned'''

# can add batch norm and dropout
class DiffusionBlock(nn.Module):
    def __init__(self, 
                 nunits, 
                 dropout_rate, 
                 skip_connection, 
                 group_norm,
                 activation = nn.SiLU):
        super(DiffusionBlock, self).__init__()
        
        self.skip_connection = skip_connection # boolean
        self.act = activation(inplace=False)
        
        # for the moment, implementing as batch norm
        self.group_norm1 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.group_norm2 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.mlp_1 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm1)
        self.mlp_2 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm2)
        
    def forward(self, x: torch.Tensor):
        if self.skip_connection:
            x_skip = x
        x = self.act(self.mlp_1(x))
        x = self.mlp_2(x)
        if self.skip_connection:
            x= x + x_skip
        return self.act(x)
    

# can add batch norm and dropout
class DiffusionBlockTime(nn.Module):
    def __init__(self, 
                 nunits, 
                 time_embedding_size, 
                 dropout_rate, 
                 skip_connection, 
                 group_norm,
                 activation = nn.SiLU):
        super(DiffusionBlockTime, self).__init__()
        
        self.skip_connection = skip_connection # boolean
        self.act = activation(inplace=False)
        
        # for the moment, implementing as batch norm
        self.group_norm1 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.group_norm2 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.mlp_1 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm1)
        # remove dropout from embedding?
        self.t_proj = nn.Sequential(self.dropout, 
                                    nn.Linear(time_embedding_size, nunits), 
                                    self.act)
        self.mlp_2 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm2)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        if self.skip_connection:
            x_skip = x
        x = self.act(self.mlp_1(x))
        x += self.t_proj(t_emb)
        x = self.mlp_2(x)
        if self.skip_connection:
            x = x + x_skip
        return self.act(x)

# can add batch norm and dropout
class DiffusionBlockConditioned(nn.Module):
    def __init__(self, 
                 nunits, 
                 dropout_rate, 
                 skip_connection, 
                 group_norm,
                 time_emb_size = False, 
                 a_emb_size = False,
                 activation = nn.SiLU):
        super(DiffusionBlockConditioned, self).__init__()
        
        self.skip_connection = skip_connection # boolean
        self.act = activation(inplace=False)
        self.time = time_emb_size != False
        self.a = a_emb_size != False
        
        # for the moment, implementing as batch norm
        self.group_norm1 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.group_norm2 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.mlp_1 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm1)
        # remove dropout from embedding?
        if self.time:
            self.t_proj = nn.Sequential(self.dropout, 
                                        nn.Linear(time_emb_size, nunits), 
                                        self.act)
        if self.a:
            self.a_proj = nn.Sequential(self.dropout, 
                                        nn.Linear(a_emb_size, nunits), 
                                        self.act)
        self.mlp_2 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm2)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, a_emb:torch.Tensor):
        if self.skip_connection:
            x_skip = x
        x = self.act(self.mlp_1(x))
        if self.time:
            x += self.t_proj(t_emb) 
        if self.a:
            x += self.a_proj(a_emb)
        x = self.mlp_2(x)
        if self.skip_connection:
            x = x + x_skip
        return self.act(x)