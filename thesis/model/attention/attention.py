from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

class SelfAttention(nn.Module):
    """ Self Attention module.

    Args:
        dim: embedding dimension
        num_heads: number of parallel attention heads
        qkv_bias: If True, add a learnable bias to query, key, value. Defaults to False.
        attn_drop: Dropout ratio of attention weight. Defaults to 0.
        proj_drop: Dropout ratio of output. Defaults to 0.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        # check that dim can be split evenly across heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads" 
        self.num_heads = num_heads
        
        # compute the dimension of each head
        head_dim = dim // num_heads
        # scale factor for attn scores, to prevent them from growing too large
        self.scale = head_dim ** -0.5
        # one big linear layer to compute query, key, value in one go
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # dropout for attn weights
        self.attn_drop = nn.Dropout(attn_drop)
        # combine attn head outputs
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # shape of input: (batch_size, seq_length, dim)
        # compute query, key, value in one go and reshape for multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [QKV, batch, heads, tokens, head_dim]
        # split qkv into separate tensors for query, key, value
        q, k, v = qkv.unbind(0)
        
        # compute attn scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # turn attn scores into probabilities
        attn = attn.softmax(dim=-1)
        # randomly drop some attn weights during training for regularization.
        attn = self.attn_drop(attn)
        
        # compute attn heads output and merge them
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
   

class CrossAttention(nn.Module):

    def __init__(self, query_dim, kv_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim_q = query_dim // num_heads
        self.scale = head_dim_q ** -0.5

        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.k = nn.Linear(kv_dim, kv_dim, bias=qkv_bias)
        self.v = nn.Linear(kv_dim, kv_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        
        self.proj = nn.Linear(kv_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, return_attn=True):
        B, Nq, Cq = query.shape
        _, Nk, Ck = key.shape
        
        head_dim_kv = Ck // self.num_heads

        q = self.q(query).reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, Nk, self.num_heads, head_dim_kv).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, Nk, self.num_heads, head_dim_kv).permute(0, 2, 1, 3)

        # attn scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn if return_attn else None

        # attn probabilities
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # compute attn output and merge heads
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, Ck)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn:
            # attn output & attn scores (before softmax)
            return x, attn_weights
        else:
            return x
    
