import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    The causal mask ensures that a token can only attend to previous tokens and itself.
    """
    def __init__(self, embedding_dim, num_heads, context_size, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        
        # Key, Query, Value projections in a single batch
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim)
        
        # Output projection
        self.c_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask (lower triangular matrix)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(context_size, context_size)).view(1, 1, context_size, context_size)
        )

    def forward(self, x):
        B, T, C = x.size() # batch_size, context_size, embedding_dim
        
        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        
        # Reshape to (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Causal self-attention: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply the causal mask (mask out future tokens)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Multiply attention weights by values
        y = att @ v # (B, num_heads, T, head_dim)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
