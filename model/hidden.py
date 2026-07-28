import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    The hidden layer (MLP) within the transformer block.
    Adds non-linearity and expands/contracts the representation.
    """
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
