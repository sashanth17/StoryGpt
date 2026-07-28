import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer using PyTorch.
    Maps token IDs to dense vectors of size `embedding_dim`.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # PyTorch's nn.Embedding is highly optimized and supports GPU acceleration
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, context_size) containing token IDs.
        Returns:
            Tensor of shape (batch_size, context_size, embedding_dim)
        """
        return self.embedding(x)

class PositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding Layer for GPT.
    Provides positional context to the model since self-attention has no notion of sequence order.
    """
    def __init__(self, max_context_size, embedding_dim):
        super().__init__()
        self.max_context_size = max_context_size
        self.embedding_dim = embedding_dim
        
        # Learned positional embeddings for each position in the context window (standard for GPT-2/3)
        self.embedding = nn.Embedding(max_context_size, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, context_size). We just need the shape to generate positions.
        Returns:
            Tensor of shape (batch_size, context_size, embedding_dim)
        """
        batch_size, context_size = x.size()
        
        if context_size > self.max_context_size:
            raise ValueError(f"Context size {context_size} exceeds max {self.max_context_size}")
            
        # Create position IDs: [0, 1, 2, ..., context_size - 1]
        # Placed on the same device as the input tensor `x`
        positions = torch.arange(0, context_size, dtype=torch.long, device=x.device)
        
        # Expand positions to match the batch size: (batch_size, context_size)
        positions = positions.unsqueeze(0).expand(batch_size, context_size)
        
        # Get embeddings for these positions
        return self.embedding(positions)
