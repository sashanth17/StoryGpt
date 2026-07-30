import torch
import torch.nn as nn
from torch.nn import functional as F

from model.embedding import TokenEmbedding, PositionalEmbedding
from model.attention import CausalSelfAttention
from model.hidden import FeedForward
from model.output import OutputLayer

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_size, dropout=0.1):
        super().__init__()
        # Layer norm before attention
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, context_size, dropout)
        
        # Layer norm before feedforward
        self.ln_2 = nn.LayerNorm(embedding_dim)
        hidden_dim = 4 * embedding_dim
        self.ff = FeedForward(embedding_dim, hidden_dim, dropout)
        
    def forward(self, x):
        # Pre-LayerNorm architecture with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x

class tinyStory(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.context_size = context_size
        
        # 1. Embeddings
        self.token_emb = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_emb = PositionalEmbedding(context_size, embedding_dim)
        
        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, context_size, dropout)
            for _ in range(num_layers)
        ])
        
        # 3. Output Layer (LM Head)
        self.output = OutputLayer(embedding_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Crop context if it exceeds the model's max context size
        if T > self.context_size:
            idx = idx[:, -self.context_size:]
            if targets is not None:
                targets = targets[:, -self.context_size:]
            
        x = self.token_emb(idx) + self.pos_emb(idx)
        
        for block in self.blocks:
            x = block(x)
            
        logits = self.output(x)
        
        if targets is None:
            return logits, None
        
        # Calculate loss if targets are provided
        B, T, C = logits.size()
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """
        Autoregressively generates new tokens given a conditioning sequence `idx`.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context size
            idx_cond = idx if idx.size(1) <= self.context_size else idx[:, -self.context_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Pluck the logits for the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_id is not None and idx_next.item() == eos_id:
                break
                
        return idx
