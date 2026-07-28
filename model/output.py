import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    """
    The language modeling head that converts the final embeddings into vocabulary logits.
    """
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        # Standard GPT LayerNorm before the final projection
        self.ln_f = nn.LayerNorm(embedding_dim)
        # Final linear layer to project to vocab size
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
