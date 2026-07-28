import torch
from dataLoader import create_dataloader
import os
import sys

from model.embedding import TokenEmbedding, PositionalEmbedding
from model.attention import CausalSelfAttention
from model.hidden import FeedForward
from model.output import OutputLayer

if __name__ == "__main__":
    bin_file = "validation.bin"

    if not os.path.exists(bin_file):
        print(f"Warning: {bin_file} not found. Please run prepare_data.py first.")
        sys.exit(1)

    # Setup hyperparameters
    BLOCK_SIZE = 256   # Context window (T)
    BATCH_SIZE = 4     # Batch size (B)
    STRIDE = 256       # no overlap
    VOCAB_SIZE = 4097  # Base vocab (4096 items) + EOS token (1)
    
    # Model Hyperparameters
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    HIDDEN_DIM = 4 * EMBEDDING_DIM
    DROPOUT = 0.1

    # Create the loader
    print("\nCreating DataLoader...")
    loader = create_dataloader(
        bin_file, 
        block_size=BLOCK_SIZE, 
        batch_size=BATCH_SIZE, 
        stride=STRIDE, 
        vocab_size=VOCAB_SIZE,
        num_workers=0
    )

    # Fetch the very first batch
    X, Y = next(iter(loader))

    print(f"\n--- Batch Info ---")
    print(f"Input Shape (X): {X.shape}")
    print(f"Target Shape (Y): {Y.shape}")
    
    print("\n--- Pipelining the Model Layers ---")
    
    # 1. Initialize Layers
    token_emb = TokenEmbedding(VOCAB_SIZE, EMBEDDING_DIM)
    pos_emb = PositionalEmbedding(BLOCK_SIZE, EMBEDDING_DIM)
    
    # Normally we have multiple blocks, but here we demonstrate a single block pipeline
    attention = CausalSelfAttention(EMBEDDING_DIM, NUM_HEADS, BLOCK_SIZE, DROPOUT)
    # LayerNorms are applied before Attention and FeedForward (Pre-LN architecture)
    ln_1 = torch.nn.LayerNorm(EMBEDDING_DIM)
    
    hidden = FeedForward(EMBEDDING_DIM, HIDDEN_DIM, DROPOUT)
    ln_2 = torch.nn.LayerNorm(EMBEDDING_DIM)
    
    output = OutputLayer(EMBEDDING_DIM, VOCAB_SIZE)
    
    # 2. Pipeline / Forward Pass
    print("1. Embedding Layer...")
    x = token_emb(X) + pos_emb(X)
    print(f"   Shape after embeddings: {x.shape}")
    
    print("2. Attention Layer...")
    # Pre-LN -> Attention -> Residual
    attn_out = attention(ln_1(x))
    x = x + attn_out
    print(f"   Shape after attention: {x.shape}")
    
    print("3. Hidden (FeedForward) Layer...")
    # Pre-LN -> FeedForward -> Residual
    hidden_out = hidden(ln_2(x))
    x = x + hidden_out
    print(f"   Shape after hidden: {x.shape}")
    
    print("4. Output Layer (LM Head)...")
    logits = output(x)
    print(f"   Final Logits Shape: {logits.shape}")
    
    # 3. Calculate Loss
    # We flatten the batches and sequence length to calculate cross entropy
    # logits shape: (B, T, V) -> (B*T, V)
    # Y shape: (B, T) -> (B*T)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), Y.view(-1))
    print(f"\n--- Loss Calculation ---")
    print(f"Sample Initial Cross-Entropy Loss: {loss.item():.4f}")
    
    # 4. Optional generation sanity check (can we backprop?)
    loss.backward()
    print("Backward pass successful! Gradients calculated.")