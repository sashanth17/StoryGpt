import torch
from dataLoader import create_dataloader
from gpt import tinyStory
import os
import sys

def train():
    bin_file = "train.bin"
    if not os.path.exists(bin_file):
        print(f"Error: {bin_file} not found. Run prepare_data.py first.")
        sys.exit(1)

    # Hyperparameters
    BLOCK_SIZE = 256
    BATCH_SIZE = 8
    STRIDE = 128
    VOCAB_SIZE = 4097
    
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DROPOUT = 0.1
    
    LEARNING_RATE = 3e-4
    MAX_ITERS = 20000 # Small number for demonstration (adjust as needed for full training)
    EVAL_INTERVAL = 10
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. DataLoader
    print("Loading Dataset...")
    loader = create_dataloader(
        bin_file, 
        block_size=BLOCK_SIZE, 
        batch_size=BATCH_SIZE, 
        stride=STRIDE, 
        vocab_size=VOCAB_SIZE,
        num_workers=0
    )
    data_iter = iter(loader)

    # 2. Initialize Model
    print("Initializing tinyStory Model...")
    model = tinyStory(
        vocab_size=VOCAB_SIZE, 
        context_size=BLOCK_SIZE, 
        embedding_dim=EMBEDDING_DIM, 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    )
    model.to(device)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("\n--- Starting Training ---")
    model.train()
    
    for i in range(MAX_ITERS):
        try:
            X, Y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            X, Y = next(data_iter)
            
        X, Y = X.to(device), Y.to(device)
        
        # Forward pass
        logits, loss = model(X, Y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if i % EVAL_INTERVAL == 0 or i == MAX_ITERS - 1:
            print(f"Iter {i:3d}/{MAX_ITERS} | Loss: {loss.item():.4f}")

    # 5. Save Weights
    print("\n--- Saving Weights ---")
    torch.save(model.state_dict(), "tinystory_weightsv3.pth")
    print("Training complete! Weights saved to tinystory_weights.pth")

if __name__ == "__main__":
    train()
