import torch
import os
import sys

# Add parent directory to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataLoader import create_dataloader
from gpt import tinyStory

def train():
    bin_file = "instruction_train.bin"
    if not os.path.exists(bin_file):
        print(f"Error: {bin_file} not found. Run prepare_instruction_data.py first.")
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
    
    # Smaller learning rate for fine-tuning
    LEARNING_RATE = 5e-5
    # Iterations for instruction tuning
    MAX_ITERS = 1000 
    EVAL_INTERVAL = 50
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. DataLoader
    print("Loading Instruction Dataset...")
    loader = create_dataloader(
        bin_file, 
        block_size=BLOCK_SIZE, 
        batch_size=BATCH_SIZE, 
        stride=STRIDE, 
        vocab_size=VOCAB_SIZE,
        num_workers=0
    )
    data_iter = iter(loader)

    # 2. Initialize Model and Load Pretrained Weights
    print("Initializing tinyStory Model...")
    model = tinyStory(
        vocab_size=VOCAB_SIZE, 
        context_size=BLOCK_SIZE, 
        embedding_dim=EMBEDDING_DIM, 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    )
    
    base_weights_path = "tinystory_weightsv3.pth"
    if os.path.exists(base_weights_path):
        print(f"Loading pretrained weights from {base_weights_path}...")
        model.load_state_dict(torch.load(base_weights_path, map_location=device))
    else:
        print(f"Warning: {base_weights_path} not found. Training from scratch.")
        
    model.to(device)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("\n--- Starting Instruction Fine-Tuning ---")
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
            print(f"Iter {i:4d}/{MAX_ITERS} | Loss: {loss.item():.4f}")

    # 5. Save Weights
    print("\n--- Saving Fine-Tuned Weights ---")
    out_path = "StoryGenerater.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Instruction tuning complete! Weights saved to {out_path}")

if __name__ == "__main__":
    train()
