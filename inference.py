import torch
import sys
import os
from gpt import tinyStory

# Add Tokenizer to path so we can import it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Tokenizer', 'v2')))
from tokenizer import Tokenizer

def run_inference():
    # Hyperparameters MUST exactly match the trained model in trainGpt.py
    BLOCK_SIZE = 256
    VOCAB_SIZE = 4097
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DROPOUT = 0.0 # Dropout is disabled for inference
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    weights_path = "tinystory_weights.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Could not find {weights_path}. Run trainGpt.py first!")
        sys.exit(1)

    # 1. Initialize Tokenizer
    print("Loading Tokenizer...")
    tokenizer = Tokenizer()

    # 2. Initialize Model and Load Weights
    print("Loading Model...")
    model = tinyStory(
        vocab_size=VOCAB_SIZE, 
        context_size=BLOCK_SIZE, 
        embedding_dim=EMBEDDING_DIM, 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    )
    
    # Load the state dict (weights)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 3. Encode Prompt
    prompt = "Once upon a time, in a big forest"
    print(f"\nPrompt: '{prompt}'")
    
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    # 4. Generate Story
    print("Generating story...\n")
    max_new_tokens = 50
    # generate function autoregressively adds new tokens
    generated_idx = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    
    # 5. Decode back to text
    generated_tokens = generated_idx[0].tolist()
    story = tokenizer.decode(generated_tokens)
    
    print("--- Generated Story ---")
    print(story)
    print("-----------------------")

if __name__ == "__main__":
    run_inference()
