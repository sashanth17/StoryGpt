import json
import os
import sys
import numpy as np

# Add Tokenizer/v2 to path so we can import it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tokenizer', 'v2')))
try:
    from tokenizer import Tokenizer
except ImportError:
    print("Could not import Tokenizer. Please ensure the Tokenizer/v2 path is correct.")
    sys.exit(1)

def main():
    if not os.path.exists("instruction_tuning/instruction_dataset2.json"):
        print("instruction_dataset.json not found! Run generate_dataset.py first.")
        sys.exit(1)
        
    print("Loading instruction dataset...")
    with open("instruction_tuning/instruction_dataset2.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} examples.")
    
    tokenizer = Tokenizer()
    eos_id = tokenizer.eos_token_id
    
    all_tokens = []
    
    print("Tokenizing dataset...")
    for item in dataset:
        instruction = item.get("instruction", "")
        response = item.get("response", "")
        
        # Format for instruction tuning
        text = f"User: {instruction}\nModel: {response}"
        
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        
    print(f"Total tokens for instruction tuning: {len(all_tokens)}")
    
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile("instruction_train.bin")
    print("Saved binary tokens to instruction_train.bin")

if __name__ == "__main__":
    main()
