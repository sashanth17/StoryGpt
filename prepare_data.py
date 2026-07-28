import os
import sys
import csv
import numpy as np

# Add Tokenizer/v2 to path so we can import it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Tokenizer', 'v2')))
from tokenizer import Tokenizer

def preprocess_data(csv_path, bin_path):
    """
    Reads a CSV dataset, tokenizes the text using Tokenizer v2, and saves 
    the resulting tokens as a continuous binary file of uint16 integers.
    """
    print(f"Initializing tokenizer...")
    tokenizer = Tokenizer()
    all_tokens = []
    eos_id = tokenizer.eos_token_id
    
    print(f"Reading and tokenizing {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Assuming the stories are in the first column or just raw text lines.
        # We use csv.reader to handle quotes properly.
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            text = row[0]
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(eos_id)  # Append EOS token after each story
            
    print(f"Total tokens processed: {len(all_tokens)}")
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"Saved binary tokens to {bin_path}")

if __name__ == "__main__":
    csv_file = "dataset/validation.csv"
    bin_file = "validation.bin"
    
    # Tokenize and save the validation data
    if not os.path.exists(bin_file):
        preprocess_data(csv_file, bin_file)
    else:
        print(f"Found existing binary dataset at {bin_file}, skipping preprocessing.")
