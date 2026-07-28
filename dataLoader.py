import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

class AutoregressiveDataset(Dataset):
    """
    A PyTorch Dataset designed for causal language modeling.
    It reads a massive 1D array of tokenized integers and slices it into
    chunks of size (block_size + 1) to create inputs (X) and targets (Y).
    
    Args:
        data_path (str): Path to the binary file containing tokenized integers.
        block_size (int): The sequence length (context window) for the model.
        stride (int): The number of tokens to shift to create the next sequence. 
                      stride=1 means max overlap, stride=block_size means no overlap.
        vocab_size (int, optional): The vocabulary size, useful for validations.
    """
    def __init__(self, data_path, block_size, stride=1, vocab_size=None):
        super().__init__()
        self.block_size = block_size
        self.stride = stride
        self.vocab_size = vocab_size
        
        # Verify the file exists before trying to map it
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cannot find tokenized data at {data_path}")
            
        # We use numpy.memmap to map the file directly from the hard drive.
        # This means if your dataset is 50GB, it won't crash your computer's RAM.
        # It only loads the specific chunks of data that PyTorch asks for during training.
        # Note: We assume the tokenizer saved the data as 16-bit integers (uint16).
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # We need at least enough tokens to form one full block + 1 target
        assert len(self.data) > block_size, "Dataset is smaller than the block size!"

    def __len__(self):
        # Calculate how many sequences we can extract given the stride
        # We subtract (block_size + 1) because each chunk needs block_size + 1 tokens
        # to generate block_size inputs and block_size targets.
        available_tokens = len(self.data) - (self.block_size + 1)
        if available_tokens < 0:
            return 0
        return (available_tokens // self.stride) + 1

    def __getitem__(self, idx):
        # Calculate the starting token index for this chunk based on the stride
        start_idx = idx * self.stride
        
        # Grab a chunk of tokens of length: block_size + 1
        chunk = self.data[start_idx : start_idx + self.block_size + 1]
        
        # Slice the chunk to create Inputs (X) and Targets (Y)
        # X is the first `block_size` tokens.
        # Y is the last `block_size` tokens (shifted right by 1 position).
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y

def create_dataloader(
    data_path, 
    block_size, 
    batch_size, 
    stride=1, 
    vocab_size=None, 
    num_workers=2, 
    shuffle=True, 
    pin_memory=True, 
    drop_last=True
):
    """
    Creates and configures the PyTorch DataLoader to manage the batches.
    Fully configurable for production pipelines.
    """
    # 1. Instantiate our dataset
    dataset = AutoregressiveDataset(
        data_path=data_path, 
        block_size=block_size, 
        stride=stride, 
        vocab_size=vocab_size
    )
    
    # 2. Wrap it in the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return loader

if __name__ == "__main__":
    bin_file = "validation.bin"
    
    if not os.path.exists(bin_file):
        print(f"Warning: {bin_file} not found. Please run prepare_data.py first.")
        sys.exit(1)

    # Setup hyperparameters
    BLOCK_SIZE = 256   # Context window
    BATCH_SIZE = 4     # Batch size
    STRIDE = 128       # Overlap by half
    VOCAB_SIZE = 4097  # Base vocab (4096) + EOS token (1)

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

    print(f"Batch Size (B): {BATCH_SIZE}")
    print(f"Context Length / Block Size (T): {BLOCK_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Input Shape (X): {X.shape}")
    print(f"Target Shape (Y): {Y.shape}\n")

    print("Look at the very first sequence in the batch:")
    print(f"Input Sequence (X[0, :10]...):  {X[0, :10].tolist()} ...")
    print(f"Target Sequence (Y[0, :10]...): {Y[0, :10].tolist()} ...")
    print("\nNotice how the Target sequence is exactly the Input sequence, just shifted to the left by one position!")