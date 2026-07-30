# Dataset Preparation Pipeline

Autoregressive language models cannot process raw text; they require structured, numerical input. The `prepare_data.py` script bridges this gap by transforming raw CSV files from the TinyStories dataset into a highly optimized binary format ready for high-speed GPU training.

## The Preprocessing Pipeline

The script executes the following sequence to prepare the data. This approach is designed to handle multi-gigabyte datasets without exceeding system RAM limits.

### 1. Parsing & Chunking

The script reads the raw `train.csv` file. To prevent memory bottlenecks (OOM errors) when processing gigabytes of text, it streams the data in manageable chunks rather than loading the entire file into RAM. By flushing data to the disk periodically, the RAM footprint remains negligible regardless of the dataset size.

### 2. C++ Tokenization

The raw text is passed into our custom, compiled C++ BPE Tokenizer wrapper. This engine rapidly converts the strings into arrays of integer Token IDs. Because the tokenization logic is executed natively in C++ via `ctypes`, it circumvents Python's Global Interpreter Lock (GIL) and string-processing overhead, allowing for extremely fast encoding.

### 3. Sequence Packing & Boundary Injection

Because TinyStories are very short (often 50-100 words), training on one story at a time wastes GPU compute on padding. Instead, the script concatenates all stories into one massive, continuous 1D array.

To prevent the model from learning false relationships between the end of one story and the beginning of the next, a phantom `<|endoftext|>` token (ID: 4096) is injected exactly at the boundary of every story.

### 4. Binary Serialization

Storing hundreds of millions of integers in text formats like CSV or JSON introduces massive I/O overhead during training. Instead, `prepare_data.py` writes the integer array directly to disk as raw, contiguous binary bytes (`.bin`).

---

## Technical Specifications: Why `uint16`?

Memory bandwidth is often the primary bottleneck in language model training.

Our custom tokenizer has a vocabulary size of exactly 4097 (4096 base tokens + 1 EOS token), making 4096 the highest possible token ID. A 16-bit unsigned integer (`uint16`) can safely store values up to 65,535.

By forcing the Numpy arrays to use `np.uint16` before writing to disk, we achieve significant performance gains over standard 32-bit (`int32`) or 64-bit (`int64`) integers:

- **Storage Efficiency**: The dataset footprint on the hard drive is reduced by 50% to 75%.
- **Memory-Mapping (mmap) Speed**: During training, the PyTorch `DataLoader` uses `np.memmap` to stream this binary file directly into RAM. Halving the byte size doubles the speed at which the CPU can fetch batches and transfer them across the PCIe bus to the GPU.

_(Note: PyTorch requires 64-bit integers for its `Embedding` and `CrossEntropyLoss` functions, so the `DataLoader` will seamlessly cast these `uint16` chunks up to `torch.long` at the very last second before they hit the model.)_
