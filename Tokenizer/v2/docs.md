# Tokenizer V2: High-Performance BPE Architecture

Tokenizer V2 is a production-grade, highly optimized Byte-Pair Encoding (BPE) implementation. It completely resolves the severe performance bottlenecks of naive Python tokenizers by migrating heavy computational workloads to C++ while retaining the ease and accessibility of Python.

This document details the algorithmic optimizations, the architectural design, and the seamless Python-C++ interconnectivity pipeline that powers V2.

---

## 1. Training Pipeline & Algorithmic Analysis (`train.cpp`)

The core challenge of training a BPE tokenizer on large datasets (e.g., gigabytes of text) is that naive implementations repeatedly scan the entire corpus to find and merge pairs, resulting in an unacceptable $O(N \times V)$ time complexity (where $N$ is text length and $V$ is vocabulary size). 

V2 reduces this workload drastically through several major algorithmic leaps:

### O(1) Localized Merge Updates (Doubly-Linked Lists)
Instead of storing words as flat arrays/strings and rebuilding them upon every merge, each word in the corpus is internally structured as a **doubly-linked list** (`struct TokenNode`). 
When tokens `A` and `B` merge into `AB`:
1. The node for `A` updates its ID to `AB`.
2. The node for `B` is unlinked and logically deleted.
3. The system only updates the frequencies of the two strictly adjacent pairs (`prev-A` and `B-next`) and adds the newly formed pairs (`prev-AB` and `AB-next`).
**Analysis:** This reduces the update complexity from $O(L)$ (where $L$ is word length) down to **$O(1)$ constant time** per affected word.

### Fast Affected-Word Tracking
When a pair merges, we must not rescan the entire corpus to find where that pair exists. V2 maintains a hash map (`pair_to_words`) mapping every active pair to a deduplicated `std::vector<int>` of word indices. When a pair is popped from the priority queue, the trainer instantly iterates *only* over the exact words containing that pair.

### Perfect 64-bit Hashing
Using standard `std::pair` in `std::unordered_map` requires a custom hash function. Traditional XOR hashes (`hash(a) ^ hash(b)`) suffer from massive collision rates. V2 implements a perfect 64-bit bitwise combine:
```cpp
return ((uint64_t)left << 32) | (uint32_t)right;
```
**Analysis:** This guarantees zero collisions for pair lookups, maximizing memory access speeds to pure $O(1)$.

### Lazy-Deletion Max-Heap with Garbage Collection
To find the most frequent pair globally, V2 uses a `std::priority_queue`. Since C++ heaps don't support dynamic priority updates, V2 uses **lazy deletion**: when a pair's frequency changes, the new frequency is simply pushed onto the heap. When popping, the system checks if the popped frequency matches the true frequency in the hash map; if not, it's discarded.
To prevent infinite memory growth, a **Garbage Collection** sweep rebuilds the heap entirely from scratch every 1,000 merges, purging all dead entries.

### True Byte-Level BPE & GPT-2 Pre-tokenization
V2 strictly initializes its base vocabulary with raw bytes (`0` to `255`), ensuring 100% compatibility with any UTF-8 text (unlike character-level BPEs which crash on unseen Unicode characters). 
Furthermore, it uses `std::regex` to approximate GPT-2's pre-tokenization rules, gracefully splitting punctuation, numbers, and contractions line-by-line using memory-efficient streaming (`std::getline`).

---

## 2. Inference Pipeline & Interconnectivity (`tokenizer.py` + `inference.cpp`)

While training is a one-time cost, inference (encoding text into tokens) happens constantly in production. V2 implements a hybrid architecture: the heavy BPE crunching is done in C++, but it is seamlessly wrapped in a pure Python interface.

### Python-C++ Interconnectivity (`ctypes`)
We avoid heavy build systems like CMake or PyBind11. Instead, V2 uses Python's built-in `ctypes` library to directly load a compiled C++ shared library (`.so` or `.dylib`). 
- **Auto-Compilation:** The Python module (`tokenizer.py`) is self-aware. On import, if it detects the C++ backend library is missing, it automatically spawns a `subprocess` to invoke `g++` and compile `inference.cpp` in the background. The user experiences zero friction.
- **Memory Management:** Python encodes the input string into raw UTF-8 bytes and passes a raw C-pointer to the C++ backend. C++ allocates an integer array for the token IDs, passes the pointer back to Python, and Python safely calls a C++ `free_tokens` function after copying the data.

### Zero-Overhead Initialization (Binary Formats)
Standard tokenizers suffer from slow startup times because they must parse massive `vocab.json` files. V2 bypasses this entirely:
- **`id_to_token.bin`**: A tightly packed binary file mapping IDs to raw token bytes. Python loads this instantly into a dictionary cache for lightning-fast decoding.
- **`merges.bin`**: Instead of parsing strings from `merges.txt`, `train.cpp` exports the raw 32-bit integer token IDs of the merges. The C++ `inference.cpp` backend reads these integers directly into memory, entirely skipping expensive string-parsing and hash-map lookups during initialization.

### The Inference BPE Loop
Inside `inference.cpp`, the BPE algorithm applies merges in strictly the same rank order discovered during training. It matches the exact same regex pre-tokenization and byte-level breakdown, ensuring 1-to-1 parity between training logic and inference logic, all executed at native C++ speeds.

---

## 3. Quick Start Guide

**1. Training:**
Compile and run the single-file C++ trainer. It will stream your dataset, compute merges, and generate `vocab.json`, `merges.txt`, and the highly-optimized binary files.
```bash
g++ -std=c++20 -O3 train.cpp -o train
./train
```

**2. Inference:**
In your Python project, simply import the tokenizer. It will handle all C++ compilation and interconnectivity automatically.
```python
from tokenizer import Tokenizer

tok = Tokenizer()
tokens = tok.encode("Hello, World!")
text = tok.decode(tokens)
```
