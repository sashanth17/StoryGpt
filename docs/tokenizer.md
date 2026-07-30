# Tokenizer

The StoryGPT tokenizer is a custom implementation of **Byte Pair Encoding (BPE)**, built from scratch to convert raw text into sequences of integers that the model can understand.

## What is BPE?

Byte Pair Encoding is a data compression technique that iteratively replaces the most frequent pair of bytes (or characters) in a sequence with a single, unused token. In the context of LLMs, BPE bridges the gap between character-level tokenization (which creates sequences that are too long) and word-level tokenization (which creates vocabularies that are too massive). It ensures the model has a reasonably sized vocabulary while remaining capable of processing _any_ arbitrary text without encountering "Unknown" tokens.

<a href="../assets/bpeConcept.png">
    <img src="../assets/bpeConcept.png" width="800">
</a>

## How BPE Training Works (Step-by-Step)

Training our BPE tokenizer involves identifying the most frequent adjacent pairs of tokens across the entire TinyStories dataset and merging them into new, single tokens. This is a greedy, iterative process.

### Step 1: Base Vocabulary Initialization

We begin by treating every individual raw byte (256 possible values) in our training data as a distinct token.

- **Base Vocabulary:** 256 byte tokens.

Imagine a highly simplified corpus consisting of just two words, represented by their base characters:

- `("c", "a", "t")` (Frequency: 10)
- `("h", "a", "t")` (Frequency: 5)

### Step 2: Counting Pairs

The algorithm scans the entire corpus and counts every adjacent pair of tokens.

- `("c", "a")`: 10 occurrences
- `("a", "t")`: 15 occurrences (10 from "cat", 5 from "hat")
- `("h", "a")`: 5 occurrences

### Step 3: Merging the Most Frequent Pair

The pair `("a", "t")` is the most frequent. We create a new merge rule and assign it a new, unique Token ID. We then add this merged token to our vocabulary.

- **New Merge Rule:** `("a", "t") -> "at"`
- **Updated Vocabulary Size:** 257

We then traverse the corpus and apply this merge everywhere:

- `("c", "at")` x 10
- `("h", "at")` x 5

### Step 4: Iteration

We repeat Steps 2 and 3 for a predetermined number of iterations (our target vocabulary size).
In the next iteration, the pair `("c", "at")` appears 10 times, making it the most frequent.

- **New Merge Rule:** `("c", "at") -> "cat"`
- **Updated Vocabulary Size:** 258

We continue this loop 3,840 times.

## Saving and Loading Merges (Serialization)

The order in which these merges are learned is absolutely critical. During inference (encoding _new_ text), the tokenizer must apply these exact rules in the exact order they were learned to ensure consistent tokenization.

For example, if we encounter the word "hat" during inference, we consult our ranked list of learned merges:

1.  `("a", "t") -> "at"`
2.  `("c", "at") -> "cat"`

Applying Rule #1: `["h", "a", "t"]` becomes `["h", "at"]`. We check Rule #2, but there is no `"c"` next to an `"at"`, so the final tokenization is `["h", "at"]`.

To guarantee reproducible behavior across Python and C++ environments, the training script serializes two files:

1.  **`merges.bin`**: An ordered binary list of every tuple pair that was merged. The index of the tuple dictates its priority during encoding.
2.  **`id_to_token.bin`**: The final mapping dictionary, mapping integer IDs back to the actual byte sequences (e.g., `256: "at"`, `257: "cat"`). This is used for decoding model output back into human-readable text.

## Python vs. C++ Implementation

StoryGPT includes two distinct versions of the tokenizer:

1. **Python Implementation (`v1`)**: Used exclusively during the initial offline training phase to learn the `merges.bin`. Prioritizes code readability and debugging over speed.
2. **C++ Implementation (`v2`)**: A highly optimized runtime wrapper used during data preprocessing and model inference. It utilizes custom hash maps, priority queues, and lazy updates to dramatically reduce latency when applying thousands of ranked merge rules to massive texts.

## Final Vocabulary Specifications

- **Base Vocabulary**: 256 raw bytes.
- **Merge Operations**: 3840 pairs learned specifically from the linguistic patterns of the TinyStories dataset.
- **Total Vocabulary Size**: 4097 (256 base + 3840 merges + 1 `<|endoftext|>` token injected during serialization).
