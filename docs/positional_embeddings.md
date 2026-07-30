# Positional Embeddings

Unlike older architectures like Recurrent Neural Networks (RNNs) or LSTMs, which read text sequentially from left to right, the Transformer architecture has no inherent concept of sequence order. It processes every single token in the context window simultaneously, in parallel.

To a raw Transformer, a sentence is just a random "bag of words." To fix this, we use **Positional Embeddings** (`model/embedding.py`).

## The "Dog Bites Man" Problem

Imagine feeding two sentences to a Transformer without positional embeddings:

1.  **"Dog bites man"** (Tokens: `[Dog, bites, man]`)
2.  **"Man bites dog"** (Tokens: `[Man, bites, dog]`)

Because the Self-Attention mechanism evaluates all tokens at the exact same time, it simply sees that both sentences contain the exact same three word vectors. It has no way to distinguish who is doing the biting and who is getting bitten. The mathematical output for both sentences would be identical.

## How it works: Giving Words a ZIP Code

To solve this, we give every position in the sentence its own unique mathematical coordinate, or "ZIP Code."

In addition to the Token Embedding look-up table (which contains 4097 vectors), we create a second look-up table of size `[Context Length, Embedding Dimension]`. If our context length ($T$) is 256, this table contains 256 unique vectors, labeled 0 through 255.

When a word enters the model, we combine two pieces of information:

1.  **The Token Vector:** Represents _what_ the word is (e.g., "Dog").
2.  **The Position Vector:** Represents _where_ the word is (e.g., Position 0).

We physically **add** these two 256-dimension vectors together before sending them into the Attention layers.

$$
\text{Final Input Vector} = \text{TokenEmbed}(X_i) + \text{PosEmbed}(i)
$$

Now, when the model processes "Dog" in position 0, its vector is mathematically different than when "Dog" appears in position 2. The attention mechanism can now easily differentiate between "Dog bites man" and "Man bites dog."

## Learnable vs. Fixed Position

While the original "Attention Is All You Need" paper used fixed, hard-coded sinusoidal (sine and cosine) waves to generate these positional vectors, StoryGPT utilizes **Learnable Positional Embeddings**.

Instead of using math formulas, we initialize the 256 position vectors with random numbers. During training, as the model evaluates its loss and backpropagates errors, it adjusts these position vectors alongside the word vectors. The model literally learns for itself the absolute best mathematical way to represent sequence positioning for its specific task.
