# Multi-Head Causal Self-Attention

Self-Attention (`model/attention.py`) is the core engine of the Transformer. While the Embedding Layer gives a word its baseline dictionary definition, Self-Attention allows words to look around at the rest of the sentence to understand _context_.

## The Problem: What does "he" mean?

Consider the sentence:

> _"Shane drinks coffee and he also likes to play football."_

When the model processes the word **"he"**, how does it know who "he" is? It could be Shane, it could be someone else mentioned earlier, or it could be a complete mystery.

Attention solves this by allowing the word **"he"** to mathematically "ask" the rest of the sentence for context, and update its own meaning based on the answers.

## The Mechanism: Queries, Keys, and Values

For every single token in the sequence, the model creates three new vectors using linear projections:

1. **Query (Q)**: What information this token is looking for.
   - _Example:_ The word **"he"** generates a Query vector that essentially says: _"I am a singular male pronoun looking for a recent male subject."_
2. **Key (K)**: What information this token contains.
   - _Example:_ The word **"Shane"** generates a Key vector that broadcasts: _"I am a singular male noun."_ The word **"coffee"** generates a Key broadcasting: _"I am an inanimate liquid."_
3. **Value (V)**: The actual underlying content/meaning of the token that it will share if selected.

### The Attention Handshake ($Q \cdot K^T$)

To find out which words are relevant, the model calculates the **dot product** between the Query of the current token ("he") and the Keys of all the preceding tokens.

- The dot product of `Q("he")` and `K("Shane")` results in a **very high score** (their mathematical traits align perfectly).
- The dot product of `Q("he")` and `K("coffee")` results in a **very low score**.

These raw scores are passed through a **Softmax function**, which turns them into percentages (e.g., "Shane" = 95%, "coffee" = 1%, "drinks" = 4%).

Finally, we multiply these percentages by the **Value (V)** vectors. The word "he" absorbs 95% of Shane's Value vector. **The token "he" is no longer just a generic pronoun; its mathematical representation has been fundamentally updated to mean "Shane".**

![Attention Mechanism Diagram](assets/attention_qkv.png)
_(Placeholder: Create an image showing the word "he" firing a Query laser at the Key shields of previous words, with "Shane" glowing brightest)_

## Causal Masking (No Peeking!)

Because StoryGPT is an autoregressive decoder (it generates text one word at a time), it is strictly forbidden from looking into the future.

When generating the word "coffee", the model cannot know that the word "football" is coming later in the sentence. We enforce this physically using a **Causal Mask**.

Before the Softmax step, we apply a lower-triangular matrix over our attention scores. We replace the scores of all "future" tokens with $-\infty$. When passed through Softmax, $e^{-\infty}$ becomes exactly `0`.

- "he" is allowed to attend to "Shane".
- "Shane" is **not** allowed to attend to "he".

## Multi-Head Attention

Instead of performing one massive attention calculation with the full 256-dimensional vector, we slice the dimension into multiple "heads" (e.g., 8 heads of 32 dimensions each) and run them in parallel.

Why? Because words relate to each other in many different ways simultaneously.

- **Head 1** might focus on **Pronoun Resolution** (linking "he" to "Shane").
- **Head 2** might focus on **Verb-Object relationships** (linking "drinks" to "coffee" and "play" to "football").
- **Head 3** might focus on **Punctuation and Grammar** (tracking when the last comma occurred).

By using Multi-Head Attention, the model can track multiple independent contextual threads across the sequence at the exact same time, before gluing them all back together for the next layer.
