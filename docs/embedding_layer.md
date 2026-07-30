# Embedding Layer

The embedding layer (`model/embedding.py`) is the very first component of the neural network. It acts as the model's fundamental dictionary, translating raw integer IDs into rich, mathematical representations.

## The Problem with Raw IDs

When the model receives a token ID (e.g., `1452` for the word "cat"), this integer has absolutely no mathematical meaning. In the realm of integers, `1452` is closer to `1453` than it is to `100`. But in language, token `1452` ("cat") might be semantically identical to token `100` ("feline"), and completely unrelated to token `1453` ("skyscraper").

Feeding raw integers to a neural network would force it to learn a completely chaotic, random mathematical landscape.

## The Intuition: A Massive Look-up Table

The Embedding Layer solves this by acting as a massive look-up table. It maps every discrete token ID in our vocabulary (4097 possible tokens) to a continuous, dense vector of floats (in our case, dimension 256).

Think of it as a grid with 4097 rows and 256 columns. When the model sees Token ID `4`, it literally just grabs Row 4 of this grid and passes that 256-number array to the rest of the network.

## Visualizing "Multi-Dimensional Space"

How does a 256-number array represent "meaning"?

Imagine if, instead of 256 dimensions, we only had **3 dimensions** (or "traits") to describe words. Let's arbitrarily name these traits: `[Fluffiness, Royalty, Danger]`.

Here is how the embedding layer might learn to represent different words in this 3D space:

- **"Dog"** $\rightarrow$ `[0.90, 0.00, 0.40]` _(Very fluffy, not royal, slightly dangerous)_
- **"Cat"** $\rightarrow$ `[0.85, 0.10, 0.30]` _(Very fluffy, slightly royal, slightly dangerous)_
- **"King"** $\rightarrow$ `[0.00, 0.95, 0.50]` _(Not fluffy, highly royal, moderately dangerous)_
- **"Queen"** $\rightarrow$ `[0.00, 0.95, 0.20]` _(Not fluffy, highly royal, less dangerous)_

If you plot these vectors on an X, Y, Z graph, **"Dog" and "Cat" will be physically grouped very close together**, while **"King" and "Queen" will be clustered in a completely different corner of the graph**.

![Embedding Space Visualization](../assets/embedding_space.png)
_(Placeholder: Create an image showing a 3D scatter plot with "Dog"/"Cat" clustered together and "King"/"Queen" clustered elsewhere)_

### The Mathematics of Meaning

Because the model represents words as coordinates in space, it can perform linear algebra on language. The distance between vectors becomes a measure of semantic similarity.

Furthermore, the physical _direction_ between words encodes relationships. The mathematical vector required to travel from "Man" to "King" is nearly identical to the vector required to travel from "Woman" to "Queen".

In StoryGPT, we don't just use 3 traits; we use **256 abstract traits**. The model decides what these 256 traits mean entirely on its own during training via backpropagation.

## Dimensions

Here is how the shape of the data transforms as it passes through this layer:

- **Input Tensor**: `[Batch Size, Context Length]`
  _Example: `[32, 256]` (A batch of 32 sequences, each containing 256 raw integer IDs)._
- **Output Tensor**: `[Batch Size, Context Length, Embedding Dimension]`
  _Example: `[32, 256, 256]` (Every single one of those integer IDs has been expanded into a 256-float vector)._

_Note: In the full architecture, a **Positional Embedding** (which uses the exact same `[256]` dimensional shape) is added directly to this token vector so the model knows where the word is located within the sentence._
