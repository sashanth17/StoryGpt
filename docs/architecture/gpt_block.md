# The GPT Block & Architecture

**File:** `gpt.py`

The complete GPT architecture is created by assembling the individual layers (Embeddings, Attention, FeedForward, and Output) into a unified pipeline.

## TransformerBlock

A single `TransformerBlock` represents one "layer" of the GPT model. It contains:
1. **Pre-Layer Normalization (`ln_1`)**: Normalizes the input before it enters the attention mechanism. This prevents gradients from exploding and stabilizes training.
2. **Causal Self-Attention**: Allows tokens to communicate with each other.
3. **Residual Connection**: The original input is added *back* to the attention output (`x = x + attn(ln(x))`). This allows gradients to flow directly through the network unimpeded.
4. **Pre-Layer Normalization (`ln_2`)**: Normalizes the data before the neural network.
5. **FeedForward Layer**: Expands and contracts the data to learn complex features.
6. **Residual Connection**: Added back to the stream.

## tinyStory (The Master Class)

The `tinyStory` class is the main model that you instantiate. 

**Initialization:**
1. Generates `TokenEmbedding` and `PositionalEmbedding`.
2. Creates a `nn.ModuleList` of `TransformerBlock`s. The number of blocks is determined by the `num_layers` parameter. 
3. Concludes with the `OutputLayer`.

**Forward Pass:**
When data is passed into `tinyStory(idx, targets)`:
1. Token embeddings and positional embeddings are generated and added together.
2. The data passes sequentially through every `TransformerBlock`.
3. The data enters the `OutputLayer` to generate logits.
4. If `targets` are provided (during training), the model automatically calculates the Cross-Entropy Loss by comparing the predicted logits against the true target tokens.

**Generation:**
The `generate()` method handles autoregressive inference. It takes a context, runs the forward pass, plucks the logits for the *final* token, applies temperature scaling, optionally crops to `top_k`, converts logits to probabilities using `softmax`, and samples a new token. It then appends this token to the sequence and repeats the process.
