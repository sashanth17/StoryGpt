# Neural Network Layers

Outside of the attention mechanism, the GPT model relies on standard Neural Network layers to process, expand, and finalize the token representations.

## FeedForward (Hidden Layer)
**Location:** `model/hidden.py`

The `FeedForward` layer (often called a Multi-Layer Perceptron or MLP) is applied to every token independently after the self-attention mechanism.
While self-attention allows tokens to communicate with *each other*, the FeedForward layer allows each token to process the aggregated information individually.

- **Expansion**: It projects the `embedding_dim` into a much larger `hidden_dim` (typically 4x larger). This gives the network the capacity to memorize and store complex patterns.
- **Non-Linearity**: A `GELU` (Gaussian Error Linear Unit) activation function is applied. This non-linearity is what allows the model to approximate complex, non-linear functions.
- **Contraction**: It projects the dimensions back down to the original `embedding_dim` so it can be added back to the residual stream.

## Output Layer (LM Head)
**Location:** `model/output.py`

The `OutputLayer` is the final step in the model. Once the token representations have passed through all Transformer Blocks, they must be converted into actual vocabulary predictions.

1. **LayerNorm**: A final LayerNorm is applied to stabilize the output representations.
2. **Linear Projection**: A linear layer (`lm_head`) projects the `embedding_dim` to the `vocab_size` without biases. 
   - If the vocabulary size is 4097, this layer outputs a vector of size 4097 for every single token in the sequence. These values are the "logits" (raw, unnormalized predictions for what the next token should be).
