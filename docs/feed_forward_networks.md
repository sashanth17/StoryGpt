# Feed Forward Networks

After the Self-Attention mechanism contextualizes the tokens, the representations are passed through a Feed Forward Network (FFN) located in `model/hidden.py`.

## Purpose
While Self-Attention allows tokens to communicate with *each other*, the FFN operates on each token *independently*. It acts as a massive key-value memory bank where the model stores learned facts and transformations.

## Architecture
The FFN in StoryGPT consists of two linear transformations with a non-linear activation function in between:
1. **Expansion Layer**: Projects the embedding dimension (256) up to a much larger hidden dimension (1024).
2. **Activation**: Applies the GELU (Gaussian Error Linear Unit) function.
3. **Projection Layer**: Projects the 1024-dimensional space back down to the original 256 embedding dimension.

Expanding the dimension allows the network to learn complex, high-dimensional representations before condensing them back into the residual stream.
