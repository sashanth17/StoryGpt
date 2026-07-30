# Transformer Block

The `TransformerBlock` (located in `gpt.py`) is the recurring structural unit of StoryGPT. The model stacks 4 of these blocks sequentially.

## Components of the Block
Each block connects the Attention and Feed Forward sub-layers using two crucial mechanisms:

### 1. Residual Connections
Also known as skip connections, residual connections add the original input of a layer directly to its output (`x = x + layer(x)`).
In deep neural networks, gradients tend to vanish as they propagate backward. The residual connection acts as an uninterrupted "highway" that allows gradients to flow smoothly back to the early layers, stabilizing training.

### 2. Layer Normalization
LayerNorm (`nn.LayerNorm`) standardizes the outputs of a layer so that they have a mean of 0 and a variance of 1. StoryGPT uses **Pre-Normalization**, meaning LayerNorm is applied *before* the Attention and FFN modules, which has been shown to improve training stability compared to the original Post-Normalization architecture.
