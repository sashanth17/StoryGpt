# Causal Self-Attention

The `CausalSelfAttention` layer (located in `model/attention.py`) is the core mechanism that allows the GPT model to look at the context of a sequence and determine which words are most relevant to predicting the next word.

## How it works

1. **Linear Projections**:
   The input tensor (representing token embeddings) is projected into three distinct representations using a single linear layer:
   - **Query (Q)**: What the current token is "looking for".
   - **Key (K)**: What the current token "contains".
   - **Value (V)**: The actual information the token provides if it is selected.

2. **Multi-Head Splitting**:
   The `Q`, `K`, and `V` matrices are split across multiple "heads". This allows the model to attend to different concepts simultaneously (e.g., one head might look for grammatical structure, while another looks for character names).

3. **Attention Calculation**:
   The model calculates the dot-product between Queries and Keys (`Q @ K^T`) to compute raw attention scores. These scores tell us how much focus the current token should place on every other token. The scores are scaled down by the square root of the head dimension to keep gradients stable.

4. **Causal Masking**:
   Because GPT is an autoregressive model (it predicts the *next* token based on *past* tokens), it must not be allowed to "look ahead" into the future. 
   A lower triangular mask (the `bias` buffer) is applied to the attention scores. All future positions are set to negative infinity (`-inf`), which zero out during the softmax step.

5. **Aggregation**:
   The attention weights are multiplied by the Values (`V`). Finally, the outputs of all heads are concatenated and passed through a final linear projection layer to mix the information.
