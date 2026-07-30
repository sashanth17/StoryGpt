# Inference and Autoregressive Generation

Inference (`inference.py`) is the process of generating new text using the trained model.

## Autoregressive Loop
The `generate()` method in `gpt.py` is autoregressive, meaning it feeds its own outputs back into itself as inputs.
1. The user provides a text prompt (e.g., "Once upon a time").
2. The model predicts the next token.
3. The next token is appended to the prompt.
4. The new prompt is fed into the model to predict the *next* token.
5. This repeats until `max_new_tokens` is reached or the `<|endoftext|>` token is generated.

## Sampling Strategies
If we simply select the token with the highest probability (Greedy Search), the stories become incredibly repetitive and boring. StoryGPT implements two sampling strategies to introduce creativity:

### Temperature
Temperature scales the logits before the Softmax function.
- **T = 1.0**: Standard probabilities.
- **T < 1.0**: Sharpen the distribution (makes the model more confident, less creative).
- **T > 1.0**: Flatten the distribution (makes the model more chaotic, more creative).

### Top-K Sampling
Instead of considering all 4097 tokens, the model sorts the probabilities and truncates the list to only the top `K` options (e.g., K=40). The probabilities are re-normalized among these top K tokens. This prevents the model from selecting statistically absurd words (the "long tail" of the distribution) while maintaining creativity.
