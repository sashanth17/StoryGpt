# Training Pipeline

The pre-training loop for StoryGPT is defined in `trainGpt.py`.

## The Objective
Language modeling is a next-token prediction task. The model is presented with a sequence of tokens (context) and must predict the probability distribution for the immediate next token. 

## The Loop
For each training step:
1. **Fetch Batch**: The DataLoader provides `X` (Inputs) and `Y` (Targets).
2. **Forward Pass**: `X` is passed through the model. The Output Projection layer produces logits (unnormalized log probabilities) of size `[Batch Size, Context Length, Vocabulary Size]`.
3. **Loss Calculation**: We compare the predicted logits against the actual target tokens `Y` using Cross Entropy Loss.
4. **Backward Pass**: Gradients are calculated (Backpropagation).
5. **Optimization**: The `AdamW` optimizer updates the network's weights based on the gradients.

## Optimization & Checkpointing
StoryGPT uses a learning rate of `3e-4` and saves the model state dictionary (`tinystory_weights.pth`) periodically. To monitor progress, the script calculates and logs the training loss. As the loss approaches 2.0, the model begins generating coherent English syntax and story structure.
