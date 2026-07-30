# Backpropagation and Cross Entropy Loss

During training, the model must measure how "wrong" its predictions are and adjust its weights accordingly.

## Cross Entropy Loss
When the model predicts the next token, it outputs logits for all 4097 possible vocabulary tokens. 
Cross Entropy Loss converts these logits into a probability distribution using a Softmax function, and then penalizes the model based on how much probability mass it assigned to the *correct* target token `Y`.

- If the model assigns 99% probability to the correct token, the loss is near 0.
- If the model assigns 0.1% probability to the correct token, the loss is high.

## Backpropagation
Once the loss is calculated (a single scalar float), PyTorch's Autograd engine computes the gradients of the loss with respect to every single learnable parameter (weight) in the 5.32 million parameter model.
These gradients indicate the direction and magnitude that each weight should be shifted to slightly reduce the loss for that specific batch. The AdamW optimizer then applies this shift.
