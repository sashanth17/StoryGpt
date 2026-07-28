# The Training Pipeline

**File:** `trainGpt.py`

The training pipeline ties the raw data to the model and optimizes the weights using backpropagation.

## How it works

1. **Device Selection**: 
   The script checks your hardware and automatically selects the fastest processor available: `cuda` (Nvidia GPU), `mps` (Apple Silicon GPU), or `cpu`.

2. **DataLoader**:
   The `create_dataloader` function initializes the `AutoregressiveDataset` which memory-maps the binary dataset file. It slices the data into overlapping sequences governed by the `BLOCK_SIZE` and `STRIDE`, and groups them into batches (`BATCH_SIZE`).

3. **Model Initialization**:
   The `tinyStory` model is instantiated and transferred to the selected `device`. 

4. **Optimizer**:
   The `AdamW` optimizer is initialized. It manages the learning rate and weight decay during training. 

5. **The Training Loop**:
   For a defined number of iterations (`MAX_ITERS`):
   - A batch of Inputs (`X`) and Targets (`Y`) are pulled from the dataloader.
   - The data is sent to the device.
   - A forward pass is executed (`model(X, Y)`), yielding the Cross-Entropy Loss.
   - Gradients are zeroed (`optimizer.zero_grad()`).
   - Gradients are calculated via backpropagation (`loss.backward()`).
   - The optimizer updates the model weights (`optimizer.step()`).

6. **Saving**:
   Once training completes, the model's weight dictionary is saved to disk as `tinystory_weights.pth` using `torch.save()`. This allows the model to be loaded later for inference without retraining.
