# Training on a Custom Dataset

If you want to train the GPT model on your own text data, follow these steps to prepare your dataset, configure the hyper-parameters, and execute training.

## 1. Prepare your Dataset File
First, gather your text data and save it as a CSV file (e.g., `my_custom_data.csv`). Make sure the text you want the model to learn from is inside the first column.

## 2. Tokenize and Binarize
Before the GPT can train on your data, it must be converted into a binary array of integer tokens using your custom Tokenizer.

1. Open `prepare_data.py`.
2. Update the file paths in the `__main__` block:
   ```python
   csv_file = "my_custom_data.csv"
   bin_file = "my_custom_data.bin"
   ```
3. Run the script:
   ```bash
   python3 prepare_data.py
   ```
   This will output a `.bin` file containing your tokenized dataset, and will automatically inject the EOS (End of Sequence) tokens where appropriate.

## 3. Configure Training Hyperparameters
Open `trainGpt.py` and modify the parameters to suit your new dataset and hardware capabilities.

- **`bin_file`**: Set this to the name of your new `.bin` file (e.g., `"my_custom_data.bin"`).
- **`VOCAB_SIZE`**: Ensure this matches your tokenizer's maximum vocabulary size + 1 (for the EOS token). If you haven't changed your tokenizer, leave this as `4097`.
- **`BLOCK_SIZE`**: The maximum context window. Increase this (e.g., `512` or `1024`) if you want the model to remember longer passages of text at once.
- **`BATCH_SIZE`**: How many sequences to process simultaneously. If you get an Out Of Memory (OOM) error on your GPU, decrease this number.
- **Model Size (`EMBEDDING_DIM`, `NUM_HEADS`, `NUM_LAYERS`)**: Increase these for a smarter, larger model, but note that training time will increase significantly.
- **`MAX_ITERS`**: Increase this from `50` to a much larger number (e.g., `5000` or `10000`) for actual, rigorous training. 

## 4. Train the Model
Run the training script:
```bash
python3 trainGpt.py
```
The script will automatically utilize your GPU (CUDA or MPS) if available. The loss should decrease over time. Once training finishes, the weights will be saved as `tinystory_weights.pth`.

## 5. Generate Text
Open `inference.py` to test your newly trained model.
1. **CRITICAL:** Ensure the hyperparameters (`BLOCK_SIZE`, `EMBEDDING_DIM`, `NUM_LAYERS`, etc.) in `inference.py` *exactly* match the ones you set in `trainGpt.py`. If they differ, the weights will fail to load!
2. Change the `prompt` variable to a string of your choice to kickstart the generation.
3. Run the script:
   ```bash
   python3 inference.py
   ```
