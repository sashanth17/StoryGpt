# Autoregressive DataLoader

The DataLoader (`dataLoader.py`) is responsible for feeding the tokenized dataset into the neural network during training.

## Memory Mapping (`np.memmap`)
Modern LLM datasets are hundreds of gigabytes (or terabytes) in size. They cannot fit into RAM. 
To solve this, our DataLoader uses `numpy.memmap()`. This function maps the raw binary file (`train.bin`) directly from the hard drive to a Numpy array interface. PyTorch only loads the specific chunks of data it needs for the current batch into RAM, allowing us to train on datasets of infinite size.

## Input (X) and Target (Y) Shifting
StoryGPT is an autoregressive model. It predicts the *next* token based on the previous tokens. 
If our context length (`block_size`) is 256, the DataLoader fetches a chunk of 257 tokens from the dataset.
- **Input (`X`)**: Tokens 0 to 255.
- **Target (`Y`)**: Tokens 1 to 256.

The model is trained to map `X[t]` to `Y[t]`, effectively predicting the immediate next token at every position.
