# Instruction Fine-Tuning

A base language model is only trained to *predict the next word*. If you prompt it with "Write a bedtime story", it might continue with "about a dog." rather than actually writing the story. 
Instruction Fine-Tuning converts the base model into an assistant that follows commands.

## The Pipeline (`instruction_tuning/`)
1. **Dataset Generation (`generate_dataset.py`)**: 
   We use a local LLM (like Phi-3 via Ollama) to synthesize hundreds of instruction-response pairs based on our original dataset. 
   Example format: `User: Tell me a bedtime story.\nModel: Once upon a time...`
2. **Tokenization (`prepare_instruction_data.py`)**:
   These pairs are tokenized and saved as a binary file, just like the pre-training dataset.
3. **Supervised Fine-Tuning (`instruction_finetune.py`)**:
   We load the pre-trained `tinystory_weights.pth` and resume training on the instruction dataset. We use a lower learning rate (`5e-5`) to avoid catastrophically forgetting the grammar learned during pre-training.

## Output
After tuning for ~1000 iterations, the model learns the `User: ... \nModel: ` structural pattern and successfully triggers the story generation behavior in response to direct commands.
