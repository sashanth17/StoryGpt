#!/bin/bash
set -e

echo "1. Generating Dataset..."
python3 -u generate_dataset.py

echo "2. Preparing Binary Data..."
python3 -u prepare_instruction_data.py

echo "3. Fine-tuning Model..."
python3 -u instruction_finetune.py

echo "Pipeline complete!"
