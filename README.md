# NeuralAI

**NeuralAI** is a custom language model fine-tuned from SmolLM2-360M-Instruct for chat and assistant tasks using QLoRA fine-tuning.

## Features

- QLoRA fine-tuning (4-bit NF4 quantization)
- Validation split with perplexity metrics
- LR scheduler with warmup + gradient clipping
- InstructionDataset class for chat-formatted data
- Float16 model loading (avoids CUDA quantization issues)

## Quick Start

```bash
# Install dependencies
pip install torch transformers peft bitsandbytes accelerate datasets

# Run training
python training/train_neuralai.py --data ./data/train.jsonl --output ./checkpoints

# Or with Colab
!pip install torch transformers peft bitsandbytes accelerate datasets
!python training/train_neuralai.py --epochs 3 --batch-size 4
```

## Project Structure

```
novai-model/
├── TRAINING_PLAN.md        # Training roadmap and progress
├── data/
│   └── train.jsonl         # Training data (chat format)
├── checkpoints/            # Model checkpoints
│   └── training_config.json
└── training/
    └── train_neuralai.py   # Main training script
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | HuggingFaceTB/SmolLM2-360M-Instruct |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| Learning Rate | 2e-4 |
| Batch Size | 4 |
| Max Length | 512 |
| Context | 2048 tokens |

## Training Data Format

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are NeuralAI, a helpful AI assistant."},
    {"role": "user", "content": "What can you help me with?"},
    {"role": "assistant", "content": "I can help you with..."}
  ]
}
```

## Deployment

After training, upload to HuggingFace:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="checkpoints/final_model",
    repo_id="YOUR_USERNAME/neuralai-model"
)
```

## Status

- ✅ Training infrastructure ready
- 🔄 Training data preparation in progress
- ⏳ Colab training pending
- ⏳ Production deployment pending