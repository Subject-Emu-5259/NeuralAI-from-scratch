# NeuralAI

## What This Is

NeuralAI is a from-scratch implementation of a GPT-style transformer language model. Every component — attention, embeddings, training loop, text generation — is hand-coded in pure PyTorch with no HuggingFace or external LLM libraries.

This project is designed to be:

- **Educational** — understand every line of a real LLM
- **Scalable** — configs from CPU-only (4GB RAM) to 24GB+ GPU  
- **Modern** — includes Flash Attention, RoPE, SwiGLU, KV-cache

## Architecture

```
Input Tokens
    |
[Token Embedding] + [Positional Embedding or RoPE]
    |
[Dropout]
    |
+-- TransformerBlock x N --+
|   LayerNorm               |
|   CausalSelfAttention     |  <-- Flash Attention (PyTorch 2.0+)
|   Residual +              |
|   LayerNorm               |
|   FeedForward (GELU/SwiGLU)|
|   Residual +              |
+---------------------------+
    |
[Final LayerNorm]
    |
[LM Head (Linear, weight-tied)]
    |
Logits (vocab_size)
```

**Key design choices:**

- Pre-LayerNorm (GPT-2 style) for training stability  
- Weight tying between token embedding and LM head (~25% parameter savings)  
- GPT-2 style weight initialization (scaled residual projections)  
- Optional gradient checkpointing (~40% RAM savings during training)

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-head causal self-attention | ✅ Done | Built from scratch |
| Flash Attention (SDPA) | ✅ Done | PyTorch 2.0+ auto-detected |
| Rotary Position Embeddings (RoPE) | ✅ Done | Better long-context |
| KV-Cache for inference | ✅ Done | Fast autoregressive decoding |
| Gradient Checkpointing | ✅ Done | ~40% memory savings |
| SwiGLU FFN option | ✅ Done | Modern activation |
| Top-p (nucleus) sampling | ✅ Done | Better quality output |
| Top-k sampling | ✅ Done |  |
| Repetition penalty | ✅ Done | Reduces output loops |
| Streaming generation | ✅ Done | Real-time token output |
| BPE Tokenizer (GPT-2) | ✅ Done | tiktoken compatible |
| Text classification fine-tuning | ✅ Done | Sentiment analysis |
| Instruction fine-tuning | ✅ Done | Alpaca-style |
| INT8 inference | ✅ Done | CPU memory optimization |
| Web UI | ✅ Done | Flask-based chat interface |
| Google Colab training notebook | ✅ Done | Free GPU training |

## Model Configurations

| Config | Params | RAM/VRAM | Use Case |
|--------|--------|----------|----------|
| nano | ~2M | <512MB | Testing / CI |
| 4gb | ~8M | <2GB | CPU / 4GB RAM laptop |
| small | ~30M | ~4GB VRAM | Colab T4 / single GPU |
| medium | ~124M | ~8GB VRAM | GPT-2 equivalent |
| large | ~345M | ~24GB VRAM | GPT-2 Medium equivalent |

## Project Structure

```
NeuralAI-from-scratch/
├── from-scratch/           # Core model source code
│   ├── attention.py        # Multi-head causal attention + RoPE + Flash Attn
│   ├── gpt.py              # Full GPT model + gradient checkpointing
│   ├── config.py           # Model configs + auto device detection
│   ├── generate.py         # Text generation: greedy, top-k, top-p, streaming
│   ├── train.py            # Training loop with AdamW + cosine LR
│   ├── train_deploy.py     # Training for deployment/production
│   ├── tokenizer.py        # BPE tokenizer wrapper
│   ├── dataloader.py       # Text data loading + batching
│   ├── finetune.py         # Fine-tuning on custom datasets
│   ├── classifier.py       # Text classification head
│   ├── instruction_tuner.py # Instruction following (Alpaca-style)
│   ├── load_pretrained.py  # Load GPT-2 pretrained weights
│   ├── int8_generator.py   # INT8 quantized inference
│   ├── production.py       # Production inference API
│   ├── web_ui.py           # Flask chat interface
│   ├── data/               # Training data directory
│   └── web_ui/             # Web UI templates/assets
├── checkpoints/            # Saved model checkpoints
├── NeuralAI_Colab_Training.ipynb  # Google Colab training notebook
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch>=2.0.0 tiktoken numpy
```

### 2. Train a Model (CPU, 4GB RAM)

```python
from config import GPT_CONFIG_4GB, get_device
from gpt import GPT

device = get_device()  # auto-detects CUDA > MPS > CPU
model = GPT(GPT_CONFIG_4GB).to(device)
print(model.summary())
# NeuralAI GPT Model Summary
# Parameters : 8,200,000 (8.2M)
# Size (FP32): ~32 MB
# Layers     : 4
# Context    : 256 tokens
```

### 3. Generate Text

```python
from generate import generate, generate_stream

# Standard generation
text = generate(
    model, tokenizer, "Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
)
print(text)

# Streaming generation (prints tokens in real-time)
for token in generate_stream(model, tokenizer, "The future of AI"):
    print(token, end="", flush=True)
```

### 4. Train on Google Colab

Open `NeuralAI_Colab_Training.ipynb` in Google Colab for free GPU training.

## Enhancement Changelog

### v2.0 (Current)

- **attention.py**: Flash Attention via `F.scaled_dot_product_attention`, RoPE, KV-cache, combined QKV projection
- **gpt.py**: Gradient checkpointing, SwiGLU FFN, GPT-2 weight init, model summary, full KV-cache integration
- **config.py**: Auto device detection (CUDA/MPS/CPU), 5-tier config registry (nano/4gb/small/medium/large), `get_config()` helper
- **generate.py**: Top-p nucleus sampling, repetition penalty, streaming generator, KV-cache inference

### v1.0 (Original)

- Base GPT implementation from LLMs-from-scratch blueprint
- Basic multi-head attention, GELU FFN, AdamW training
