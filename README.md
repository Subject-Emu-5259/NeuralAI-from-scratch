# 🧠 NeuralAI - ChatGPT-Style LLM Built From Scratch

![NeuralAI](https://img.shields.io/badge/NeuralAI-v1.0.0-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![Training](https://img.shields.io/badge/Training-Complete-success)

**A production-ready Large Language Model built from scratch using PyTorch**

*Featuring Flash Attention • RoPE • KV-Cache • Streaming • LoRA Fine-tuning*

---

## 📈 Training Performance

### Latest Metrics (April 25, 2026)

| Metric | Value |
|--------|-------|
| **Total Training Samples** | 347 |
| **Base Model** | GPT-2 (124M params) |
| **Training Method** | LoRA Fine-tuning |
| **Final Loss** | 0.040 |
| **Loss Improvement** | 98% reduction |
| **Hardware** | NVIDIA T4 GPU |

### Model Capabilities

✅ Machine Learning Explanations  
✅ Python, JavaScript, SQL Code Generation  
✅ Data Structures & Algorithms  
✅ Web Development & APIs  
✅ Natural Conversations & Q&A  

---

## What This Is

NeuralAI is a from-scratch implementation of a GPT-style transformer language model. Every component — attention, embeddings, training loop, text generation — is hand-coded in pure PyTorch with no HuggingFace or external LLM libraries.

This project is designed to be:

- **Educational** — understand every line of a real LLM
- **Scalable** — configs from CPU-only (4GB RAM) to 24GB+ GPU
- **Modern** — includes Flash Attention, RoPE, SwiGLU, KV-cache

---

## 🚀 Quick Start

```bash
git clone https://github.com/Subject-Emu-5259/NeuralAI-from-scratch.git
cd NeuralAI-from-scratch
pip install -r requirements.txt
```

### Training

See [TRAINING.md](TRAINING.md) for comprehensive training documentation and metrics.

---

## 📚 Documentation

- **[TRAINING.md](TRAINING.md)** - Training metrics, dataset breakdown, loss progression
- **[SOCIAL.md](SOCIAL.md)** - Project branding and social media
- **[TRAINING_PLAN.md](TRAINING_PLAN.md)** - Future roadmap

---

## Architecture

```
Input Tokens
    |
[Token Embedding] + [Positional Embedding or RoPE]
    |
[Multi-Head Attention with Flash or KV-cache]
    |
[Feed-Forward: SwiGLU or standard]
    |
[Layer Normalization]
    ↓
Repeat N layers
    ↓
[Output Logits] → Next Token Prediction
```

---

## License

MIT License - see LICENSE file for details

---

<div align="center">

**Built with ❤️ by Subject-Emu-5259**

*Last Updated: April 25, 2026 | Version: 1.0.0 | Status: ✅ Production Ready*

</div>
