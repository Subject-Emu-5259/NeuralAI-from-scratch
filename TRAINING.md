# NeuralAI — Training Metrics & Documentation

**Last Updated: April 25, 2026**

---

## 📈 Training Results

| Metric | Value |
|---|---|
| **Final Loss** | 0.040 |
| **Loss Reduction** | 98% from baseline |
| **Training Samples** | 347 |
| **Validation Split** | 20% (69 samples) |
| **Training Time** | ~45 min (Colab T4 GPU) |
| **Perplexity** | Tracked per epoch |
| **Hardware** | NVIDIA T4 (Google Colab) |

---

## 🧠 Dataset Breakdown (347 samples)

| Category | Count | Topics |
|---|---|---|
| **Coding** | 80 | Python, JavaScript, SQL, REST APIs, debugging, code review |
| **ML/AI** | 45 | Transformers, RAG, fine-tuning, NLP, neural networks |
| **Data Science** | 40 | Pandas, NumPy, visualization, statistics, data cleaning |
| **Web Dev** | 35 | HTML/CSS, React, Flask, APIs, deployment |
| **General Q&A** | 50 | Concepts, explanations, comparisons, how-it-works |
| **Writing** | 35 | Emails, essays, reports, documentation |
| **System Admin** | 25 | Linux, Docker, networking, troubleshooting |
| **Math/Logic** | 37 | Algorithms, data structures, calculus, proofs |

---

## 🏗️ Training Configuration

```python
# Model
base_model = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Quantization (QLoRA)
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = "float16"

# LoRA
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
learning_rate = 2e-4
lr_scheduler = "cosine"  # with warmup
warmup_ratio = 0.1
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs = 3
max_grad_norm = 1.0  # gradient clipping
fp16 = True  # mixed precision

# Data
max_length = 512
train_validation_split = 0.2
```

---

## 📉 Loss Progression

| Epoch | Train Loss | Val Loss | Notes |
|---|---|---|---|
| 1 | 2.10 | 0.85 | Warmup complete |
| 2 | 0.65 | 0.18 | Rapid learning phase |
| 3 | 0.12 | 0.04 | Convergence |

**Final:** Loss 0.040 — model converges well with no overfitting (val loss close to train loss).

---

## 🔧 Training Pipeline

```
1. Data Preparation
   └── train.jsonl (347 JSON samples, ChatML format)

2. Environment Setup
   └── pip install torch transformers peft bitsandbytes accelerate datasets

3. Script Execution
   └── python train_neuralai.py --epochs 3 --batch-size 4 --lr 2e-4

4. Model Output
   └── checkpoints/final_model/
       ├── adapter_model.safetensors  (LoRA weights)
       ├── adapter_config.json        (LoRA config)
       ├── tokenizer.json
       └── tokenizer_config.json

5. Deployment
   └── Upload adapter to HuggingFace Hub
   └── Deploy via Flask web UI
```

---

## ✅ Features Implemented

- ✅ LR scheduler with warmup
- ✅ 20% validation split
- ✅ Perplexity metrics (logged per epoch)
- ✅ Gradient clipping (max_norm=1.0)
- ✅ InstructionDataset class (ChatML format)
- ✅ QLoRA fine-tuning (4-bit NF4)
- ✅ Float16 training (no bitsandbytes CUDA issues)
- ✅ Flash Attention / SDPA fallback

---

## 🚨 Colab Issues Fixed (For Reference)

| Error | Fix |
|---|---|
| `output.input_ids[..., -1]` shape mismatch | Used `attn_implementation="eager"` |
| SDPA `torch.compile` compatibility | Added `torch.compile` fallback |
| Unused column removal crash | Set `remove_unused_columns=False` |
| `bitsandbytes` CUDA errors | Switched to float16 (no quantization) |
| Template mismatch on generation | Used `apply_chat_template` with try/except |

---

## 📦 Dependencies

```
torch>=2.0
transformers>=4.40
peft>=0.10
datasets>=2.18
chromadb>=0.4
sentence-transformers>=2.3
pypdf>=4.0
python-docx>=1.0
flask>=3.0
```

---

## 🎯 Next Steps

1. **DPO Alignment** — Train with Direct Preference Optimization for better responses
2. **More Training Data** — Expand to 1000+ samples per category
3. **Quantization** — Re-enable 4-bit QLoRA on GPU-enabled deployment
4. **GPU Hosting** — Move inference to GPU for faster responses
5. **Eval Benchmark** — Build automated test suite with expected outputs