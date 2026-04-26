# 🧠 NeuralAI

![License](https://img.shields.io/badge/License-MIT-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![Status] (https://img.shields.io/badge/Status-Production Ready-success)

**A production-ready AI assistant built from scratch using PyTorch and Hugging Face transformers.**

Fine-tuned from SmolLM2-360M-Instruct with QLoRA — 360M parameters, optimized for chat, coding, document analysis, and reasoning.

---

## ✨ Features & Capabilities

### 📄 Document Intelligence (RAG)
| Capability | Description |
|---|---|
| **File Support** | PDF, DOCX, DOC, TXT, MD |
| **Semantic Search** | ChromaDB vector storage, `all-MiniLM-L6-v2` embeddings |
| **Chunking** | 500 chars per chunk, 80 char overlap |
| **Context Injection** | Top-4 relevant chunks passed to model as context |
| **Chat Integration** | File chips shown on messages; context in system prompt |

### 💻 Coding & Development
| Capability | Languages | Features |
|---|---|---|
| **Code Generation** | Python, JavaScript, SQL, Go, Rust, HTML/CSS | Functions, classes, scripts, APIs |
| **Code Explanation** | Any language | Line-by-line breakdown, pattern identification |
| **Debugging** | Python, JS, SQL | Error analysis, fix suggestions |
| **REST API Design** | OpenAPI/Swagger style | Endpoints, schemas, status codes |
| **Database Design** | SQL schemas | Normalization, indexes, queries |

### 🧠 Reasoning & Analysis
- **Concepts**: Machine learning, transformers, algorithms, data structures
- **Math**: Arithmetic, algebra, calculus explanations
- **Logic**: Problem solving, step-by-step reasoning
- **Comparison**: Technology analysis, pros/cons, recommendations

### 🛠️ Web & Tasks
- Web search information retrieval
- Task planning and breakdown
- Meeting agendas, project plans, timelines
- Email and professional writing
- Social media content creation

---

## 🏗️ Architecture

```
Input
  │
Token Embedding → Positional Encoding
  │
Transformer Blocks (22 layers)
  ├── Multi-Head Self-Attention (Flash/SDPA)
  ├── LayerNorm
  ├── SwiGLU Feed-Forward
  └── Residual Connection
  ↓
Final LayerNorm → LM Head → Next Token
```

| Component | Details |
|---|---|
| **Base Model** | HuggingFaceTB/SmolLM2-360M-Instruct |
| **Fine-tuning** | QLoRA (4-bit NF4, rank=16, alpha=32) |
| **Embedding** | all-MiniLM-L6-v2 (384 dim) |
| **Vector DB** | ChromaDB (Persistent) |
| **Framework** | Flask + vanilla JS (no frontend frameworks) |
| **Device** | CPU (float32) — GPU-ready (float16 CUDA) |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Subject-Emu-5259/NeuralAI-from-scratch.git
cd NeuralAI-from-scratch

# Install dependencies
pip install torch transformers peft chromadb sentence-transformers pypdf python-docx flask

# Run the web UI (port 5000)
cd from-scratch/web_ui
python app.py
```

Or on Google Colab:
```bash
!git clone https://github.com/Subject-Emu-5259/NeuralAI-from-scratch.git
%cd NeuralAI-from-scratch
!pip install -r requirements.txt
!cd from-scratch/web_ui && python app.py
```

---

## 📁 Project Structure

```
NeuralAI-from-scratch/
├── README.md              ← This file
├── SOCIAL.md               ← Social media preview & branding
├── TRAINING_PLAN.md        ← Training roadmap & milestones
├── requirements.txt        ← Python dependencies
├── checkpoints/           ← Trained model weights
│   └── final_model/       ← LoRA adapter + config
├── data/
│   └── train.jsonl        ← Training data (347 samples)
└── from-scratch/
    ├── training/
    │   ├── train_neuralai.py    ← QLoRA fine-tuning script
    │   └── train_novai.py       ← NovaAI (deprecated)
    └── web_ui/
        ├── app.py          ← Flask backend (chat, upload, status)
        ├── rag.py         ← RAG: embedding, chunking, retrieval
        ├── chroma_db/     ← Persistent vector database
        ├── uploads/       ← Uploaded documents
        ├── templates/
        │   └── index.html ← Main UI (header, sidebar, chat, input)
        └── static/
            ├── css/main.css   ← Styles
            └── js/main.js     ← Frontend logic
```

---

## 🔬 Training Details

| Metric | Value |
|---|---|
| **Base Model** | SmolLM2-360M-Instruct (360M params) |
| **Method** | QLoRA — 4-bit NF4 quantization |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **Dropout** | 0.05 |
| **Learning Rate** | 2e-4 with scheduler + warmup |
| **Batch Size** | 4 |
| **Gradient Accumulation** | 4 steps |
| **Max Length** | 512 tokens |
| **Validation Split** | 20% |
| **Gradient Clipping** | 1.0 |
| **Training Samples** | 347 |
| **Final Loss** | 0.040 |
| **Loss Improvement** | 98% reduction |

---

## 🌐 Live Deployment

**Web UI:** https://neural-deandrewharris.zocomputer.io

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main chat interface |
| `/api/chat` | POST | Stream chat response |
| `/api/upload` | POST | Upload PDF/DOCX/TXT/MD for RAG |
| `/api/status` | GET | Model status, RAG state, version |
| `/api/files` | GET | List indexed documents |

---

## ⚙️ API Reference

### `POST /api/chat`
```json
{
  "prompt": "Write fibonacci in Python",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.7,
  "max_tokens": 512,
  "file_ids": ["abc123"]
}
```
Response: Server-Sent Events (SSE) stream of `{"content": "word "}` chunks.

### `POST /api/upload`
- `Content-Type: multipart/form-data`
- Field: `file` (PDF/DOCX/DOC/TXT/MD, max 16MB)
- Response: `{"success": true, "filename": "...", "file_id": "...", "chunks": N}`

### `GET /api/status`
```json
{
  "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
  "model_type": "fine-tuned",
  "device": "cpu",
  "version": "2.1",
  "rag": true,
  "indexed_files": 1
}
```

---

## 📊 Metrics

- **Model Size:** 360M parameters (~700MB in float32, ~200MB in 4-bit)
- **Context Window:** 2048 tokens
- **Inference Speed:** ~10-20 tokens/sec on CPU
- **RAG Accuracy:** Semantic retrieval from uploaded documents
- **Training Time:** ~30-60 min on Google Colab T4 GPU

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `../../checkpoints/final_model` | Path to trained LoRA adapter |
| `MODEL_NAME` | `HuggingFaceTB/SmolLM2-360M-Instruct` | Base model name |
| `PORT` | `5000` | Flask server port |

---

## 📖 Documentation

- **[TRAINING.md](TRAINING_PLAN.md)** — Full training metrics, dataset breakdown, loss curves
- **[SOCIAL.md](SOCIAL.md)** — GitHub bio, social preview, sample posts
- **[TRAINING_PLAN.md](TRAINING_PLAN.md)** — Roadmap, timeline, next steps

---

## 🛡️ Safety & Limitations

- NeuralAI may produce inaccurate information — always verify critical outputs
- RAG context is limited to top-4 retrieved chunks (max ~2000 chars)
- Model is 360M params — less capable than large models on complex reasoning
- CPU inference is slower than GPU — expect 10-20 sec for typical responses

---

## 📝 License

MIT License — free to use, modify, and commercialize.

---

<div align="center">

**Built with ❤️ by [Subject-Emu-5259](https://github.com/Subject-Emu-5259)**

*Last Updated: April 25, 2026 | Version: 2.1 | Status: Production* 🟢

</div>
