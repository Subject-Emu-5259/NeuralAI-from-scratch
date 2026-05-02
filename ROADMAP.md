# 🚀 NeuralAI Expansion Roadmap

**Version Target: 5.0**\
**Timeline: 6-8 weeks**

---

## 📋 Overview

This roadmap outlines the expansion of NeuralAI from a 360M parameter local model to a more capable, aligned, and production-ready AI system.

### Current State

| Component | Status |
| --- | --- |
| Base Model | SmolLM2-360M-Instruct (fine-tuned with QLoRA) |
| Training Samples | 347 |
| Inference | CPU (float32) |
| Alignment | None (supervised fine-tuning only) |
| Evaluation | Manual testing |
| Tools | Chat, RAG, Terminal, Neural Uplink |

### Target State

| Component | Target |
| --- | --- |
| Training Samples | 1000+ (3x expansion) |
| Inference | GPU-hosted (float16 CUDA) |
| Alignment | DPO (Direct Preference Optimization) |
| Evaluation | Automated benchmarks |
| Tools | **Expanded tool ecosystem (PRIORITY)** |

---

## 🎯 Phase 1: Expanded Tool Ecosystem (Weeks 1-2) **PRIORITY**

### Goal: Build tools that make NeuralAI actually useful NOW

### 1.1 Why Tools First?

| Impact | Training Data | Tools |
|--------|---------------|-------|
| User-visible value | Low (minor quality improvement) | **High (new capabilities)** |
| Time to implement | 2 weeks | 1-2 weeks each |
| Dependencies | None | None |
| ROI | Marginal | **Transformative** |

### 1.2 Tools to Build

#### Tool 1: Code Execution Sandbox ⚡

```python
# tools/code_sandbox.py
class CodeSandbox:
    """Execute generated code safely"""

    def run_python(self, code, timeout=30):
        # Run in isolated environment
        pass

    def run_javascript(self, code, timeout=30):
        # Node.js execution
        pass
```

**Features:**
- Safe execution in sandboxed environment
- Timeout protection
- Output capture (stdout/stderr)
- Support for Python and JavaScript

#### Tool 2: File Manager 📁

```python
# tools/file_manager.py
class FileManager:
    """Advanced file operations"""

    def search(self, pattern, directory):
        # Search files by content/pattern
        pass

    def organize(self, directory, rules):
        # Auto-organize files by type
        pass

    def batch_rename(self, files, pattern):
        # Rename multiple files
        pass
```

**Features:**
- Search by content or filename
- Read/write/create/delete files
- Batch operations
- Directory listing with metadata

#### Tool 3: Web Fetcher 🌐

```python
# tools/web_fetcher.py
class WebFetcher:
    """Fetch and parse web content"""

    def fetch(self, url):
        # Get page content
        pass

    def extract(self, url, selector):
        # Extract specific elements
        pass

    def summarize(self, url):
        # Summarize page content
        pass
```

**Features:**
- Fetch and parse HTML
- Extract text, links, images
- Summarize content with local model
- No external API calls (privacy-first)

#### Tool 4: Database Connector 🗄️

```python
# tools/db_connector.py
class DatabaseConnector:
    """Connect to various databases"""

    def connect_sqlite(self, path):
        pass

    def connect_postgres(self, connection_string):
        pass

    def query(self, sql):
        pass

    def schema(self):
        # Get database schema
        pass
```

**Features:**
- SQLite (built-in, no setup)
- PostgreSQL support
- Query execution with results
- Schema inspection

#### Tool 5: Git Assistant 🔧

```python
# tools/git_assistant.py
class GitAssistant:
    """Git operations assistant"""

    def status(self):
        pass

    def commit(self, message):
        pass

    def branch(self, name):
        pass

    def push(self):
        pass

    def pr_create(self, title, body):
        pass
```

**Features:**
- Git status, log, diff
- Stage/commit/push
- Branch management
- PR creation (via GitHub CLI)

### 1.3 Tool Integration in Chat

```javascript
// Tool detection in chat
const TOOL_PATTERNS = {
    code_exec: /run this code|execute this|run the (python|js|javascript)/i,
    file_op: /search (files|folder)|organize files|rename files|read file|write file/i,
    web_fetch: /fetch (this |the )?url|get content from|scrape this page/i,
    database: /query (the )?database|connect to (postgres|sqlite|mysql)/i,
    git: /git (status|commit|push|branch)|create a pr/i,
};

function detectTool(message) {
    for (const [tool, pattern] of Object.entries(TOOL_PATTERNS)) {
        if (pattern.test(message)) {
            return tool;
        }
    }
    return null;
}
```

### 1.4 Implementation Checklist

- [ ] Create `tools/` directory
- [ ] Implement CodeSandbox with timeout protection
- [ ] Implement FileManager with search
- [ ] Implement WebFetcher with parsing
- [ ] Implement DatabaseConnector (SQLite first)
- [ ] Implement GitAssistant using `subprocess`
- [ ] Update `neuralai_router.py` with new tool routes
- [ ] Add tool detection to chat UI
- [ ] Test each tool independently

---

## 🎯 Phase 2: GPU-Hosted Inference (Weeks 3-4)

### Goal: Deploy on GPU for faster inference

### 2.1 Current Performance

| Metric | CPU (Current) |
| --- | --- |
| Time to first token | 2-3 sec |
| Tokens/sec | 5-10 |
| Response time (100 tokens) | 10-20 sec |

### 2.2 Target Performance

| Metric | GPU (Target) |
| --- | --- |
| Time to first token | 0.1-0.3 sec |
| Tokens/sec | 50-100 |
| Response time (100 tokens) | 1-2 sec |

### 2.3 GPU Options

#### Option A: Local GPU (Recommended)

- NVIDIA RTX 3060+ or better
- CUDA 11.8+
- 8GB+ VRAM

#### Option B: Cloud GPU

- RunPod, Lambda Labs, or AWS
- NVIDIA T4 or A10G
- \~$0.20-0.50/hr

#### Option C: Zo Hosted Service

- Register as user service with GPU
- Auto-restart on failure
- Persistent endpoint

### 2.4 GPU Inference Code

```python
# inference/gpu_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPUInference:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, prompt, max_tokens=512, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 2.5 Service Registration

```python
# Register as Zo user service with GPU
# In Zo Computer settings
{
    "label": "neuralai-gpu",
    "mode": "http",
    "local_port": 5000,
    "entrypoint": "python app.py",
    "public": true,
    "gpu": "nvidia-rtx-3060"  # Request GPU
}
```

---

## 🎯 Phase 3: DPO Alignment (Weeks 4-5)

### Goal: Improve response quality through preference optimization

### 3.1 What is DPO?

Direct Preference Optimization (DPO) trains the model to prefer "chosen" responses over "rejected" ones, improving:

- Response helpfulness
- Factual accuracy
- Code correctness
- Safety alignment

### 3.2 DPO Dataset Structure

```json
{
  "prompt": "Write a function to sort a list",
  "chosen": "def sort_list(items):\n    return sorted(items)",
  "rejected": "def sort_list(items):\n    # TODO: implement\n    pass"
}
```

### 3.3 DPO Training Pipeline

```markdown
Training Data (1000+ samples)
       ↓
Supervised Fine-tuning (SFT)
       ↓
Generate Pairs (chosen/rejected)
       ↓
DPO Training (preference learning)
       ↓
Aligned Model
```

### 3.4 DPO Configuration

| Parameter | Value |
| --- | --- |
| Learning Rate | 5e-5 |
| Beta | 0.1 |
| Batch Size | 4 |
| Epochs | 1-2 |
| Max Length | 512 |

### 3.5 Implementation

```python
# training/train_dpo.py
from trl import DPOTrainer
from transformers import AutoModelForCausalLM

def train_dpo(model, train_dataset, output_dir):
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Uses same model as reference
        train_dataset=train_dataset,
        beta=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        max_length=512,
    )
    trainer.train()
    trainer.save_model(output_dir)
```

---

## 🎯 Phase 4: Training Data Expansion (Weeks 5-6)

### Goal: Expand from 347 to 1000+ training samples

### 4.1 Data Categories to Expand

#### Current Categories (347 samples)

- General chat \~50
- Coding \~80
- RAG/document \~40
- Terminal commands \~30
- Math/reasoning \~40
- Explanations \~50
- Tool usage \~57

#### New Categories to Add (653+ samples)

| Category | Samples | Description |
| --- | --- | --- |
| **Advanced Coding** | 100 | Multi-file projects, debugging, refactoring |
| **API Design** | 50 | REST endpoints, OpenAPI specs, webhooks |
| **Database Operations** | 50 | SQL queries, schema design, migrations |
| **DevOps Commands** | 50 | Docker, Kubernetes, CI/CD pipelines |
| **File Operations** | 40 | Read, write, search, organize |
| **Web Scraping** | 30 | Fetching, parsing, extracting data |
| **Data Analysis** | 60 | Pandas, NumPy, visualization |
| **Testing** | 40 | Unit tests, integration tests, mocking |
| **Security** | 30 | Auth patterns, encryption, validation |
| **Error Handling** | 50 | Try-catch patterns, logging, recovery |
| **Documentation** | 40 | README, API docs, comments |
| **Code Review** | 30 | Style, best practices, optimization |
| **Multi-turn Reasoning** | 50 | Complex problem solving chains |
| **Tool Chaining** | 40 | Combining multiple tools in sequence |

### 4.2 Data Generation Script

```python
# training/generate_training_data.py
```

### 4.3 Data Quality Checks

- [ ] Validate JSONL format

- [ ] Check for duplicate prompts

- [ ] Verify response quality (manual review)

- [ ] Balance categories (avoid overfitting)

- [ ] Add diverse system prompts

---

## 🎯 Phase 5: Automated Evaluation (Week 6-7)

### Goal: Automated benchmarks for model quality

### 5.1 Evaluation Metrics

| Metric | Description | Target |
| --- | --- | --- |
| **Perplexity** | Language model quality | &lt; 15 |
| **Code Correctness** | Generated code runs | &gt; 80% |
| **Response Helpfulness** | Human preference sim | &gt; 70% |
| **Factual Accuracy** | Correct information | &gt; 85% |
| **Safety Score** | No harmful outputs | 100% |

### 5.2 Benchmark Suite

```python
# eval/benchmarks.py

class NeuralAIBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def run_all(self):
        self.results = {
            "perplexity": self.eval_perplexity(),
            "code_correctness": self.eval_code(),
            "response_helpfulness": self.eval_helpfulness(),
            "factual_accuracy": self.eval_facts(),
            "safety": self.eval_safety(),
        }
        return self.results

    def eval_perplexity(self):
        # Calculate perplexity on held-out validation set
        pass

    def eval_code(self):
        # Generate code, run it, check for errors
        test_cases = [
            ("Write a function to reverse a string", "reverse_string"),
            ("Implement quicksort", "quicksort"),
            ("Create a REST API endpoint", "api_endpoint"),
        ]
        # Execute generated code and check output
        pass

    def eval_helpfulness(self):
        # Use GPT-4 or similar to score responses
        pass

    def eval_facts(self):
        # Test on factual QA dataset
        pass

    def eval_safety(self):
        # Test for harmful outputs
        pass
```

### 5.3 Continuous Evaluation

```yaml
# .github/workflows/eval.yml
name: Model Evaluation
on:
  push:
    paths: ['checkpoints/**', 'training/**']
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Benchmarks
        run: python eval/benchmarks.py
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval/results.json
```

---

## 🚦 Reorganized Implementation Order

1. **Week 1-2**: **Tools (Phase 1)** ← PRIORITY
2. **Week 3-4**: GPU deployment (Phase 2)
3. **Week 4-5**: DPO alignment (Phase 3)
4. **Week 5-6**: Training data expansion (Phase 4)
5. **Week 6-7**: Evaluation suite (Phase 5)

---

## 📝 Notes

- **Tools deliver immediate value** — users can do more with NeuralAI right away
- GPU can be added at any time (speed improvement)
- DPO requires preference data (human annotations or AI-generated)
- Evaluation should run continuously
- Training data expansion has lower priority — model is already functional

---

**NeuralAI v5.0 Roadmap**\
*Created: May 2, 2026*\
*Reorganized: May 2, 2026*\
*Target Completion: June 2026*