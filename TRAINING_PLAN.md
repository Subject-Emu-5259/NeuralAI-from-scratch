# NeuralAI Custom Model Training Plan

## Overview
**Goal:** Create a custom "NeuralAI" language model fine-tuned for chat and assistant tasks.

## Hardware Assessment
- **Local:** 4GB RAM, 3 CPU cores, **NO GPU** ❌
- **Solution:** Use cloud training (Google Colab free tier or RunPod)

## Model Selection: SmolLM2-360M-Instruct

### Why SmolLM2?
| Criteria | SmolLM2-360M | TinyLlama 1.1B | Phi-2 2.7B |
|----------|-------------|-----------------|------------|
| **Parameters** | 360M | 1.1B | 2.7B |
| **RAM (inference)** | ~1GB | ~2GB | ~5GB |
| **Training time** | Fastest | Fast | Medium |
| **Performance** | Excellent for size | Good | Excellent |
| **Our hardware fit** | ✅ Perfect | ✅ Good | ⚠️ Tight |

### Key Specs
- **Parameters:** 360 million
- **Architecture:** SmolLM2 (modern, efficient)
- **Context:** 2048 tokens
- **License:** Apache 2.0 (free commercial use)
- **Special:** Chat-finetuned variant available

## Training Method: QLoRA (Quantized Low-Rank Adaptation)

### Why QLoRA?
- **Memory reduction:** 75% less than full fine-tuning
- **Speed:** Fast training on consumer GPUs
- **Quality:** Near full fine-tuning performance
- **Cost:** Can train on free Colab tier

### Configuration
```yaml
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
quantization: 4-bit (NF4)
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2e-4
batch_size: 4
epochs: 3
max_length: 512
```

## Training Data

### Phase 1: General Assistant
- OpenAssistant conversations
- Custom NeuralAI prompts
- Chat format data

### Phase 2: Domain Specialization
- User uploaded documents
- Conversation history
- Saved prompts

### Phase 3: Alignment
- DPO (Direct Preference Optimization)
- Safety training

## Training Pipeline

### Option A: Google Colab (FREE)
1. Open Colab notebook
2. Enable GPU (T4 free tier)
3. Run training script
4. Save model to HuggingFace
5. Deploy to NeuralAI server

### Option B: RunPod (Pay-as-you-go)
- **GPU:** RTX 4080 ($0.74/hr)
- **Training time:** ~2 hours
- **Total cost:** ~$1.50

### Option C: HuggingFace AutoTrain
- **Platform:** HuggingFace Spaces
- **GPU:** A100 ($2.50/hr)
- **Training time:** ~30 minutes
- **Total cost:** ~$1.25

## Deployment Strategy

### Model Hosting Options
1. **Local CPU inference** (this server)
   - Pros: Free, private
   - Cons: Slow (~5 tokens/sec)

2. **HuggingFace Inference API**
   - Pros: Fast, easy
   - Cons: Rate limits, costs

3. **Groq API** (current)
   - Pros: Very fast
   - Cons: Rate limits

### Model Switch Implementation
- Add `/api/model` endpoint
- Support: `groq`, `neuralai`, `local`
- UI: Model selector in header

## Timeline

| Week | Task | Status |
|------|------|--------|
| 1 | Set up training infrastructure | ✅ Complete |
| 2 | Collect training data | ✅ In Progress |
| 3 | Train base model (Colab) | ⏳ In Progress |
| 4 | Evaluate and iterate | Not Started |
| 5 | Deploy to production | Not Started |

## Success Metrics
- **Response quality:** Human evaluation
- **Speed:** >3 tokens/sec on CPU
- **Accuracy:** Pass eval suite (10+ questions)
- **Safety:** No harmful outputs

## Features Implemented
- ✅ LR scheduler with warmup
- ✅ Validation split (20%)
- ✅ Perplexity metrics
- ✅ Gradient clipping
- ✅ InstructionDataset class
- ✅ QLoRA fine-tuning
- ✅ Float16 model loading (avoids bitsandbytes CUDA issues)

## Next Actions
1. ✅ Add model switch UI
2. ✅ Create training script (train_neuralai.py)
3. ⏳ Prepare training data
4. ⏳ Run training on Colab
5. ⏳ Deploy trained model