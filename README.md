# NeuralAI-from-scratch

NeuralAI is a fine-tuned SmolLM2-360M-Instruct model optimized for efficient, organic discourse and coding assistance.

## Latest Progress
- **Base Model**: Switched to `SmolLM2-360M-Instruct` for superior reasoning and speed.
- **Advanced Fine-Tuning**: Completed training on 2,000 high-quality samples from the `ultrachat_200k` dataset.
- **Optimizations**: Implemented QLoRA (4-bit quantization) and Flash Attention (SDPA) for Tesla T4 compatibility.
- **Target Modules**: Expanded LoRA targeting to all linear layers for improved code logic.

## Roadmap
- [ ] **Phase 1**: Expand training to 10k+ samples for deeper domain expertise.
- [ ] **Phase 2**: Implement RLHF (Reinforcement Learning from Human Feedback) for safety and alignment.
- [ ] **Phase 3**: Develop a specialized coding-only derivative model.
- [ ] **Phase 4**: Deploy as a lightweight local assistant with a custom Web UI.
