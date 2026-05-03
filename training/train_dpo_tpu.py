#!/usr/bin/env python3
"""
NeuralAI DPO (Direct Preference Optimization) TPU Training Script
Optimized for Google Colab TPU v5e-1
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# Ensure TPU compatibility
# For v5e-1, we use a single TPU core
os.environ["ACCELERATE_USE_TPU"] = "true"

@dataclass
class TPUConfig:
    base_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path: str = "checkpoints/final_model"
    output_dir: str = "checkpoints/dpo_tpu_model"
    
    # DPO parameters
    beta: float = 0.1
    learning_rate: float = 5e-5
    batch_size: int = 1  # Keep small for TPU memory
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    max_prompt_length: int = 256
    epochs: int = 1

def train_tpu():
    config = TPUConfig()
    print(f"Loading model: {config.base_model}")
    
    # TPU usually uses bfloat16 for efficiency
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA adapter if it exists
    if Path(config.adapter_path).exists():
        print(f"Loading LoRA adapter from {config.adapter_path}")
        model = PeftModel.from_pretrained(model, config.adapter_path, is_trainable=True)
    else:
        # Fallback: Initialize new LoRA if no adapter found
        print("No adapter found. Initializing new LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Load dataset
    print("Loading dataset...")
    data_path = "data/train_dpo_expanded.jsonl"
    pairs = []
    with open(data_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    dataset = Dataset.from_list([
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    # DPO Training Config
    training_args = DPOConfig(
        output_dir=config.output_dir,
        beta=config.beta,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        bf16=True, # TPUs love bfloat16
        logging_steps=1,
        save_strategy="no", # Save manually at the end
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting TPU Training...")
    trainer.train()
    
    trainer.save_model(config.output_dir)
    print(f"Training complete. Model saved to {config.output_dir}")

if __name__ == "__main__":
    train_tpu()
