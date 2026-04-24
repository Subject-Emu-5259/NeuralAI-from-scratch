#!/usr/bin/env python3
"""
NovaAI Model Training Script
============================
Fine-tune TinyLlama 1.1B using QLoRA for chat/assistant tasks.

Usage:
    python train_novai.py --data ./data/train.jsonl --output ./checkpoints

Requirements (for Colab):
    pip install torch transformers peft bitsandbytes accelerate datasets
"""
import os
import json
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train NovaAI model")
    parser.add_argument("--data", type=str, default="./data/train.jsonl", help="Training data file")
    parser.add_argument("--output", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 NovaAI Model Training")
    print("=" * 60)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"❌ Training data not found: {args.data}")
        print("\nCreating sample data...")
        create_sample_data(args.data)
        print(f"✅ Sample data created: {args.data}")
    
    # Check dependencies
    missing = []
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        print("✅ PEFT (LoRA)")
    except ImportError:
        missing.append("peft")
    
    try:
        import bitsandbytes
        print("✅ BitsAndBytes (quantization)")
    except ImportError:
        missing.append("bitsandbytes")
    
    try:
        import accelerate
        print("✅ Accelerate")
    except ImportError:
        missing.append("accelerate")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr use Google Colab with:")
        print("  !pip install torch transformers peft bitsandbytes accelerate")
        sys.exit(1)
    
    # Import for training
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    import transformers
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("\n⚠️ No GPU - training on CPU (will be slow)")
        device = "cpu"
    
    # Load training data
    print(f"\n📚 Loading training data from: {args.data}")
    samples = []
    with open(args.data, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"   Loaded {len(samples)} samples")
    
    # Model config
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    def format_chat(example):
        """Format messages into chat template."""
        messages = example.get("messages", [])
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                text += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}</s>\n"
        return {"text": text}
    
    def tokenize(example):
        """Tokenize the formatted text."""
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
    
    print("\n🔧 Preparing dataset...")
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(format_chat)
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True if device == "cuda" else False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    print("\n🏋️ Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print("-" * 60)
    
    trainer.train()
    
    # Save
    print("\n💾 Saving model...")
    model.save_pretrained(os.path.join(args.output, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output, "final_model"))
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {args.output}/final_model")
    print("\n📤 To upload to HuggingFace:")
    print("   from huggingface_hub import HfApi")
    print("   api = HfApi()")
    print("   api.upload_folder(folder_path='checkpoints/final_model', repo_id='YOUR_USERNAME/novai-model')")
    print("=" * 60)


def create_sample_data(output_path):
    """Create sample training data."""
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "What can you help me with?"},
                {"role": "assistant", "content": "I can help you with a wide range of tasks including answering questions, writing content, analyzing documents, coding assistance, and much more. What would you like help with today?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "Explain how RAG works"},
                {"role": "assistant", "content": "RAG (Retrieval-Augmented Generation) combines document retrieval with AI generation. When you ask a question, the system searches your uploaded documents for relevant content, then feeds that context to the AI model to generate an informed response."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "Write a professional email"},
                {"role": "assistant", "content": "I'd be happy to help you write a professional email. Could you tell me:\n1. Who is the recipient?\n2. What is the purpose?\n3. Any key points to include?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "Review this code for bugs"},
                {"role": "assistant", "content": "Please paste the code you'd like me to review. I'll check for logic errors, security vulnerabilities, performance issues, and best practice violations."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "Summarize this document"},
                {"role": "assistant", "content": "To summarize a document, either upload it using the Docs button or paste the text directly. I'll create a concise summary highlighting key points."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! I'm NovaAI, your AI assistant. How can I help you today?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "What are you?"},
                {"role": "assistant", "content": "I'm NovaAI, an AI-powered assistant with document intelligence and semantic search capabilities. I can help with questions, writing, coding, document analysis, and more."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are NovaAI, a helpful AI assistant."},
                {"role": "user", "content": "Help me brainstorm ideas"},
                {"role": "assistant", "content": "I'd love to help brainstorm! What topic or problem would you like ideas for? The more context you provide, the better I can help."}
            ]
        },
    ]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
