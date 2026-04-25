
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def train():
    model_id = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Model & Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    config = AutoConfig.from_pretrained(model_id)
    config.use_cache = False
    config._attn_implementation = 'sdpa'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Setup (Targeting more modules for better reasoning)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, peft_config)

    # 4. Load Authentic Ground-Truth Data (UltraChat 200k subset)
    # This dataset covers 100+ topics and complex instructions
    dataset = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft[:2000]') 

    def tokenize(example):
        # UltraChat uses the 'messages' format which apply_chat_template handles perfectly
        text = tokenizer.apply_chat_template(example['messages'], tokenize=False)
        return tokenizer(text, truncation=True, max_length=1024, padding='max_length')

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # 5. Scaled Training Arguments
    args = TrainingArguments(
        output_dir='./checkpoints_advanced',
        num_train_epochs=1, # 1 epoch on 2000 high-quality samples is substantial for 360M model
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=20,
        save_strategy='no',
        optim='paged_adamw_32bit',
        lr_scheduler_type='cosine'
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print('Starting advanced training on 2000 ground-truth samples...')
    trainer.train()
    model.save_pretrained('./checkpoints/final_model_advanced')
    print('Advanced Training complete!')

if __name__ == "__main__":
    train()
