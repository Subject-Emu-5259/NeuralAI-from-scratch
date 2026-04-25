
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def train():
    model_id = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Config & Model
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

    # 3. LoRA Setup
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, peft_config)

    # 4. Data Loading & Formatting
    dataset = load_dataset('json', data_files='./data/train.jsonl', split='train')
    
    def formatting_func(example):
        # Handle standard ChatML or instruction formats
        if example.get('messages') is not None:
            try:
                return tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
            except Exception:
                return ""
        elif example.get('instruction') and example.get('response'):
            return f"<|user|>\n{example['instruction']}<|endoftext|>\n<|assistant|>\n{example['response']}<|endoftext|>"
        elif example.get('text'):
            return example['text']
        return ""

    def tokenize(example):
        text = formatting_func(example)
        # If the result is empty, use a dummy string to avoid training errors
        if not text:
            text = tokenizer.eos_token
        return tokenizer(text, truncation=True, max_length=512, padding='max_length')

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # 5. Training
    args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy='epoch',
        optim='paged_adamw_32bit'
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print('Starting training...')
    trainer.train()
    model.save_pretrained('./checkpoints/final_model')
    print('Training complete!')

if __name__ == '__main__':
    train()
