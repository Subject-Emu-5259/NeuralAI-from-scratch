#!/usr/bin/env python3
"""
NeuralAI — Flask Web UI Backend
Serves the NeuralAI chat interface with local model inference.
"""
import os
import sys
import json
import torch
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from datetime import datetime

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "final_model")
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")

# Model state
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the fine-tuned NeuralAI model or fall back to base."""
    global model, tokenizer
    
    model_file = os.path.join(MODEL_PATH, "adapter_model.safetensors")
    base_model_file = os.path.join(MODEL_PATH, "pytorch_model.bin")
    
    if os.path.exists(model_file) or os.path.exists(base_model_file):
        print(f"[NeuralAI] Loading fine-tuned model from {MODEL_PATH}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            quantization_config = None
            if device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            print("[NeuralAI] Fine-tuned model loaded successfully")
            return
        except Exception as e:
            print(f"[NeuralAI] Failed to load fine-tuned model: {e}")
            print("[NeuralAI] Falling back to base model")
    
    print(f"[NeuralAI] Loading base model: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quantization_config = None
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print("[NeuralAI] Base model loaded")

# Load model on startup (only if explicitly requested)
def lazy_load():
    global model, tokenizer
    if model is None:
        load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_tokens', 256)
    temperature = data.get('temperature', 0.7)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    lazy_load()
    
    def generate():
        try:
            # Build prompt with system context
            messages = [
                {"role": "system", "content": "You are NeuralAI, a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Format with chat template
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|system|>\n{content}</s>\n"
                elif role == "user":
                    text += f"<|user|>\n{content}</s>\n"
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            # Remove the input prompt from response
            response = response[len(text):].strip()
            
            for chunk in response.split():
                yield f"data: {chunk} \n\n"
                import time; time.sleep(0.02)
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/status', methods=['GET'])
def status():
    lazy_load()
    model_type = "fine-tuned" if os.path.exists(os.path.join(MODEL_PATH, "adapter_model.safetensors")) else "base"
    return jsonify({
        'model': MODEL_NAME,
        'model_type': model_type,
        'device': device,
        'loaded': model is not None
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"NeuralAI starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)