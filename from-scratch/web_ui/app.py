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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/workspace/Projects/NeuralAI/checkpoints/final_model")
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, tokenizer
    model_file = os.path.join(MODEL_PATH, "adapter_model.safetensors")
    base_model_file = os.path.join(MODEL_PATH, "pytorch_model.bin")

    if os.path.exists(model_file) or os.path.exists(base_model_file):
        print(f"[NeuralAI] Loading fine-tuned model from {MODEL_PATH}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            print("[NeuralAI] Fine-tuned model loaded successfully")
            return
        except Exception as e:
            print(f"[NeuralAI] Failed to load fine-tuned model: {e}")

    print(f"[NeuralAI] Loading base model: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print("[NeuralAI] Base model loaded")

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

    messages = data.get('messages', [])
    prompt = data.get('prompt', '')

    if messages:
        prompt = "\n".join([
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in messages
        ])

    max_new_tokens = data.get('max_tokens', 256)
    temperature = data.get('temperature', 0.7)

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    lazy_load()

    def generate():
        try:
            text = f"<|system|>\nYou are NeuralAI, a helpful AI assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response[len(text):].strip()

            for chunk in response.split():
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
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
