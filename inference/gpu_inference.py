#!/usr/bin/env python3
"""
NeuralAI GPU Inference Module
High-performance inference with GPU acceleration
"""

import os
import time
import torch
from typing import Optional, List, Dict, Generator, Union
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from peft import PeftModel
    from threading import Thread
except ImportError:
    print("Install: pip install transformers peft torch")


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path: Optional[str] = None
    device: str = "auto"  # auto, cuda, cpu, mps
    dtype: str = "auto"  # auto, float16, bfloat16, float32
    max_memory: Optional[Dict] = None
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Streaming
    stream: bool = True
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.dtype == "auto":
            self.dtype = "float16" if self.device == "cuda" else "float32"


class GPUInference:
    """GPU-accelerated inference for NeuralAI"""
    
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.tokenizer = None
        self.device = self.config.device
        self._load_model()
    
    def _load_model(self):
        """Load model with optimal settings"""
        print(f"Loading model: {self.config.model_name}")
        print(f"Device: {self.device}, Dtype: {self.config.dtype}")
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float32)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "torch_dtype": torch_dtype,
        }
        
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
            if self.config.max_memory:
                load_kwargs["max_memory"] = self.config.max_memory
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        # Load adapter if provided
        if self.config.adapter_path and Path(self.config.adapter_path).exists():
            print(f"Loading adapter: {self.config.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)
        
        self.model.eval()
        
        # Print model info
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model loaded: {param_count:.1f}M parameters")
    
    def format_prompt(
        self,
        message: str,
        history: List[Dict] = None,
        system_prompt: str = None
    ) -> str:
        """Format prompt for the model"""
        system_prompt = system_prompt or "You are NeuralAI, a helpful AI assistant."
        
        # Build conversation
        conversation = f"<|system|>\n{system_prompt}\n"
        
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    conversation += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    conversation += f"<|assistant|>\n{content}\n"
        
        # Add current message
        conversation += f"<|user|>\n{message}\n<|assistant|>\n"
        
        return conversation
    
    def generate(
        self,
        message: str,
        history: List[Dict] = None,
        system_prompt: str = None,
        **kwargs
    ) -> str:
        """Generate response (non-streaming)"""
        prompt = self.format_prompt(message, history, system_prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": kwargs.get("temperature", self.config.temperature) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    def generate_stream(
        self,
        message: str,
        history: List[Dict] = None,
        system_prompt: str = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate response with streaming"""
        prompt = self.format_prompt(message, history, system_prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        gen_kwargs = {
            **inputs,
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": kwargs.get("temperature", self.config.temperature) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Run generation in thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Yield tokens
        for token in streamer:
            yield token
        
        thread.join()
    
    def benchmark(self, num_runs: int = 10) -> Dict:
        """Benchmark inference speed"""
        prompt = "Write a short greeting message."
        
        latencies = []
        tokens_generated = []
        
        print(f"Running benchmark ({num_runs} runs)...")
        
        for i in range(num_runs):
            start = time.time()
            response = self.generate(prompt, max_new_tokens=50)
            latency = time.time() - start
            
            latencies.append(latency)
            tokens_generated.append(len(response.split()))
        
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_sec = avg_tokens / avg_latency
        
        results = {
            "device": self.device,
            "dtype": self.config.dtype,
            "runs": num_runs,
            "avg_latency_sec": round(avg_latency, 3),
            "min_latency_sec": round(min(latencies), 3),
            "max_latency_sec": round(max(latencies), 3),
            "avg_tokens": round(avg_tokens, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
        }
        
        print("\nBenchmark Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        
        return results
    
    def get_device_info(self) -> Dict:
        """Get GPU/CPU device information"""
        info = {
            "device": self.device,
            "torch_version": torch.__version__,
        }
        
        if self.device == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                "cuda_version": torch.version.cuda,
            })
        
        return info


class BatchInference:
    """Batch processing for multiple prompts"""
    
    def __init__(self, inference: GPUInference):
        self.inference = inference
    
    def process_batch(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Process multiple prompts in batches"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                response = self.inference.generate(prompt, **kwargs)
                batch_results.append(response)
            
            results.extend(batch_results)
            print(f"Processed {min(i + batch_size, len(prompts))}/{len(prompts)}")
        
        return results
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        prompt_key: str = "prompt",
        **kwargs
    ) -> int:
        """Process prompts from a JSONL file"""
        results = []
        
        with open(input_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get(prompt_key, "")
                if prompt:
                    response = self.inference.generate(prompt, **kwargs)
                    item["response"] = response
                    results.append(item)
        
        with open(output_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        
        print(f"Processed {len(results)} items to {output_file}")
        return len(results)


# Flask integration for web UI
def create_flask_app(inference: GPUInference):
    """Create Flask app with GPU inference"""
    from flask import Flask, request, jsonify, Response
    
    app = Flask(__name__)
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        message = data.get('prompt', '')
        history = data.get('messages', [])
        
        if data.get('stream', True):
            def generate():
                for token in inference.generate_stream(message, history):
                    yield f"data: {json.dumps({'content': token})}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            response = inference.generate(message, history)
            return jsonify({'content': response})
    
    @app.route('/api/status', methods=['GET'])
    def status():
        info = inference.get_device_info()
        return jsonify({
            "model": inference.config.model_name,
            "device": info.get("device"),
            "gpu_name": info.get("gpu_name", "CPU"),
            "status": "ready"
        })
    
    @app.route('/api/benchmark', methods=['POST'])
    def benchmark():
        results = inference.benchmark()
        return jsonify(results)
    
    return app


def main():
    """Demo GPU inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuralAI GPU Inference")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--server", action="store_true", help="Start Flask server")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    config = InferenceConfig(
        model_name=args.model,
        adapter_path=args.adapter,
        device=args.device
    )
    
    inference = GPUInference(config)
    
    if args.benchmark:
        inference.benchmark()
    elif args.server:
        app = create_flask_app(inference)
        print(f"Starting server on port {args.port}")
        app.run(host='0.0.0.0', port=args.port)
    else:
        # Interactive demo
        print("\nNeuralAI GPU Inference (type 'quit' to exit)")
        print("-" * 40)
        
        while True:
            message = input("\nYou: ").strip()
            if message.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\nNeuralAI: ", end="", flush=True)
            for token in inference.generate_stream(message):
                print(token, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
