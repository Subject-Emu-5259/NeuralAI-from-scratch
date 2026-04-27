#!/usr/bin/env python3
"""
NeuralAI - Flask Web UI Backend
Local model inference + RAG + Neural Uplink hybrid routing.
"""
import os, json, time, torch, hashlib, requests
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from rag import index_document, query_documents, rebuild_index_registry

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(BASE_DIR), "..", "..", "checkpoints", "final_model"))
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")
ALLOWED = {".pdf", ".docx", ".doc", ".txt", ".md"}
REGISTRY_FILE = os.path.join(BASE_DIR, ".indexed_files.json")
UPLINK_URL = os.environ.get("UPLINK_URL", "http://localhost:7000")

AGENT_PATTERNS = [
    "research", "plan", "analyze", "compare", "design a system",
    "debug", "architect", "multi-step", "break down", "investigate",
    "strategy", "workflow", "pipeline", "help me build", "help me create",
    "create a plan", "deep dive", "outline", "step by step",
]

def should_use_agent(prompt: str) -> bool:
    p = prompt.lower()
    for pat in AGENT_PATTERNS:
        if pat in p:
            return True
    return len(prompt) > 600

def query_uplink(user_msg: str, conversation_history: list) -> str:
    payload = {
        "task": user_msg,
        "context": {"conversation": conversation_history[-6:] if conversation_history else []}
    }
    try:
        resp = requests.post(f"{UPLINK_URL}/api/v1/zo/tasks", json=payload, timeout=25)
        data = resp.json()
        result = data.get("result", data.get("error", str(data)))
        if isinstance(result, dict):
            result = result.get("result", str(result))
        return str(result) if result else None
    except Exception as e:
        return f"[Agent error: {e}]"

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_registry(reg):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f)

INDEXED_FILES = load_registry()
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
rebuild_index_registry()

def load_model():
    global model, tokenizer
    fine_path = os.path.join(MODEL_PATH, "adapter_model.safetensors")
    if os.path.exists(fine_path) or os.path.exists(os.path.join(MODEL_PATH, "adapter_model.bin")):
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
            model.eval()
            print("[NeuralAI] Fine-tuned model loaded")
            return
        except Exception as e:
            print(f"[NeuralAI] Fine-tuned load failed: {e}")
    print(f"[NeuralAI] Loading base model: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.eval()
    print("[NeuralAI] Base model loaded")

def lazy_load():
    global model, tokenizer
    if model is None:
        load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported type: {ext}"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    result = index_document(filepath)
    file_id = result.get("file_id", hashlib.sha256(filename.encode()).hexdigest()[:16])
    INDEXED_FILES[file_id] = filename
    save_registry(INDEXED_FILES)
    return jsonify({
        "success": True, "filename": filename, "file_id": file_id,
        "chunks": result.get("chunks", 0),
        "message": f'"{filename}" indexed — {result.get("chunks", 0)} chunks ready.'
    })

@app.route("/api/files", methods=["GET"])
def list_files():
    return jsonify({"files": list(INDEXED_FILES.values())})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    prompt_only = data.get("prompt", "")
    max_new_tokens = int(data.get("max_tokens", 512))
    temperature = float(data.get("temperature", 0.7))
    file_ids = data.get("file_ids", [])
    lazy_load()

    def generate():
        try:
            last_user = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "").strip()
                    break
            user_content = last_user or prompt_only
            if not user_content:
                yield "data: " + json.dumps({"error": "No message content"}) + "\n\n"
                yield "data: [DONE]\n\n"
                return

            # Hybrid: route complex tasks to Neural Uplink agents
            if should_use_agent(user_content):
                yield "data: " + json.dumps({"content": "[Neural Uplink] Routing to agent network...\n"}) + "\n\n"
                agent_response = query_uplink(user_content, messages)
                for word in agent_response.split():
                    yield "data: " + json.dumps({"content": word + " "}) + "\n\n"
                yield "data: [DONE]\n\n"
                return

            # Standard local model response
            chat = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if role in ("system", "user", "assistant") and content:
                    chat.append({"role": role, "content": content})
            if not chat or chat[-1]["role"] != "user":
                chat.append({"role": "user", "content": user_content})

            doc_context = ""
            if file_ids:
                for fid in file_ids:
                    if fid in INDEXED_FILES:
                        docs = query_documents(user_content, top_k=3)
                        if docs:
                            chunks_text = "\n\n---\n\n".join(
                                f"[From {d['source']}]: {d['content']}" for d in docs
                            )
                            doc_context = f"\n\nRelevant context from uploaded documents:\n{chunks_text}\n"
                        break

            system_content = (
                "You are NeuralAI, a helpful and friendly AI assistant built from SmolLM2-360M-Instruct. "
                "You can read and answer questions about uploaded documents when document context is provided. "
                "When a user asks about an uploaded file, use the provided document context to answer accurately. "
                "Do NOT say you cannot read files — you CAN when context is provided. "
                "Be concise, friendly, and helpful."
            )
            if doc_context:
                system_content += doc_context

            enriched_chat = [{"role": "system", "content": system_content}]
            for msg in chat:
                if msg["role"] != "system":
                    enriched_chat.append(msg)

            try:
                prompt = tokenizer.apply_chat_template(
                    enriched_chat, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = ""
                for msg in enriched_chat:
                    prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_len = inputs["input_ids"].shape[1]
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not response:
                response = "I'm not sure how to respond. Could you rephrase?"

            for word in response.split():
                yield "data: " + json.dumps({"content": word + " "}) + "\n\n"
                time.sleep(0.015)
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
            yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/status", methods=["GET"])
def status():
    lazy_load()
    uplink_ok = False
    try:
        import urllib.request
        urllib.request.urlopen(f"{UPLINK_URL}/health", timeout=2)
        uplink_ok = True
    except Exception:
        pass
    return jsonify({
        "model": MODEL_NAME,
        "model_type": "fine-tuned" if os.path.exists(os.path.join(MODEL_PATH, "adapter_model.safetensors")) else "base",
        "device": device,
        "loaded": model is not None,
        "version": "2.3",
        "rag": True,
        "indexed_files": len(INDEXED_FILES),
        "uplink": "connected" if uplink_ok else "offline"
    })

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[NeuralAI] Starting on http://0.0.0.0:{port}")
    print(f"[NeuralAI] Device: {device}")
    print(f"[NeuralAI] Neural Uplink: {UPLINK_URL}")
    app.run(host="0.0.0.0", port=port, debug=False)
