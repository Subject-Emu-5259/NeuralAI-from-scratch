import hashlib
import json
import os
import time
import asyncio
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

# NeuralAI Engine - Router + Local Model + Uplink + Tools
try:
    from neuralai_engine import neuralai_chat, neuralai_route, local_model
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    def neuralai_route(msg):
        return ("local", None)
    neuralai_chat = None
    local_model = None


def strip_terminal_prefix(msg: str) -> str:
    """Remove terminal command prefixes."""
    lower = msg.lower()
    for prefix in ["run ", "execute ", "shell ", "command "]:
        if lower.startswith(prefix):
            return msg[len(prefix):].strip()
    return msg

try:
    import torch
except Exception:
    torch = None

try:
    import requests
except Exception:
    requests = None

try:
    from rag import index_document, query_documents, rebuild_index_registry
except Exception:
    def index_document(filepath: str, collection_name: str = "documents") -> dict:
        return {"chunks": 0, "error": "RAG backend unavailable"}

    def query_documents(query: str, collection_name: str = "documents", top_k: int = 4) -> list[dict]:
        return []

    def rebuild_index_registry(collection_name: str = "documents") -> dict:
        return {}

try:
    from terminal import terminal_bp
except Exception:
    from flask import Blueprint
    terminal_bp = Blueprint("terminal", __name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR.parent.parent / "checkpoints" / "final_model"))
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")
UPLINK_URL = os.environ.get("UPLINK_URL", "http://localhost:7000")
PORT = int(os.environ.get("PORT", "5000"))
ALLOWED = {".pdf", ".docx", ".doc", ".txt", ".md"}
REGISTRY_FILE = BASE_DIR / ".indexed_files.json"
VERSION = os.environ.get("NEURALAI_VERSION", "3.0")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.register_blueprint(terminal_bp)

INDEXED_FILES: dict[str, str] = {}
model = None
tokenizer = None
model_error: str | None = None

def load_registry() -> dict[str, str]:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_registry() -> None:
    REGISTRY_FILE.write_text(json.dumps(INDEXED_FILES, indent=2, sort_keys=True))


def model_device():
    if torch is None or model is None:
        return "cpu"
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cpu"


def model_type() -> str:
    adapter_files = [Path(MODEL_PATH) / "adapter_model.safetensors", Path(MODEL_PATH) / "adapter_model.bin"]
    if any(p.exists() for p in adapter_files):
        return "fine-tuned"
    if model is not None:
        return "base"
    if model_error:
        return "fallback"
    return "unknown"


def query_uplink(user_msg: str, conversation_history: list[dict]) -> str:
    if requests is None:
        return "[Uplink unavailable: requests dependency missing]"
    payload = {
        "task": user_msg,
        "context": {"conversation": conversation_history[-6:] if conversation_history else []},
    }
    try:
        resp = requests.post(f"{UPLINK_URL}/api/v1/zo/tasks", json=payload, timeout=25)
        data = resp.json()
        result = data.get("result", data.get("error", str(data)))
        if isinstance(result, dict):
            result = result.get("result", str(result))
        return str(result) if result else ""
    except Exception as exc:
        return f"[Agent error: {exc}]"


def load_model() -> None:
    global model, tokenizer, model_error
    if model is not None or model_error:
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        adapter_path = Path(MODEL_PATH)
        has_adapter = adapter_path.exists() and (
            (adapter_path / "adapter_model.safetensors").exists() or
            (adapter_path / "adapter_model.bin").exists()
        )

        if has_adapter:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto" if torch is not None and torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch is not None and torch.cuda.is_available() else torch.float32,
            )
            model = PeftModel.from_pretrained(model, str(adapter_path))
            print(f"[NeuralAI] Fine-tuned model loaded with LoRA adapter from {MODEL_PATH}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto" if torch is not None and torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch is not None and torch.cuda.is_available() else torch.float32,
            )
            print(f"[NeuralAI] Base model loaded: {MODEL_NAME}")

        model.eval()
        model_error = None
    except Exception as exc:
        model = None
        tokenizer = None
        model_error = str(exc)
        print(f"[NeuralAI] Model load failed: {exc}")


def build_doc_context(user_content: str, file_ids: list[str]) -> str:
    if not file_ids:
        return ""
    try:
        docs = query_documents(user_content, top_k=3)
    except Exception:
        docs = []
    if not docs:
        return ""
    chunks_text = "\n\n---\n\n".join(f"[From {d['source']}]: {d['content']}" for d in docs)
    return f"\n\nRelevant context from uploaded documents:\n{chunks_text}\n"


def build_prompt(messages: list[dict], user_content: str, doc_context: str) -> str:
    # Base system prompt - friendly assistant
    system_content = (
        "You are NeuralAI, a friendly AI assistant. "
        "Greet users warmly. Answer questions helpfully. "
        "Be natural and conversational. "
        "If the user asks about a file or document, you can help with that too."
    )
    # Add document context only when files are attached
    if doc_context:
        system_content = (
            "You are NeuralAI, a helpful AI assistant. "
            "The user has attached documents. Use the document context below to answer questions about those files accurately. "
            "For general questions, respond normally and naturally."
        ) + doc_context

    enriched_chat = [{"role": "system", "content": system_content}]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role in ("user", "assistant") and content:
            enriched_chat.append({"role": role, "content": content})
    if not enriched_chat or enriched_chat[-1]["role"] != "user":
        enriched_chat.append({"role": "user", "content": user_content})

    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(enriched_chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    prompt = []
    for msg in enriched_chat:
        prompt.append(f"{msg['role']}\n{msg['content']}")
    prompt.append("assistant")
    return "\n\n".join(prompt)


def answer_with_model(messages: list[dict], user_content: str, doc_context: str, max_new_tokens: int, temperature: float) -> str:
    load_model()
    if model is None or tokenizer is None or torch is None:
        base = "I'm up, but the local model backend isn't available yet."
        if doc_context:
            return base + " I did find uploaded document context, but I can't summarize it without the model."
        return base + f" You said: {user_content}"

    prompt = build_prompt(messages, user_content, doc_context)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        device = model_device()
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max(32, min(max_new_tokens, 512)),
                do_sample=temperature > 0,
                temperature=max(0.1, min(temperature, 1.5)),
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[-1]
        decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        if not decoded:
            decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return decoded or "I couldn't generate a reply right now."
    except Exception as exc:
        return f"[Model error: {exc}]"


def stream_words(text: str):
    words = text.split()
    if not words:
        yield f"data: {json.dumps({'content': text})}\n\n"
        return
    for word in words:
        yield f"data: {json.dumps({'content': word + ' '})}\n\n"
        time.sleep(0.01)


INDEXED_FILES = load_registry()
try:
    rebuild_index_registry()
except Exception:
    pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(
        {
            "model": MODEL_NAME,
            "model_type": model_type(),
            "device": model_device(),
            "version": VERSION,
            "rag": True,
            "uplink": "connected" if requests is not None else "offline",
            "indexed_files": len(INDEXED_FILES),
            "model_error": model_error,
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "version": VERSION})


@app.route("/api/files", methods=["GET"])
def list_files():
    return jsonify({"files": list(INDEXED_FILES.values()), "ids": list(INDEXED_FILES.keys())})


@app.route("/api/files/<file_id>", methods=["DELETE"])
def delete_file(file_id):
    if file_id not in INDEXED_FILES:
        return jsonify({"error": "File not found"}), 404
    filename = INDEXED_FILES[file_id]
    del INDEXED_FILES[file_id]
    save_registry()
    try:
        filepath = UPLOAD_FOLDER / filename
        if filepath.exists():
            filepath.unlink()
    except Exception:
        pass
    return jsonify({"success": True, "deleted": filename})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported type: {ext}"}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    result = index_document(str(filepath))
    file_id = result.get("file_id", hashlib.sha256(filename.encode()).hexdigest()[:16])
    INDEXED_FILES[file_id] = filename
    save_registry()

    return jsonify(
        {
            "success": True,
            "filename": filename,
            "file_id": file_id,
            "chunks": result.get("chunks", 0),
            "message": f'"{filename}" indexed — {result.get("chunks", 0)} chunks ready.',
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", []) or []
    prompt_only = data.get("prompt", "")
    max_new_tokens = int(data.get("max_tokens", 512))
    temperature = float(data.get("temperature", 0.7))
    file_ids = data.get("file_ids", []) or []

    def generate():
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "").strip()
                break
        user_content = last_user or prompt_only

        if not user_content:
            yield f"data: {json.dumps({'error': 'No message content'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # NEW ROUTING: Use clean router
        route, tool = neuralai_route(user_content)

        if route == "tool" and tool == "terminal":
            # Terminal execution - real shell, not uplink
            cmd = strip_terminal_prefix(user_content)
            yield f"data: {json.dumps({'content': f'[Terminal] Executing: {cmd}\\n'})}\n\n"
            # For now, return a message - real terminal integration via terminal_bp
            yield f"data: {json.dumps({'content': 'Use the Terminal tab for shell commands.\\n'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if route == "uplink":
            # Uplink for heavy tasks only
            yield f"data: {json.dumps({'content': '[Neural Uplink] Routing to agent network...\\n'})}\n\n"
            agent_response = query_uplink(user_content, messages)
            for chunk in stream_words(agent_response):
                yield chunk
            yield "data: [DONE]\n\n"
            return

        # DEFAULT: Local model
        doc_context = build_doc_context(user_content, file_ids)
        answer = answer_with_model(messages, user_content, doc_context, max_new_tokens, temperature)
        for chunk in stream_words(answer):
            yield chunk
        yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
