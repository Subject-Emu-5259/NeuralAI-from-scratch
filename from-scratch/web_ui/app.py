#!/usr/bin/env python3
"""
NeuralAI - Flask Web UI Backend
Serves the NeuralAI chat interface with local model inference + RAG.
"""
import os, json, time, torch, hashlib
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from rag import index_document, query_documents

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(BASE_DIR), "..", "..", "checkpoints", "final_model"))
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")

ALLOWED = {".pdf", ".docx", ".doc", ".txt", ".md"}
INDEXED_FILES = {}  # session-level tracking

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model Loading ────────────────────────────────────────────────
def load_model():
    global model, tokenizer
    fine_safetensors = os.path.join(MODEL_PATH, "adapter_model.safetensors")
    fine_bin = os.path.join(MODEL_PATH, "adapter_model.bin")

    if os.path.exists(fine_safetensors) or os.path.exists(fine_bin):
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
            print(f"[NeuralAI] Fine-tuned load failed: {e}, falling back to base model")

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


# ── Routes ─────────────────────────────────────────────────────
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

    # Index it
    result = index_document(filepath)
    file_id = result.get("file_id", hashlib.sha256(filename.encode()).hexdigest()[:16])
    INDEXED_FILES[file_id] = filename

    return jsonify({
        "success": True,
        "filename": filename,
        "file_id": file_id,
        "chunks": result.get("chunks", 0),
        "message": f"📄 \"{filename}\" indexed — {result.get('chunks', 0)} chunks ready for questions."
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
            # Get last user message
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

            # Build chat context
            chat = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if role in ("system", "user", "assistant") and content:
                    chat.append({"role": role, "content": content})

            if not chat or chat[-1]["role"] != "user":
                chat.append({"role": "user", "content": user_content})

            # Retrieve document context if files are attached
            doc_context = ""
            if file_ids:
                for fid in file_ids:
                    if fid in INDEXED_FILES:
                        docs = query_documents(user_content, top_k=3)
                        if docs:
                            chunks = "\n\n---\n\n".join(
                                f"[From {d['source']}]: {d['content']}" for d in docs
                            )
                            doc_context = f"\n\nRelevant context from uploaded documents:\n{chunks}\n\n"
                        break

            # System prompt with document context
            system_content = (
                "You are NeuralAI, a helpful and friendly AI assistant built from SmolLM2-360M-Instruct. "
                "You can read and analyze PDF, DOCX, TXT, and MD files when uploaded. "
                "You support code generation in Python, JavaScript, SQL, and more. "
                "You can design REST APIs, explain concepts, help with data science and machine learning. "
                "Be helpful, concise, and friendly. Answer based on the conversation and any provided documents."
            )
            if doc_context:
                system_content += doc_context

            # Rebuild chat with doc-enriched system
            enriched_chat = [{"role": "system", "content": system_content}]
            for msg in chat:
                if msg["role"] != "system":
                    enriched_chat.append(msg)

            # Format with tokenizer chat template
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
                response = "I'm not sure how to respond to that. Could you rephrase?"

            for word in response.split():
                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                time.sleep(0.015)

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/status", methods=["GET"])
def status():
    lazy_load()
    model_type = "fine-tuned" if model is not None else "base"
    return jsonify({
        "model": MODEL_NAME,
        "model_type": model_type,
        "device": device,
        "loaded": model is not None,
        "version": "2.1",
        "rag": True,
        "indexed_files": len(INDEXED_FILES)
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[NeuralAI] Starting on http://0.0.0.0:{port}")
    print(f"[NeuralAI] Device: {device}")
    app.run(host="0.0.0.0", port=port, debug=False)
