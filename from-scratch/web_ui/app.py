# NeuralAI Web UI v4.0 - Enhanced with Persistence, Memory, and Settings
import hashlib
import json
import os
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context, g
from werkzeug.utils import secure_filename

# NeuralAI Engine - Router + Local Model + Uplink + Tools
try:
    from neuralai_router import neuralai_route
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False
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

# Database path
DATABASE = BASE_DIR / "neuralai.db"

MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR.parent.parent / "checkpoints" / "final_model"))
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")
UPLINK_URL = os.environ.get("UPLINK_URL", "http://localhost:7000")
PORT = int(os.environ.get("PORT", "5000"))
ALLOWED = {".pdf", ".docx", ".doc", ".txt", ".md"}
REGISTRY_FILE = BASE_DIR / ".indexed_files.json"
VERSION = os.environ.get("NEURALAI_VERSION", "4.0")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.register_blueprint(terminal_bp)

INDEXED_FILES: dict[str, str] = {}
model = None
tokenizer = None
model_error: str | None = None


# ========================================
# DATABASE LAYER
# ========================================

def get_db():
    """Get database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(str(DATABASE))
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Close database connection."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize database tables."""
    db = get_db()
    db.executescript("""
        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            message_count INTEGER DEFAULT 0
        );
        
        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
        
        -- User settings table
        CREATE TABLE IF NOT EXISTS user_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        
        -- Memory facts table
        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TEXT NOT NULL,
            importance INTEGER DEFAULT 0
        );
        
        -- Model rules table
        CREATE TABLE IF NOT EXISTS model_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );
        
        -- Preference data table for DPO
        CREATE TABLE IF NOT EXISTS preference_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            chosen TEXT NOT NULL,
            rejected TEXT,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'user_feedback',
            created_at TEXT NOT NULL
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_memory_category ON memory_facts(category);
    """)
    db.commit()
    
    # Initialize default settings if not exist
    defaults = {
        "user_bio": "A curious user exploring AI capabilities.",
        "model_temperature": "0.7",
        "model_max_tokens": "512",
        "model_name": "SmolLM2-360M-Instruct",
        "theme": "dark",
        "auto_save": "true",
    }
    now = datetime.utcnow().isoformat()
    for key, value in defaults.items():
        try:
            db.execute(
                "INSERT OR IGNORE INTO user_settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now)
            )
        except:
            pass
    db.commit()


def generate_conv_id() -> str:
    """Generate unique conversation ID."""
    import uuid
    return f"conv_{uuid.uuid4().hex[:12]}"


# ========================================
# SETTINGS API
# ========================================

@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Get all user settings."""
    db = get_db()
    rows = db.execute("SELECT key, value FROM user_settings").fetchall()
    settings = {row["key"]: row["value"] for row in rows}
    return jsonify({"settings": settings})


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update user settings."""
    data = request.get_json(silent=True) or {}
    db = get_db()
    now = datetime.utcnow().isoformat()
    
    for key, value in data.items():
        db.execute(
            "INSERT OR REPLACE INTO user_settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), now)
        )
    db.commit()
    return jsonify({"success": True, "updated": list(data.keys())})


@app.route("/api/settings/<key>", methods=["GET"])
def get_setting(key):
    """Get single setting."""
    db = get_db()
    row = db.execute("SELECT value FROM user_settings WHERE key = ?", (key,)).fetchone()
    if row:
        return jsonify({"key": key, "value": row["value"]})
    return jsonify({"error": "Setting not found"}), 404


# ========================================
# MEMORY API
# ========================================

@app.route("/api/memory", methods=["GET"])
def get_memory():
    """Get all memory facts."""
    db = get_db()
    rows = db.execute(
        "SELECT id, fact, category, importance, created_at FROM memory_facts ORDER BY importance DESC, created_at DESC"
    ).fetchall()
    facts = [dict(row) for row in rows]
    return jsonify({"facts": facts})


@app.route("/api/memory", methods=["POST"])
def add_memory():
    """Add a memory fact."""
    data = request.get_json(silent=True) or {}
    fact = data.get("fact", "").strip()
    category = data.get("category", "general")
    importance = data.get("importance", 0)
    
    if not fact:
        return jsonify({"error": "Fact is required"}), 400
    
    db = get_db()
    now = datetime.utcnow().isoformat()
    cursor = db.execute(
        "INSERT INTO memory_facts (fact, category, importance, created_at) VALUES (?, ?, ?, ?)",
        (fact, category, importance, now)
    )
    db.commit()
    return jsonify({"success": True, "id": cursor.lastrowid, "fact": fact})


@app.route("/api/memory/<int:fact_id>", methods=["DELETE"])
def delete_memory(fact_id):
    """Delete a memory fact."""
    db = get_db()
    db.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))
    db.commit()
    return jsonify({"success": True})


# ========================================
# RULES API
# ========================================

@app.route("/api/rules", methods=["GET"])
def get_rules():
    """Get all model rules."""
    db = get_db()
    rows = db.execute("SELECT id, rule, is_active, created_at FROM model_rules ORDER BY created_at DESC").fetchall()
    rules = [dict(row) for row in rows]
    return jsonify({"rules": rules})


@app.route("/api/rules", methods=["POST"])
def add_rule():
    """Add a model rule."""
    data = request.get_json(silent=True) or {}
    rule = data.get("rule", "").strip()
    is_active = data.get("is_active", 1)
    
    if not rule:
        return jsonify({"error": "Rule is required"}), 400
    
    db = get_db()
    now = datetime.utcnow().isoformat()
    cursor = db.execute(
        "INSERT INTO model_rules (rule, is_active, created_at) VALUES (?, ?, ?)",
        (rule, is_active, now)
    )
    db.commit()
    return jsonify({"success": True, "id": cursor.lastrowid})


@app.route("/api/rules/<int:rule_id>", methods=["DELETE"])
def delete_rule(rule_id):
    """Delete a model rule."""
    db = get_db()
    db.execute("DELETE FROM model_rules WHERE id = ?", (rule_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/rules/<int:rule_id>/toggle", methods=["POST"])
def toggle_rule(rule_id):
    """Toggle rule active state."""
    db = get_db()
    row = db.execute("SELECT is_active FROM model_rules WHERE id = ?", (rule_id,)).fetchone()
    if not row:
        return jsonify({"error": "Rule not found"}), 404
    
    new_state = 0 if row["is_active"] else 1
    db.execute("UPDATE model_rules SET is_active = ? WHERE id = ?", (new_state, rule_id))
    db.commit()
    return jsonify({"success": True, "is_active": new_state})


# ========================================
# CONVERSATIONS API
# ========================================

@app.route("/api/preference", methods=["POST"])
def add_preference():
    """Add a chosen/rejected preference pair for DPO."""
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    chosen = data.get("chosen", "").strip()
    rejected = data.get("rejected", "").strip()
    category = data.get("category", "general")
    
    if not prompt or not chosen:
        return jsonify({"error": "Prompt and chosen response required"}), 400
        
    db = get_db()
    now = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO preference_data (prompt, chosen, rejected, category, created_at) VALUES (?, ?, ?, ?, ?)",
        (prompt, chosen, rejected, category, now)
    )
    db.commit()
    return jsonify({"success": True})


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    """List all conversations."""
    db = get_db()
    rows = db.execute(
        "SELECT id, title, created_at, updated_at, message_count FROM conversations ORDER BY updated_at DESC LIMIT 50"
    ).fetchall()
    conversations = [dict(row) for row in rows]
    return jsonify({"conversations": conversations})


@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    """Create new conversation."""
    data = request.get_json(silent=True) or {}
    title = data.get("title", "New Chat")
    
    conv_id = generate_conv_id()
    now = datetime.utcnow().isoformat()
    
    db = get_db()
    db.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, 0)",
        (conv_id, title, now, now)
    )
    db.commit()
    return jsonify({"success": True, "id": conv_id, "title": title})


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def get_conversation(conv_id):
    """Get conversation with messages."""
    db = get_db()
    
    conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    messages = db.execute(
        "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    ).fetchall()
    
    return jsonify({
        "conversation": dict(conv),
        "messages": [dict(m) for m in messages]
    })


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    """Delete conversation and its messages."""
    db = get_db()
    db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/conversations/<conv_id>/rename", methods=["POST"])
def rename_conversation(conv_id):
    """Rename conversation."""
    data = request.get_json(silent=True) or {}
    title = data.get("title", "Untitled")
    
    db = get_db()
    db.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", 
               (title, datetime.utcnow().isoformat(), conv_id))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/conversations/<conv_id>/messages", methods=["POST"])
def add_message(conv_id):
    """Add message to conversation."""
    data = request.get_json(silent=True) or {}
    role = data.get("role", "user")
    content = data.get("content", "")
    
    if not content:
        return jsonify({"error": "Content required"}), 400
    
    db = get_db()
    now = datetime.utcnow().isoformat()
    
    # Add message
    db.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, now)
    )
    
    # Update conversation stats
    db.execute(
        "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
        (now, conv_id)
    )
    
    # Auto-rename if first user message
    if role == "user":
        count = db.execute("SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role = 'user'", (conv_id,)).fetchone()
        if count["cnt"] == 1:
            title = content[:40] + ("..." if len(content) > 40 else "")
            db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
    
    db.commit()
    return jsonify({"success": True})


# ========================================
# FILE SYSTEM HELPERS
# ========================================

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


def get_system_prompt() -> str:
    """Build system prompt from user bio, memory, and rules."""
    db = get_db()
    
    # Get user bio
    bio_row = db.execute("SELECT value FROM user_settings WHERE key = 'user_bio'").fetchone()
    user_bio = bio_row["value"] if bio_row else ""
    
    # Get active rules
    rules_rows = db.execute("SELECT rule FROM model_rules WHERE is_active = 1").fetchall()
    rules = [r["rule"] for r in rules_rows]
    
    # Get top memory facts
    memory_rows = db.execute(
        "SELECT fact FROM memory_facts ORDER BY importance DESC LIMIT 10"
    ).fetchall()
    memories = [m["fact"] for m in memory_rows]
    
    # Build prompt
    base = "You are NeuralAI, a helpful AI assistant."
    
    if user_bio:
        base += f"\n\n## User Profile\n{user_bio}"
    
    if memories:
        base += "\n\n## What you know about the user\n" + "\n".join(f"- {m}" for m in memories)
    
    if rules:
        base += "\n\n## Behavioral Guidelines\n" + "\n".join(f"- {r}" for r in rules)
    
    return base


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
    # Get dynamic system prompt
    system_content = get_system_prompt()
    
    # Add document context if files attached
    if doc_context:
        system_content += "\n\n" + doc_context

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
    # Skip model loading - use fallback response
    base = "I'm online and ready to help! The local model is currently unavailable due to memory constraints."
    if doc_context:
        return base + f" I found some document context but can't process it right now. Your message: {user_content}"
    return base + f" You said: {user_content}"


def stream_words(text: str):
    """Stream text word by word, preserving newlines."""
    # Split by lines to preserve structure
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line:
            # Stream words in the line
            words = line.split()
            for word in words:
                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                time.sleep(0.005)
        # Add newline after each line except the last empty one
        if i < len(lines) - 1:
            yield f"data: {json.dumps({'content': '\\n'})}\n\n"


INDEXED_FILES = load_registry()
try:
    rebuild_index_registry()
except Exception:
    pass


# ========================================
# ROUTES
# ========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/api/status", methods=["GET"])
def status():
    # Check if Uplink Gateway (port 8000) is healthy
    uplink_status = "offline"
    try:
        r = requests.get("http://localhost:8000/health", timeout=1)
        if r.status_code == 200:
            uplink_status = "connected"
    except:
        pass

    return jsonify(
        {
            "model": MODEL_NAME,
            "model_type": model_type(),
            "device": model_device(),
            "version": VERSION,
            "rag": True,
            "uplink": uplink_status,
            "indexed_files": len(INDEXED_FILES),
            "model_error": model_error,
            "features": ["memory", "rules", "settings", "conversations"],
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
    conv_id = data.get("conversation_id")  # NEW: conversation ID for persistence
    
    # Get settings from DB
    db = get_db()
    temp_row = db.execute("SELECT value FROM user_settings WHERE key = 'model_temperature'").fetchone()
    tokens_row = db.execute("SELECT value FROM user_settings WHERE key = 'model_max_tokens'").fetchone()
    
    max_new_tokens = int(data.get("max_tokens", tokens_row["value"] if tokens_row else 512))
    temperature = float(data.get("temperature", temp_row["value"] if temp_row else 0.7))
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

        # Save user message to conversation
        if conv_id:
            now = datetime.utcnow().isoformat()
            db_inner = get_db()
            db_inner.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, "user", user_content, now)
            )
            db_inner.execute(
                "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                (now, conv_id)
            )
            # Auto-rename if first message
            count = db_inner.execute("SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role = 'user'", (conv_id,)).fetchone()
            if count["cnt"] == 1:
                title = user_content[:40] + ("..." if len(user_content) > 40 else "")
                db_inner.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
            db_inner.commit()

        # NEW ROUTING: Use clean router
        route, tool = neuralai_route(user_content)

        if route == "tool" and tool == "terminal":
            cmd = strip_terminal_prefix(user_content)
            yield f"data: {json.dumps({'content': f'[Terminal] Executing: {cmd}\\n'})}\n\n"
            yield f"data: {json.dumps({'content': 'Use the Terminal tab for shell commands.\\n'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if route == "uplink":
            yield f"data: {json.dumps({'content': '[Neural Uplink] Routing to agent network...\\n'})}\n\n"
            agent_response = query_uplink(user_content, messages)
            for chunk in stream_words(agent_response):
                yield chunk
            
            # Save assistant response
            if conv_id:
                now = datetime.utcnow().isoformat()
                db_inner = get_db()
                db_inner.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conv_id, "assistant", agent_response, now)
                )
                db_inner.execute(
                    "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                    (now, conv_id)
                )
                db_inner.commit()
            
            yield "data: [DONE]\n\n"
            return

        # DEFAULT: Local model
        doc_context = build_doc_context(user_content, file_ids)
        answer = answer_with_model(messages, user_content, doc_context, max_new_tokens, temperature)
        for chunk in stream_words(answer):
            yield chunk
        
        # Save assistant response
        if conv_id:
            now = datetime.utcnow().isoformat()
            db_inner = get_db()
            db_inner.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, "assistant", answer, now)
            )
            db_inner.execute(
                "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                (now, conv_id)
            )
            db_inner.commit()
        
        yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


# Initialize database on startup
with app.app_context():
    init_db()
    print(f"[NeuralAI] Database initialized at {DATABASE}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
